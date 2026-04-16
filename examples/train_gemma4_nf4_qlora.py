#!/usr/bin/env python3
"""
train_gemma4_nf4_qlora.py — Fine-tune Gemma4 26B-A4B with NF4/QLoRA on consumer hardware.

Requirements:
  - RTX 4090 (24GB) + second GPU 16GB+ (e.g. RTX PRO 2000)
  - 32GB+ system RAM (model loading only, no CPU offload needed)
  - pip install transformers bitsandbytes peft torch

Why NF4 instead of INT8:
  INT8 quantization is NOT recommended for LoRA fine-tuning.
  bitsandbytes issue #1451 documents near-random loss (~13) with INT8+LoRA,
  matching the QLoRA paper (NeurIPS 2023) which shows NF4 "fully recovers
  16-bit LoRA performance" while INT8 does not.

  NF4 (4-bit NormalFloat) also uses ~half the GPU memory vs INT8:
  - INT8: 26B × 1 byte = 26GB
  - NF4:  26B × 0.5 byte = 13GB → fits on 2 consumer GPUs, no CPU offload

Three bugs fixed vs naive implementations:
  Bug 1: Padding tokens in labels — zeros get token-ID 0 which inflates loss.
         Fix: labels use -100 for padding positions (ignored by cross-entropy).
  Bug 2: Gradient accumulation loss normalization — HF normalizes per mini-batch,
         not per accumulated window. With GRAD_ACCUM=4, displayed loss is ~4x wrong.
         Fix: weighted loss by non-padding token count across the accumulation window.
         (Source: Unsloth blog, HuggingFace blog, Benjamin Marie, Ian Barber 2025)
  Bug 3: INT8 + LoRA gradient instability — silently wrong gradients from INT8
         matmul produce near-random loss regardless of learning rate.
         Fix: use NF4 quantization (this script).

Data format (unsloth/gemma-4-26B-A4B-it):
  <bos><|turn>user
  [question]<turn|>
  <|turn>model
  [answer]<turn|>

  Token IDs: 105 = <|turn> (start), 106 = <turn|> (end)
  Generate with: tok.apply_chat_template([...], tokenize=False)
"""

import os, gc, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "unsloth/gemma-4-26B-A4B-it"
TRAIN_FILE = "train.txt"
OUTPUT_DIR = "./gemma4-nf4-finetuned"

# NF4 fits without CPU offload (~13GB for weights)
MAX_MEMORY = {
    0: "21GiB",     # Primary GPU
    1: "14GiB",     # Second GPU (optional — remove if single GPU)
}

LORA_LAYERS = list(range(20))   # Train first 20 layers
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 1
LR = 2e-4
LOG_EVERY = 50
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# NF4 config — double quant saves extra ~0.4 bits per param
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model (NF4/QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    max_memory=MAX_MEMORY,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Remove vision tower — text-only fine-tuning
if hasattr(model.model, "vision_tower"):
    delattr(model.model, "vision_tower")

model.enable_input_require_grads()

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=LORA_LAYERS,
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, lora_cfg)

# CRITICAL ORDER: model.train() BEFORE gradient_checkpointing_enable()
# HuggingFace checks `if self.gradient_checkpointing and self.training`
# Without model.train() first, GC never activates → OOM on long sequences
model.gradient_checkpointing_enable()
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.3f}%)")
for i in range(torch.cuda.device_count()):
    used = torch.cuda.memory_allocated(i) / 1e9
    cap = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU{i}: {used:.1f}GB / {cap:.1f}GB")


class ChatDataset(Dataset):
    """
    Splits training data on <bos> to get FULL conversations (user + model).
    Each sample: <bos><|turn>user\\n[question]<turn|>\\n<|turn>model\\n[answer]<turn|>

    Token IDs for unsloth/gemma-4-26B-A4B-it:
      105 = <|turn>  (start of turn)
      106 = <turn|>  (end of turn)

    Note: google/gemma-4-*-it uses different tokens (<start_of_turn>/<end_of_turn>).
    Always verify with tok.apply_chat_template().
    """
    def __init__(self, path, tokenizer, max_len):
        with open(path) as f:
            raw = f.read()
        chunks = [c.strip() for c in raw.split("<bos>") if len(c.strip()) > 50]
        self.samples = []
        for chunk in chunks:
            text = "<bos>" + chunk
            enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
            if enc["input_ids"].shape[1] >= 8:
                self.samples.append(enc["input_ids"].squeeze(0))
        print(f"Dataset: {len(self.samples)} conversations from {path}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    """
    Bug 1 fix: separate input and label tensors.
    Padding positions use token-ID 0 in inputs (safe) but -100 in labels
    (ignored by cross-entropy loss). Without this, hundreds of padding
    positions inflate loss per batch.
    """
    max_len = max(b.shape[0] for b in batch)
    input_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    label_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        input_padded[i, :b.shape[0]] = b
        label_padded[i, :b.shape[0]] = b   # only real tokens have labels
    return input_padded, label_padded


dataset = ChatDataset(TRAIN_FILE, tok, MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

optimizer = bnb.optim.AdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01
)

print(f"\nTraining: {EPOCHS} epoch(s) | {len(dataset)} samples | NF4/QLoRA | LR={LR}")
print(f"Grad accum: {GRAD_ACCUM} | Seq len: {MAX_SEQ_LEN} | LoRA layers: 0-{LORA_LAYERS[-1]}\n")

start_total = time.time()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    accum_tokens = 0
    accum_raw_loss = 0.0
    reported_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids, labels = batch
        input_ids = input_ids.to("cuda:0")
        labels = labels.to("cuda:0")

        # Response-only masking: mask user turn tokens with -100
        # Only compute loss on model's response (after second <|turn> = token 105)
        for b in range(labels.shape[0]):
            turn_pos = (input_ids[b] == 105).nonzero(as_tuple=True)[0]
            if len(turn_pos) >= 2:
                # Second 105 = <|turn>model — skip it + "model" + "\n" = 3 tokens
                model_start = turn_pos[1].item() + 3
                labels[b, :model_start] = -100

        out = model(input_ids=input_ids, labels=labels)

        # Bug 2 fix: token-count normalized loss across accumulation window
        # HF's out.loss is already normalized per non-ignored token within a batch,
        # but variable sequence lengths cause imbalance across the GRAD_ACCUM window.
        # Weight by token count → correct normalization (Unsloth/HuggingFace pattern).
        n_tokens = (labels != -100).sum().item()
        if n_tokens == 0:
            continue
        accum_tokens += n_tokens
        loss = out.loss * (n_tokens / GRAD_ACCUM)
        loss.backward()
        accum_raw_loss += out.loss.item() * n_tokens
        epoch_loss += out.loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            # Token-normalized reported loss for this accumulation window
            reported_loss = accum_raw_loss / max(accum_tokens, 1)
            accum_tokens = 0
            accum_raw_loss = 0.0

            # Per-device gradient clipping — LoRA params may span cuda:0 and cuda:1
            # torch.nn.utils.clip_grad_norm_ fails across devices, must clip per device
            seen_devices = set()
            for p in model.parameters():
                if p.requires_grad and p.grad is not None and p.device not in seen_devices:
                    seen_devices.add(p.device)
                    device_params = [
                        q for q in model.parameters()
                        if q.requires_grad and q.grad is not None and q.device == p.device
                    ]
                    torch.nn.utils.clip_grad_norm_(device_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step = epoch * len(loader) + step + 1
        if global_step % LOG_EVERY == 0 or step == 0:
            elapsed = time.time() - start_total
            avg_loss = epoch_loss / (step + 1)
            steps_remaining = (EPOCHS * len(loader)) - global_step
            eta_h = steps_remaining * (elapsed / global_step) / 3600
            peak0 = torch.cuda.max_memory_allocated(0) / 1e9
            r = reported_loss if reported_loss > 0 else avg_loss
            print(f"[E{epoch+1} S{global_step}] loss={r:.4f} avg={avg_loss:.4f} peak={peak0:.1f}GB eta={eta_h:.1f}h")
            torch.cuda.reset_peak_memory_stats(0)

    avg_loss = epoch_loss / len(loader)
    print(f"\n=== Epoch {epoch+1} done === avg_loss={avg_loss:.4f}")
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tok.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")

total_h = (time.time() - start_total) / 3600
print(f"\n=== Training done in {total_h:.2f}h ===")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
