#!/usr/bin/env python3
"""
train_gemma4_26b_consumer.py — Fine-tune Gemma4 26B-A4B on consumer hardware.

Requirements:
  - RTX 4090 (24GB) or equivalent — primary compute GPU
  - 55GB+ system RAM — for CPU offload of overflow layers
  - All 10 patches from consumer-llm-patches applied
  - pip install transformers bitsandbytes peft torch

This script trains Gemma4 26B (a 26B MoE model, 4B active) with:
  - BnB INT8 quantization (halves memory vs BF16)
  - PEFT LoRA (only 0.05% of parameters trainable)
  - Gradient Checkpointing (avoids storing all activations)
  - CPU offload for layers that don't fit on GPU

Measured performance: ~6.25s/step at 512 tokens on RTX 4090 + 60GB CPU RAM.
7000 samples × 1 epoch ≈ 12-13 hours.

CRITICAL: Match your training data format to the tokenizer.
Use tokenizer.apply_chat_template() to generate training data:

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/gemma-4-26B-A4B-it")
    sample = tok.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        tokenize=False
    )
    # → '<bos><|turn>user\n[q]<turn|>\n<|turn>model\n[a]<turn|>\n'

Data format for unsloth/gemma-4-26B-A4B-it (one conversation per block):
    <bos><|turn>user
    [question]<turn|>
    <|turn>model
    [answer]<turn|>

Note: google/gemma-4-9b-it uses different tokens (<start_of_turn>/<end_of_turn>).
Always check tok.apply_chat_template() for the correct format for your model.
"""

import os, gc, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "google/gemma-4-9b-it"          # or "unsloth/gemma-4-26B-A4B-it"
TRAIN_FILE = "train.txt"                # Gemma4 chat format (see above)
OUTPUT_DIR = "./gemma4-finetuned"

# Memory allocation — adjust to your hardware
# For RTX 4090 (24GB) + second GPU (optional) + CPU RAM:
MAX_MEMORY = {
    0: "22GiB",                          # Primary GPU
    # 1: "15GiB",                        # Second GPU (optional)
    "cpu": "60GiB",                      # CPU RAM — replaces disk offload
}

LORA_LAYERS = list(range(20))           # Train first 20 layers (on primary GPU)
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 1
LR = 2e-4
LOG_EVERY = 50
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Quantization config
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # Needed for CPU offload
)

print("Loading model (INT8 + CPU offload)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    max_memory=MAX_MEMORY,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Remove vision tower (we're doing text-only fine-tuning)
if hasattr(model.model, "vision_tower"):
    delattr(model.model, "vision_tower")

model.enable_input_require_grads()

# LoRA config
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=LORA_LAYERS,
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, lora_cfg)

# CRITICAL: model.train() MUST be called BEFORE gradient_checkpointing_enable()
# HuggingFace checks `if self.gradient_checkpointing and self.training`
# Without model.train(), GC never activates even after being "enabled"
# → state.CB accumulates for all layers → OOM
model.gradient_checkpointing_enable()
model.train()  # ← THIS IS THE KEY. Do not remove.

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.3f}%)")


# Dataset — splits on <bos> to get FULL conversations (user + model response)
#
# IMPORTANT: Match your training data format to the model's tokenizer:
#   - google/gemma-4-*-it uses: <start_of_turn>user ... <end_of_turn>
#   - unsloth/gemma-4-* uses:   <|turn>user ... <turn|>
#
# Use tokenizer.apply_chat_template() to generate training data,
# then split on <bos> to get one full conversation per sample.
#
# Training data format (one sample):
#   <bos><|turn>user\n[question]<turn|>\n<|turn>model\n[answer]<turn|>
class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path) as f:
            raw = f.read()
        # Split on <bos> → each chunk is one full conversation (user + model response)
        chunks = [c.strip() for c in raw.split("<bos>") if len(c.strip()) > 50]
        self.samples = []
        for chunk in chunks:
            text = "<bos>" + chunk  # re-add BOS
            enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
            if enc["input_ids"].shape[1] >= 8:
                self.samples.append(enc["input_ids"].squeeze(0))
        print(f"Dataset: {len(self.samples)} full conversations from {path}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    max_len = max(b.shape[0] for b in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, :b.shape[0]] = b
    return padded


dataset = ChatDataset(TRAIN_FILE, tok, MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

optimizer = bnb.optim.AdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01
)

print(f"\nTraining: {EPOCHS} epoch(s) | {len(dataset)} samples | batch={BATCH_SIZE} accum={GRAD_ACCUM}")
print(f"Estimated: {len(dataset) * 6.25 / 3600:.1f}h @ 6.25s/step\n")

start = time.time()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch.to("cuda:0")
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        epoch_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

        global_step = epoch * len(loader) + step + 1
        if global_step % LOG_EVERY == 0 or step == 0:
            elapsed = time.time() - start
            avg_loss = epoch_loss / (step + 1)
            remaining = (EPOCHS * len(loader) - global_step) * (elapsed / global_step)
            print(f"[E{epoch+1} S{global_step}] loss={avg_loss:.4f} eta={remaining/3600:.1f}h")
            torch.cuda.reset_peak_memory_stats(0)

    avg_loss = epoch_loss / len(loader)
    print(f"\n=== Epoch {epoch+1} done === avg_loss={avg_loss:.4f}")
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tok.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")

total_h = (time.time() - start) / 3600
print(f"\n=== Training done in {total_h:.2f}h ===")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
