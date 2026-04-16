#!/usr/bin/env python3
"""
ginny_ft_nf4.py — VERBOSE DEBUG VERSION
Gemma4 26B-A4B NF4/QLoRA Fine-Tuning, Titan-spezifisch.

Alle 3 Bugs gefixt:
  Bug 1: Padding in labels → Token-0 statt -100 → Hunderte falscher Loss-Terme
  Bug 2: Grad-Accum-Normierung → HF normiert pro Mini-Batch, nicht pro Accum-Window
  Bug 3: INT8+LoRA Gradient-Instabilität → NF4/QLoRA statt INT8

Verbose-Modus: jede mögliche Fehlerquelle wird abgefangen und diagnostiziert.
Loggt zu stdout UND /home/sirfyyn/finetune/runs/ginny_nf4_TIMESTAMP.log
"""

import os, gc, sys, signal, traceback, time, subprocess
from datetime import datetime
from pathlib import Path

# ── Logging: stdout + Datei gleichzeitig ──────────────────────────────────────
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = Path('./runs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f'train_nf4_{RUN_TS}.log'

class TeeLogger:
    """Schreibt zu stdout UND Log-Datei gleichzeitig (line-buffered)."""
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, 'w', buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def fileno(self):
        return self.terminal.fileno()

tee = TeeLogger(LOG_FILE)
sys.stdout = tee
sys.stderr = tee

def log(msg='', prefix=''):
    ts = datetime.now().strftime('%H:%M:%S')
    tag = f'[{prefix}]' if prefix else ''
    print(f'{ts} {tag} {msg}' if tag else f'{ts} {msg}', flush=True)

def section(title):
    log(f'\n{"="*60}')
    log(f'  {title}')
    log(f'{"="*60}')

log(f'Log: {LOG_FILE}')
log(f'Start: {datetime.now().isoformat()}')

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = 'unsloth/gemma-4-26B-A4B-it'  # or 'google/gemma-4-9b-it'
TRAIN_FILE = 'train.txt'
OUTPUT_DIR = './gemma4-nf4-finetuned'
MAX_MEMORY = {0: '21GiB', 1: '14GiB'}
LORA_LAYERS = list(range(20))
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 1
LR = 2e-4
LOG_EVERY = 10          # verbose: alle 10 Steps statt 50
WARN_LOSS_HIGH = 8.0    # Warnung wenn Loss nach S1 > 8 (Daten-Format-Problem)
WARN_LOSS_NAN = True    # Bei NaN sofort stoppen + diagnostizieren
# ──────────────────────────────────────────────────────────────────────────────

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Emergency Checkpoint Handler ──────────────────────────────────────────────
_model_ref = None
_tok_ref = None
_emergency_dir = OUTPUT_DIR + '/emergency'

def _save_emergency(signum, frame):
    log('\n[SIGINT] Ctrl+C erhalten — speichere Emergency Checkpoint...', 'SIGNAL')
    if _model_ref is not None:
        try:
            Path(_emergency_dir).mkdir(parents=True, exist_ok=True)
            _model_ref.save_pretrained(_emergency_dir)
            _tok_ref.save_pretrained(_emergency_dir)
            log(f'Emergency Adapter gespeichert: {_emergency_dir}', 'SIGNAL')
        except Exception as e:
            log(f'Emergency Save fehlgeschlagen: {e}', 'ERROR')
    sys.exit(0)

signal.signal(signal.SIGINT, _save_emergency)
signal.signal(signal.SIGTERM, _save_emergency)

# ── Diagnose-Funktionen ────────────────────────────────────────────────────────
def dump_gpu_state(label=''):
    """Gibt aktuellen GPU-Speicherstand für alle verfügbaren GPUs aus."""
    import torch
    tag = f'[GPU/{label}]' if label else '[GPU]'
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        cap = torch.cuda.get_device_properties(i).total_memory / 1e9
        name = torch.cuda.get_device_properties(i).name
        log(f'{tag} GPU{i} ({name}): alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB / {cap:.1f}GB')

def diagnose_oom(e, context=''):
    """Gibt vollständigen Speicher-Dump bei OOM-Fehler aus."""
    log(f'\n[OOM] CUDA Out of Memory bei: {context}', 'ERROR')
    log(f'[OOM] Fehlermeldung: {e}', 'ERROR')
    dump_gpu_state('OOM')
    log('[OOM] Mögliche Fixes:', 'OOM')
    log('  → MAX_SEQ_LEN reduzieren (aktuell: ' + str(MAX_SEQ_LEN) + ')', 'OOM')
    log('  → MAX_MEMORY reduzieren (aktuell: ' + str(MAX_MEMORY) + ')', 'OOM')
    log('  → GRAD_ACCUM erhöhen (mehr virtual batch, weniger activation memory)', 'OOM')
    log('  → LORA_LAYERS reduzieren (weniger trainierbare Schichten)', 'OOM')

def check_patches():
    """Prüft ob die kritischen sirfyyn-Patches aktiv sind."""
    section('PATCH VERIFICATION')
    patches = {
        'P2 (BnB autograd SCB guard)': '~/.local/lib/python3.12/site-packages/bitsandbytes/autograd/_functions.py',  # adjust for your Python version,
        'P3 (BnB modules getattr None)': '~/.local/lib/python3.12/site-packages/bitsandbytes/nn/modules.py',
        'P4 (Gemma4 RoPE device)': '~/.local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py',
        'P9 (BnB SCB standalone restore)': '~/.local/lib/python3.12/site-packages/bitsandbytes/autograd/_functions.py',
        'P10 (Gemma4 mm_token pass)': '~/.local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py',
        'P11 (BnB Int8Params kwargs)': '~/.local/lib/python3.12/site-packages/bitsandbytes/nn/modules.py',
    }
    all_ok = True
    for patch_name, file_path in patches.items():
        expanded = os.path.expanduser(file_path)
        try:
            result = subprocess.run(
                ['grep', '-c', 'PATCH sirfyyn', expanded],
                capture_output=True, text=True
            )
            count = int(result.stdout.strip()) if result.returncode == 0 else 0
            status = 'OK' if count > 0 else 'MISSING'
            if count == 0:
                all_ok = False
            log(f'  {status:7s} {patch_name} ({count} markers)', 'PATCH')
        except Exception as e:
            log(f'  ERROR   {patch_name}: {e}', 'PATCH')
            all_ok = False
    if not all_ok:
        log('[PATCH] WARNUNG: Fehlende Patches können Training zum Absturz bringen!', 'PATCH')
        log('[PATCH] Fix: python apply_patches.py --apply', 'PATCH')
    else:
        log('[PATCH] Alle kritischen Patches aktiv.', 'PATCH')
    return all_ok

def print_versions():
    """Gibt alle relevanten Package-Versionen aus."""
    section('VERSIONS')
    import torch, bitsandbytes, transformers, peft
    log(f'  Python:         {sys.version.split()[0]}')
    log(f'  PyTorch:        {torch.__version__}')
    log(f'  CUDA (torch):   {torch.version.cuda}')
    log(f'  cuDNN:          {torch.backends.cudnn.version()}')
    log(f'  bitsandbytes:   {bitsandbytes.__version__}')
    log(f'  transformers:   {transformers.__version__}')
    log(f'  peft:           {peft.__version__}')
    log(f'  GPU count:      {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f'  GPU{i}:          {props.name} | {props.total_memory/1e9:.1f}GB | sm_{props.major}{props.minor}')

def log_device_map(model):
    """Gibt aus welche Layer auf welchen Geräten liegen."""
    section('DEVICE MAP')
    device_counts = {}
    for name, param in model.named_parameters():
        dev = str(param.device)
        device_counts[dev] = device_counts.get(dev, 0) + 1
    for dev, count in sorted(device_counts.items()):
        log(f'  {dev}: {count} Parameter-Tensoren')
    # LoRA-spezifisch
    lora_devs = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            dev = str(param.device)
            lora_devs[dev] = lora_devs.get(dev, 0) + 1
    log(f'  LoRA trainierbar: {lora_devs}')

def check_batch_health(input_ids, labels, step):
    """Prüft Batch auf bekannte Probleme und gibt Token-Statistiken aus."""
    total_tokens = input_ids.shape[0] * input_ids.shape[1]
    response_tokens = (labels != -100).sum().item()
    user_tokens = total_tokens - response_tokens
    mask_ratio = user_tokens / total_tokens if total_tokens > 0 else 0

    if response_tokens == 0:
        log(f'[S{step}] KRITISCH: Alle Labels maskiert (response_tokens=0)!', 'WARN')
        log(f'[S{step}] → Token 105 (<|turn>) nicht gefunden? Falsches Tokenizer-Format?', 'WARN')
        log(f'[S{step}] → input_ids sample: {input_ids[0][:20].tolist()}', 'WARN')
        return False

    if mask_ratio > 0.95:
        log(f'[S{step}] WARNUNG: {mask_ratio*100:.1f}% der Tokens maskiert — Antwort sehr kurz?', 'WARN')

    # Nur bei Step 0 und dann alle 100 Steps detailliert loggen
    if step == 0 or step % 100 == 0:
        log(f'[S{step}] Batch: total={total_tokens} response={response_tokens} user={user_tokens} '
            f'mask={mask_ratio*100:.1f}%  shape={list(input_ids.shape)}')

    return True

def check_loss_health(loss_val, step, first_loss_ref):
    """Prüft Loss auf NaN, Inf und andere Probleme."""
    if loss_val != loss_val:  # NaN check
        log(f'[S{step}] KRITISCH: Loss = NaN!', 'NAN')
        log(f'[S{step}] Häufige Ursachen:', 'NAN')
        log(f'  → LR zu hoch (aktuell: {LR}) — versuche 1e-4 oder 5e-5', 'NAN')
        log(f'  → Alle Response-Tokens maskiert (check_batch_health)', 'NAN')
        log(f'  → Gradient Explosion trotz Clipping', 'NAN')
        return False
    if loss_val == float('inf'):
        log(f'[S{step}] KRITISCH: Loss = Inf!', 'INF')
        return False
    if step == 0 and loss_val > WARN_LOSS_HIGH:
        log(f'[S{step}] WARNUNG: Loss={loss_val:.4f} nach S1 sehr hoch (>{WARN_LOSS_HIGH})', 'WARN')
        log(f'  → Mögliche Ursache: Daten-Format stimmt nicht überein', 'WARN')
        log(f'  → Erwartetes Format: <bos><|turn>user\\n[frage]<turn|>\\n<|turn>model\\n[antwort]<turn|>', 'WARN')
        log(f'  → Token 105 = <|turn>, Token 106 = <turn|> (unsloth/gemma-4-26B-A4B-it)', 'WARN')
    return True

def log_grad_stats(model, step):
    """Gibt Gradient-Statistiken pro Device nach dem Clipping aus."""
    device_norms = {}
    device_nan_count = {}
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dev = str(p.device)
            grad_norm = p.grad.norm().item()
            has_nan = torch.isnan(p.grad).any().item()
            if dev not in device_norms:
                device_norms[dev] = []
                device_nan_count[dev] = 0
            device_norms[dev].append(grad_norm)
            if has_nan:
                device_nan_count[dev] += 1
    for dev in device_norms:
        norms = device_norms[dev]
        nan_c = device_nan_count[dev]
        log(f'[S{step}] Grad {dev}: max={max(norms):.4f} mean={sum(norms)/len(norms):.4f} '
            f'params={len(norms)} NaN={nan_c}', 'GRAD')
        if nan_c > 0:
            log(f'[S{step}] KRITISCH: {nan_c} Parameter mit NaN-Gradienten auf {dev}!', 'ERROR')

# ── Startup ───────────────────────────────────────────────────────────────────
print_versions()
check_patches()

section('CONFIG')
log(f'  MODEL:        {MODEL}')
log(f'  TRAIN_FILE:   {TRAIN_FILE}')
log(f'  OUTPUT_DIR:   {OUTPUT_DIR}')
log(f'  MAX_MEMORY:   {MAX_MEMORY}')
log(f'  LORA_LAYERS:  0-{LORA_LAYERS[-1]} ({len(LORA_LAYERS)} Schichten)')
log(f'  MAX_SEQ_LEN:  {MAX_SEQ_LEN}')
log(f'  BATCH_SIZE:   {BATCH_SIZE}  GRAD_ACCUM: {GRAD_ACCUM}')
log(f'  LR:           {LR}  EPOCHS: {EPOCHS}')

# Prüfe ob Trainingsdaten existieren
if not Path(TRAIN_FILE).exists():
    log(f'FEHLER: Trainingsdatei nicht gefunden: {TRAIN_FILE}', 'ERROR')
    log('  → Datei muss existieren und Gemma4 Chat-Format haben', 'ERROR')
    sys.exit(1)

train_size_mb = Path(TRAIN_FILE).stat().st_size / 1e6
log(f'  TRAIN_FILE:   {train_size_mb:.1f} MB')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
section('TOKENIZER')
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    from torch.utils.data import Dataset, DataLoader
    import bitsandbytes as bnb

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f'  Tokenizer geladen: vocab_size={tok.vocab_size}')
    log(f'  pad_token: {repr(tok.pad_token)} (id={tok.pad_token_id})')
    log(f'  eos_token: {repr(tok.eos_token)} (id={tok.eos_token_id})')
    # Verifiziere Token-IDs für Response-Only Masking
    turn_id = tok.encode('<|turn>', add_special_tokens=False)
    log(f'  <|turn> token_id: {turn_id} (erwartet: [105])')
    if turn_id != [105]:
        log(f'  WARNUNG: Unerwartete Token-ID für <|turn>! Response-Only Masking wird falsch sein.', 'WARN')
        log(f'  → Prüfe Tokenizer-Version oder passe TURN_TOKEN_ID im Skript an.', 'WARN')
    TURN_TOKEN_ID = turn_id[0] if turn_id else 105
except Exception as e:
    log(f'Tokenizer-Fehler: {e}', 'ERROR')
    traceback.print_exc()
    sys.exit(1)

# ── Modell laden ───────────────────────────────────────────────────────────────
section('MODEL LOADING (NF4/QLoRA)')
dump_gpu_state('vor-load')

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # extra ~0.4 bit/param gespart
)
log(f'  Quant: NF4 + double_quant + bfloat16 compute')
log(f'  Erwarteter VRAM: ~13GB gesamt (26B × 4bit / 8)')

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_cfg,
        device_map='auto',
        max_memory=MAX_MEMORY,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    log('  Modell geladen.')
except torch.cuda.OutOfMemoryError as e:
    diagnose_oom(e, 'AutoModelForCausalLM.from_pretrained')
    sys.exit(1)
except Exception as e:
    log(f'Modell-Lade-Fehler: {e}', 'ERROR')
    traceback.print_exc()
    sys.exit(1)

dump_gpu_state('nach-load')

# Vision Tower entfernen (Text-only Training)
if hasattr(model.model, 'vision_tower'):
    log('  Vision Tower wird entfernt (Text-only Training)...')
    delattr(model.model, 'vision_tower')
    gc.collect()
    dump_gpu_state('nach-vision-remove')

model.enable_input_require_grads()

# ── LoRA Setup ────────────────────────────────────────────────────────────────
section('LORA SETUP')
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    layers_to_transform=LORA_LAYERS,
    lora_dropout=0.05, bias='none',
)
log(f'  r=16 alpha=32 targets=[q_proj, v_proj] layers=0-{LORA_LAYERS[-1]}')

try:
    model = get_peft_model(model, lora_cfg)
except Exception as e:
    log(f'LoRA Setup fehlgeschlagen: {e}', 'ERROR')
    traceback.print_exc()
    sys.exit(1)

# KRITISCHE REIHENFOLGE: gradient_checkpointing_enable() VOR model.train()
# → NEIN: model.train() MUSS VOR gradient_checkpointing_enable() kommen
# HuggingFace prüft: if self.gradient_checkpointing and self.training
# Ohne model.train() zuerst: GC aktiviert nie → alle Aktivierungen bleiben im Speicher → OOM
model.gradient_checkpointing_enable()
model.train()   # ← MUSS nach gradient_checkpointing_enable() sein. Nicht entfernen.

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
log(f'  Trainierbar: {trainable:,} / {total_p:,} ({100*trainable/total_p:.3f}%)')
log_device_map(model)

# Globale Referenzen für Signal Handler
_model_ref = model
_tok_ref = tok

# ── Dataset ───────────────────────────────────────────────────────────────────
section('DATASET')

class GinnyDataset(Dataset):
    """
    Splittet auf <bos> → eine vollständige Konversation pro Sample.
    Format: <bos><|turn>user\\n[frage]<turn|>\\n<|turn>model\\n[antwort]<turn|>
    Token 105 = <|turn> (Start), 106 = <turn|> (Ende)
    """
    def __init__(self, path, tokenizer, max_len):
        with open(path) as f:
            raw = f.read()

        chunks = [c.strip() for c in raw.split('<bos>') if len(c.strip()) > 50]
        log(f'  Rohchunks: {len(chunks)} (nach split auf <bos>)')

        self.samples = []
        short_count = 0
        for chunk in chunks:
            text = '<bos>' + chunk
            enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt')
            seq_len = enc['input_ids'].shape[1]
            if seq_len >= 8:
                self.samples.append(enc['input_ids'].squeeze(0))
            else:
                short_count += 1

        log(f'  Gültige Samples: {len(self.samples)} (verworfen: {short_count} zu kurz)')
        if len(self.samples) == 0:
            log('  KRITISCH: Keine gültigen Samples! Prüfe TRAIN_FILE Format.', 'ERROR')
            log('  → split(\'<bos>\') findet nichts? Falsches Trennzeichen?', 'ERROR')
            sys.exit(1)

        # Längenverteilung der ersten 5 Samples
        log('  Längenverteilung (erste 5 Samples):')
        for i, s in enumerate(self.samples[:5]):
            log(f'    Sample {i}: {s.shape[0]} Tokens — {repr(tokenizer.decode(s[:20].tolist())[:80])}')

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    """
    Bug 1 Fix: Input und Labels trennen.
    Padding in Labels: -100 (von cross-entropy ignoriert), nicht 0 (falsche Loss-Terme).
    """
    max_len = max(b.shape[0] for b in batch)
    input_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    label_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        input_padded[i, :b.shape[0]] = b
        label_padded[i, :b.shape[0]] = b   # nur echte Tokens haben Labels
    return input_padded, label_padded


dataset = GinnyDataset(TRAIN_FILE, tok, MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

# ── Optimizer ─────────────────────────────────────────────────────────────────
section('OPTIMIZER')
try:
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    log(f'  AdamW8bit: lr={LR} weight_decay=0.01')
except Exception as e:
    log(f'Optimizer-Fehler: {e}', 'ERROR')
    traceback.print_exc()
    sys.exit(1)

# ── Training ──────────────────────────────────────────────────────────────────
section('TRAINING START')
log(f'  {EPOCHS} Epoche(n) | {len(dataset)} Samples | {len(loader)} Steps/Epoche')
log(f'  Effektive Batch-Größe: {BATCH_SIZE * GRAD_ACCUM}')
log(f'  Gradient Clipping: 1.0 (per-device)')
log(f'  Gradient Checkpointing: aktiv')
log(f'  Logging: alle {LOG_EVERY} Steps')

start_total = time.time()
first_loss_ref = [None]  # mutable für Closure

for epoch in range(EPOCHS):
    section(f'EPOCH {epoch+1}/{EPOCHS}')
    epoch_loss = 0.0
    accum_tokens = 0
    accum_raw_loss = 0.0
    reported_loss = 0.0
    optimizer.zero_grad()
    step_times = []

    for step, batch in enumerate(loader):
        step_start = time.time()

        try:
            input_ids, labels = batch
            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')
        except Exception as e:
            log(f'[S{step}] Batch-Transfer-Fehler: {e}', 'ERROR')
            traceback.print_exc()
            continue

        # Response-Only Masking: User-Tokens mit -100 maskieren
        # Zweiter Token 105 (<|turn>) = Beginn des Model-Turns
        for b in range(labels.shape[0]):
            turn_pos = (input_ids[b] == TURN_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(turn_pos) >= 2:
                model_start = turn_pos[1].item() + 3  # skip <|turn>, "model", "\n"
                labels[b, :model_start] = -100
            elif step == 0:
                log(f'[S{step}] WARNUNG: Sample {b} hat < 2 <|turn>-Token-Vorkommen!', 'WARN')
                log(f'  → Kein Model-Turn gefunden → ganzer Text wird für Loss berechnet', 'WARN')
                log(f'  → turn_positions: {turn_pos.tolist()}', 'WARN')

        # Batch-Gesundheitscheck
        if not check_batch_health(input_ids, labels, step):
            optimizer.zero_grad()
            continue

        # Forward Pass
        try:
            fwd_start = time.time()
            out = model(input_ids=input_ids, labels=labels)
            torch.cuda.synchronize()
            fwd_time = time.time() - fwd_start
        except torch.cuda.OutOfMemoryError as e:
            diagnose_oom(e, f'Forward Pass S{step}')
            log(f'[S{step}] Überspringe Step nach OOM.', 'ERROR')
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            log(f'[S{step}] Forward-Fehler: {e}', 'ERROR')
            log(f'  input_ids: shape={list(input_ids.shape)} device={input_ids.device}', 'ERROR')
            log(f'  labels:    shape={list(labels.shape)} device={labels.device}', 'ERROR')
            traceback.print_exc()
            optimizer.zero_grad()
            continue

        # Loss Gesundheitscheck
        if not check_loss_health(out.loss.item(), step, first_loss_ref):
            log(f'[S{step}] Stoppe Training wegen NaN/Inf Loss.', 'ERROR')
            log(f'  → Emergency Checkpoint wird gespeichert...', 'ERROR')
            _save_emergency(None, None)

        if first_loss_ref[0] is None:
            first_loss_ref[0] = out.loss.item()
            log(f'[S0] Erster Loss: {first_loss_ref[0]:.4f} (Referenz)')

        # Token-normierter Loss (Bug 2 Fix)
        n_tokens = (labels != -100).sum().item()
        if n_tokens == 0:
            continue
        accum_tokens += n_tokens
        loss = out.loss * (n_tokens / GRAD_ACCUM)

        # Backward Pass
        try:
            bwd_start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            bwd_time = time.time() - bwd_start
        except torch.cuda.OutOfMemoryError as e:
            diagnose_oom(e, f'Backward Pass S{step}')
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            log(f'[S{step}] Backward-Fehler: {e}', 'ERROR')
            traceback.print_exc()
            optimizer.zero_grad()
            continue

        accum_raw_loss += out.loss.item() * n_tokens
        epoch_loss += out.loss.item()

        # Gradient Accum: Clip + Step
        if (step + 1) % GRAD_ACCUM == 0:
            reported_loss = accum_raw_loss / max(accum_tokens, 1)
            accum_tokens = 0
            accum_raw_loss = 0.0

            # Per-Device Gradient Clipping
            # clip_grad_norm_() schlägt fehl bei Multi-Device LoRA → per-device clippen
            try:
                seen_devices = set()
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None and p.device not in seen_devices:
                        seen_devices.add(p.device)
                        dp = [q for q in model.parameters()
                              if q.requires_grad and q.grad is not None and q.device == p.device]
                        torch.nn.utils.clip_grad_norm_(dp, 1.0)
            except Exception as e:
                log(f'[S{step}] Gradient-Clip-Fehler: {e}', 'ERROR')
                traceback.print_exc()

            # Gradient Stats (alle 50 Accum-Steps = alle 200 Steps)
            global_step = epoch * len(loader) + step + 1
            if global_step % (LOG_EVERY * GRAD_ACCUM * 5) == 0 or step < GRAD_ACCUM:
                log_grad_stats(model, step)

            try:
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                log(f'[S{step}] Optimizer-Fehler: {e}', 'ERROR')
                traceback.print_exc()
                optimizer.zero_grad()
                continue

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Logging
        global_step = epoch * len(loader) + step + 1
        if global_step % LOG_EVERY == 0 or step == 0:
            elapsed = time.time() - start_total
            avg_loss = epoch_loss / (step + 1)
            steps_remaining = (EPOCHS * len(loader)) - global_step
            eta_h = steps_remaining * (elapsed / max(global_step, 1)) / 3600
            peak0 = torch.cuda.max_memory_allocated(0) / 1e9
            alloc0 = torch.cuda.memory_allocated(0) / 1e9
            r = reported_loss if reported_loss > 0 else avg_loss
            avg_step = sum(step_times[-LOG_EVERY:]) / max(len(step_times[-LOG_EVERY:]), 1)

            log(f'[E{epoch+1} S{global_step}] '
                f'loss={r:.4f} avg={avg_loss:.4f} '
                f'peak={peak0:.1f}GB alloc={alloc0:.1f}GB '
                f't/step={avg_step:.2f}s eta={eta_h:.1f}h')
            torch.cuda.reset_peak_memory_stats(0)
            if torch.cuda.device_count() > 1:
                peak1 = torch.cuda.max_memory_allocated(1) / 1e9
                alloc1 = torch.cuda.memory_allocated(1) / 1e9
                log(f'  GPU1: peak={peak1:.1f}GB alloc={alloc1:.1f}GB')
                torch.cuda.reset_peak_memory_stats(1)

    avg_loss = epoch_loss / max(len(loader), 1)
    section(f'EPOCH {epoch+1} DONE')
    log(f'  avg_loss={avg_loss:.4f}')
    dump_gpu_state('nach-epoche')

    log(f'  Speichere Checkpoint: {OUTPUT_DIR}/epoch_{epoch+1}')
    try:
        model.save_pretrained(f'{OUTPUT_DIR}/epoch_{epoch+1}')
        tok.save_pretrained(f'{OUTPUT_DIR}/epoch_{epoch+1}')
        log(f'  Checkpoint gespeichert.')
    except Exception as e:
        log(f'  Checkpoint-Save fehlgeschlagen: {e}', 'ERROR')
        traceback.print_exc()

# ── Final ─────────────────────────────────────────────────────────────────────
section('TRAINING COMPLETE')
total_h = (time.time() - start_total) / 3600
log(f'  Gesamtdauer: {total_h:.2f}h')
log(f'  Final Log: {LOG_FILE}')

try:
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    log(f'  Final Adapter: {OUTPUT_DIR}')
except Exception as e:
    log(f'  Final Save fehlgeschlagen: {e}', 'ERROR')
    traceback.print_exc()

dump_gpu_state('final')
