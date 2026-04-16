# consumer-llm-patches

**11 patches to train Gemma4 26B on a single RTX 4090.**

> Built by Max Bitzer (@sirfyyn) in one session, from a living room.  
> No A100. No NVLink. No enterprise budget.  
> Just a 4090, a Blackwell workstation GPU, and the refusal to accept "unsupported" as an answer.

---

## What this is

Gemma4 26B-A4B is a state-of-the-art MoE (Mixture-of-Experts) model from Google.  
Officially, fine-tuning it requires multi-GPU enterprise hardware.

These 11 patches make it trainable on:
- RTX 4090 (24GB, sm_89) — primary compute
- RTX PRO 2000 / any second GPU (optional, for weight distribution)
- 55GB+ system RAM (CPU offload instead of disk)

**Result:** Forward + backward pass through Gemma4 26B with BnB INT8 + PEFT LoRA + Gradient Checkpointing. Verified. Training runs overnight.

---

## The patches

### bitsandbytes — `autograd/_functions.py`

| Patch | Line | Problem | Fix |
|-------|------|---------|-----|
| P1 | ~149 | Pre-quantized Int8Params hit NaN (no CB/SCB) | Detect `B.dtype==int8`, build CB/SCB directly |
| P9 | ~162 | GC recompute: `post_forward` clears SCB but outer `if` skips restore | Standalone SCB restore before threshold check |
| P2 | ~247 | Backward: `state.SCB is None` → crash in `int8_mixed_scaled_mm` | SCB=None guard, recompute from CB |

### bitsandbytes — `nn/modules.py`

| Patch | Line | Problem | Fix |
|-------|------|---------|-----|
| P3 | ~1021 | `getattr(self.weight, scb_name)` crashes on meta-device weights | Use `getattr(..., None)` default |
| P11 | ~639 | `Int8Params.__new__()` rejects `_is_hf_initialized` kwarg from transformers 5.x | Add `**kwargs` to signature (absorbs any HF-injected params) |

### transformers — `models/gemma4/modeling_gemma4.py`

| Patch | Line | Problem | Fix |
|-------|------|---------|-----|
| P4 | ~751 | RoPE `cos/sin` on wrong device in cross-device MoE offload | `.to(device=x.device)` before unsqueeze |
| P5 | ~1366 | `attention_mask`, `position_ids`, `position_embeddings` on wrong device | Normalize all inputs to `hidden_states.device` |
| P7 | ~1423 | `layer_scalar` on cuda:0 when decoder layer is disk-offloaded | `self.layer_scalar.to(hidden_states.device)` |
| P10 | ~2040 | Text-only training raises: `mm_token_type_ids is required` | Replace `raise` with `pass` — text tokens need no multimodal mask |

### transformers — `integrations/sdpa_attention.py`

| Patch | Line | Problem | Fix |
|-------|------|---------|-----|
| P6 | ~92 | `attention_mask` on wrong device before SDPA | `.to(query.device)` |

### peft — `tuners/lora/bnb.py`

| Patch | Line | Problem | Fix |
|-------|------|---------|-----|
| P8 | ~267, ~549 | LoRA output on cuda:0, base result on cuda:1 | `result + output.to(result.device)` (both occurrences) |

---

## The root cause — why all of this was needed

When Gemma4 26B is loaded with `device_map="auto"` across GPU0 + GPU1 + CPU:

1. **Layers 0-19** land on GPU0 (RTX 4090, INT8-quantized)
2. **Layers 20-29** overflow to CPU RAM (INT8 with fp32 CPU offload)
3. **GPU1 (Blackwell sm_100)** gets some layers — but can't run BnB INT8 CUDA kernels (compiled for sm_89 max)

The cross-device tensor routing, the INT8 state machine, and Gemma4's multimodal forward all assume single-device or NVLink. None of them were tested on this configuration. Each patch fixes one assumption.

**The key insight:** Gradient Checkpointing must be activated via `model.train()` BEFORE `gradient_checkpointing_enable()`. Without this, GC never activates even after being "enabled" — because HuggingFace checks `if self.gradient_checkpointing and self.training`. The model starts in eval mode after `from_pretrained`.

Without GC: `MatMul8bitLt.ctx.state = state` keeps `state.CB` (968MB per disk layer) alive for ALL 30 layers simultaneously in the autograd graph → ~9.7GB accumulation → OOM.

With `model.train()` + GC: disk layers run in `torch.no_grad()` during checkpoint pass. `ctx` objects are ephemeral. `state.CB` freed after each layer. Problem solved.

---

## Performance — empirically measured

Setup: Gemma4 26B-A4B, BnB INT8, PEFT LoRA (r=16), Gradient Checkpointing, CPU offload

| Configuration | Step time | Notes |
|--------------|-----------|-------|
| Disk offload (before) | ~30-60s/step | NVMe 3GB/s bottleneck |
| CPU offload (after) | ~6.25s/step | PCIe 16GB/s, nearly flat across seq lengths |
| 64 tokens | 5.89s | |
| 128 tokens | 5.93s | |
| 256 tokens | 6.01s | |
| 512 tokens | 6.25s | **1.06× vs 64 tokens** |

**Critical finding:** Step time is nearly flat across sequence lengths (64→512 tokens = 1.06×).  
The bottleneck is CPU→GPU weight transfer (constant per forward pass), not compute.  
This is the empirical foundation for [PCIELink research](https://github.com/sirfyyn/pcielink) — async prefetching would reduce ~6s to ~1s.

---

## Tested versions

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0+cu130 | CUDA 13.0 — required for BnB 0.49.2 cpp extensions |
| bitsandbytes | 0.49.2 | P11 required for transformers ≥ 5.x |
| transformers | 5.5.4 | P4–P7, P10 apply here |
| peft | 0.19.1 | P8 applies here |
| accelerate | 1.13.0 | — |

> **Note on upgrades:** After `pip install --upgrade`, patches must be re-applied — pip overwrites the patched files.
> Use `python apply_patches.py --verify` to check patch state. See `apply_patches.py` for automation.

---

## How to apply

```bash
# Clone this repo
git clone https://github.com/sirfyyn/consumer-llm-patches
cd consumer-llm-patches

# Apply patches to your local bitsandbytes + transformers installation
python apply_patches.py --check   # verify patch targets exist
python apply_patches.py --apply   # apply all patches
python apply_patches.py --verify  # verify patches are active (grep PATCH sirfyyn)
```

---

## Training script

See `examples/train_gemma4_26b_consumer.py` — a complete fine-tuning script for Gemma4 26B on consumer hardware.

Key settings:
```python
MAX_MEMORY = {0: "22GiB", 1: "15GiB", "cpu": "60GiB"}  # CPU offload, not disk
model.train()           # CRITICAL: must be called BEFORE gradient_checkpointing_enable()
model.gradient_checkpointing_enable()
```

---

## Related research

**PCIELink** — Consumer Multi-GPU without NVLink: https://github.com/sirfyyn/pcielink

The flat step-time curve (seq-length doesn't matter, transfer does) points to a bigger problem:  
PCIe bandwidth between CPU and GPU is the bottleneck for all consumer multi-GPU AI workloads.  
PCIELink is the research project to fix this through async prefetching, compression, and a unified memory pool.

---

## Why this matters

This is not a niche problem. Everyone building their own LLM system on consumer hardware fights the same wall: no NVLink, no NVSwitch, no unified memory coherency. Enterprise tooling ignores this market. The open-source community waits — for better hardware, for NVIDIA, for someone else.

**These patches say: stop waiting. Start building.**

The 4090 + PRO 2000 setup in a living room is the lab. What works here, works for everyone with two consumer GPUs. That's thousands of developers worldwide hitting the exact same bottlenecks with no clean solution.

A working PCIELink would be:
- The first software treating consumer multi-GPU as a first-class citizen
- A foundation that simplifies FYOS, and every similar project
- A clear signal to the community: **start developing instead of waiting**

Not waiting for NVIDIA. Not waiting for NVLink prices to drop. Not waiting for the next enterprise system.

---

## Author

Max Bitzer (@sirfyyn)  
Built during FYOS development — a personal AI operating system running on consumer hardware.

Not sponsored. Not enterprise. Just someone who got tired of hearing "unsupported."

---

## License

MIT. Use it. Fork it. Fix more things.
