# consumer-llm-patches

**11 patches collected while trying to train Gemma4 26B on a single RTX 4090.**

> Built by Max Bitzer (@sirfyyn) in a living room, learning as I go.
> No A100. No NVLink. No enterprise budget.
> Just a 4090, a second consumer GPU, and a stubborn refusal to accept "unsupported" as the end of the story.

> **Honesty note (added after review):** Not every patch holds up equally well.
> After maintainer feedback and a code-review pass against current upstream,
> some of the bitsandbytes patches turned out to protect against states that
> do not reproduce organically on `bitsandbytes 0.49.x`. See
> **[VALIDATION-STATUS.md](VALIDATION-STATUS.md)** for the honest per-patch status
> (CONFIRMED / UPSTREAMABLE / HISTORICAL). Please read that file before applying
> the bitsandbytes patches.
>
> I'm still actively working on this stack; I share new findings as they come,
> and I revise or retract when new evidence contradicts what I said earlier.

---

## What this is

Gemma4 26B-A4B is a MoE (Mixture-of-Experts) model from Google.
Officially, fine-tuning it expects multi-GPU enterprise hardware.

These patches collect the fixes I needed to get a forward + backward pass
through Gemma4 26B with BnB INT8 + PEFT LoRA + Gradient Checkpointing + CPU
offload working on consumer hardware:

- RTX 4090 (24GB, sm_89) — primary compute
- RTX PRO 2000 / any second GPU (optional)
- ~55–60GB system RAM for CPU offload

Training does run overnight on this setup. Whether every individual patch is
the *right* fix is what `VALIDATION-STATUS.md` tries to answer honestly.

---

## The patches at a glance

See [VALIDATION-STATUS.md](VALIDATION-STATUS.md) for per-patch status and
reproducibility notes. Summary:

### bitsandbytes — `autograd/_functions.py` + `nn/modules.py`

| Patch | Area | Current-upstream status |
|-------|------|-------------------------|
| P1 | Int8Params pre-quantized path | **HISTORICAL** — no organic 0.49.x repro |
| P2 | Backward SCB=None guard | **HISTORICAL** — repro uses a synthetic `_force_scb_none()`, not organic 0.49.x state |
| P9 | GC recompute SCB restore | **HISTORICAL** — built on the same assumption as P2 |
| P3 | `getattr(weight, scb_name, None)` on meta-device | **CONFIRMED (defensive)** |
| P11 | `**kwargs` in `Int8Params.__new__` for transformers 5.x | **UPSTREAMABLE** — PR in preparation |

### transformers — `models/gemma4/modeling_gemma4.py` + SDPA integration

| Patch | Problem | Status |
|-------|---------|--------|
| P4 | RoPE cos/sin on CPU while Q/K on CUDA | **CONFIRMED** |
| P5 | `position_ids`/`attention_mask` not on hidden_states.device | **CONFIRMED** |
| P6 | SDPA `attention_mask` not on query device | **UPSTREAMABLE** — most portable of the set |
| P7 | `layer_scalar` on wrong device in decoder layer | **CONFIRMED** |
| P10 | `mm_token_type_ids` required for text-only path | **UPSTREAMABLE** — one-line guard, high impact |

Reproducers in the [upstream issue](https://github.com/huggingface/transformers/issues/45482).

### peft — `tuners/lora/bnb.py`

| Patch | Problem | Status |
|-------|---------|--------|
| P8 | LoRA output on cuda:0, base INT8 result on CPU → mismatch | **UPSTREAMABLE** — reproducer in [peft #3169](https://github.com/huggingface/peft/issues/3169). May be cleaner to fix at the `accelerate` layer instead. |

---

## Measured behaviour (observation, not a performance claim)

Setup: Gemma4 26B-A4B, BnB INT8, PEFT LoRA (r=16), Gradient Checkpointing, CPU offload.

| Config | Step time | |
|--------|-----------|---|
| Disk offload | ~30–60s/step | NVMe 3 GB/s bottleneck |
| CPU offload (same patches applied) | ~6.25s/step | PCIe ~16 GB/s |
| 64 tokens | 5.89s | |
| 128 tokens | 5.93s | |
| 256 tokens | 6.01s | |
| 512 tokens | 6.25s | 1.06× vs 64 tokens |

**What this is and isn't:** this is an observation about end-to-end training step
time on my setup after patches are applied. It is **not** a claim about
bitsandbytes performance — bitsandbytes' own kernels aren't what dominates
here; CPU→GPU weight transfer does. That separation matters and wasn't clear
enough in earlier versions of this README.

The nearly-flat curve across sequence lengths is what motivated the
[PCIELink](https://github.com/sirfyyn/pcielink) research direction (async
prefetching to hide transfer behind compute). That is a separate project.

---

## Tested versions

| Package | Version |
|---------|---------|
| torch | 2.11.0+cu130 |
| bitsandbytes | 0.49.2 |
| transformers | 5.5.4 (main w/ Gemma4 support) |
| peft | 0.19.1 |
| accelerate | 1.13.0 |

> After `pip install --upgrade`, patches must be re-applied.
> Use `python apply_patches.py --verify` to check state.

---

## How to apply

```bash
git clone https://github.com/sirfyyn/consumer-llm-patches
cd consumer-llm-patches

python apply_patches.py --check    # verify patch targets exist
python apply_patches.py --apply    # apply patches (emits warning for HISTORICAL ones)
python apply_patches.py --verify   # check patches are active
```

**Read [VALIDATION-STATUS.md](VALIDATION-STATUS.md) before applying.**
If you only need the patches that reproduce on current upstream, skip
P1/P2/P9 and apply the rest.

---

## Training script

See [`examples/train_gemma4_26b_consumer.py`](examples/train_gemma4_26b_consumer.py) —
a complete fine-tuning script for Gemma4 26B on consumer hardware, including
the CPU-offload `max_memory` configuration.

---

## Related research

**PCIELink** — async CPU→GPU prefetch, unified memory pool, consumer-MultiGPU
without NVLink: https://github.com/sirfyyn/pcielink

---

## Why this exists

Everyone trying to train an LLM on their own consumer hardware fights the same
wall: no NVLink, no unified coherency, tooling written assuming enterprise
topology. I hit these walls in sequence, wrote down what I changed, and
published it so the next person starts a few walls further along.

The honest part: I'm not an ML researcher. I started this a few months ago
and I work on the harder diagnoses together with an AI assistant (Claude).
Some of the patches turned out great. Others I over-claimed on — the
`VALIDATION-STATUS.md` file is the correction. Corrections welcome.

---

## Issues where maintainers are reviewing this work

- bitsandbytes #1927 — P1/P2/P9/P11 — [link](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1927)
- transformers #45482 — P4/P5/P6/P7/P10 — [link](https://github.com/huggingface/transformers/issues/45482)
- peft #3169 — P8 — [link](https://github.com/huggingface/peft/issues/3169)

---

## Author

Max Bitzer (@sirfyyn) — built during FYOS development, a personal AI
operating system on consumer hardware. Not sponsored, not enterprise.

## License

MIT.
