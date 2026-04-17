# Validation Status — per patch, per version

Status after a code-review audit against `bitsandbytes` main (post 0.49.2),
`transformers` main (Gemma4 support), `peft` 0.19.1, `accelerate` 1.13.0.

This file exists because the original README overclaimed — several patches
turned out to protect against states that do not organically occur in
current upstream. Being honest about that is more useful to anyone reading
these patches than keeping the original framing.

## Legend

- **CONFIRMED** — Reproducer reliably triggers the crash on current upstream;
  patch fixes it. Safe to apply.
- **UPSTREAMABLE** — CONFIRMED, and the fix is general enough to send as a PR.
- **HISTORICAL** — Observed against an earlier version or transient state;
  cannot reproduce organically on current upstream with a clean minimal
  reproducer. Kept in the repo for historical reference, **not recommended**
  for new users on 0.49.x+.
- **NOT-BNB** / **NOT-PEFT** — The originally-described problem is real but
  the responsible package was misattributed. See "Correct home" column.

## Status matrix

### bitsandbytes patches

| Patch | Target | 0.49.x status | Notes |
|-------|--------|---------------|-------|
| P1 | `autograd/_functions.py` L~149 | **HISTORICAL** | The `int8_vectorwise_quant(B.to(fp16))` path in 0.49.x appears to be reached with `B.dtype` still appropriate. My original repro relied on forcing an already-INT8 `B` through this path; I haven't reproduced this organically on 0.49.2. Leaving the patch here with a disclaimer. |
| P2 | `autograd/_functions.py` L~247 | **HISTORICAL** | In 0.49.x `main`, `MatMul8bitLt.forward` does **not** set `state.SCB = None` after forward, and `backward` reads `state.SCB` directly without a None-guard. My reproducer used a synthetic `_force_scb_none()` helper that does not correspond to organic 0.49.x behavior. See [bnb #1927 discussion](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1927). |
| P3 | `nn/modules.py` L~1021 | **CONFIRMED (defensive)** | `getattr(self.weight, scb_name)` without a default can still hit meta-device weights during specific offload orderings. The `getattr(..., None)` default is defensive and has no downside. |
| P9 | `autograd/_functions.py` L~162 | **HISTORICAL** | Built on the same "SCB got cleared during GC recompute" assumption as P2. If P2 is historical, P9 is too. |
| P11 | `nn/modules.py` L~639 | **UPSTREAMABLE** | `Int8Params.__new__()` not accepting `**kwargs` blocks `transformers` 5.x paths that pass `_is_hf_initialized`. 0.49.2 still has the restrictive signature. Will open a PR. |

### transformers patches

| Patch | Target | Status | Notes |
|-------|--------|--------|-------|
| P4 | `models/gemma4/modeling_gemma4.py` (RoPE) | **CONFIRMED** | Reproduced with Q/K on CUDA, cos/sin on CPU — see reproducer in [transformers #45482](https://github.com/huggingface/transformers/issues/45482). |
| P5 | `models/gemma4/modeling_gemma4.py` (inputs) | **CONFIRMED** | Cross-device offload during forward. Reproducer in the same issue. |
| P6 | `integrations/sdpa_attention.py` | **UPSTREAMABLE** | Most general of the set — any offloaded model that routes through the SDPA integration benefits. Short, portable one-liner. |
| P7 | `models/gemma4/modeling_gemma4.py` (`layer_scalar`) | **CONFIRMED** | Narrower than P5 but same root cause. |
| P10 | `models/gemma4/modeling_gemma4.py` (`mm_token_type_ids`) | **UPSTREAMABLE** | One-line guard. Unblocks every text-only Gemma4 fine-tune. Highest-impact single patch in the set. |

### peft patches

| Patch | Target | Status | Notes |
|-------|--------|--------|-------|
| P8 | `tuners/lora/bnb.py` (2 sites) | **UPSTREAMABLE** | LoRA delta on CUDA, base INT8 result on CPU after offload. Reproducer in [peft #3169](https://github.com/huggingface/peft/issues/3169). A corresponding accelerate-side fix may be cleaner long-term. |

### Non-package claims (previously in README, now removed)

| Claim | Status | Correct home |
|-------|--------|--------------|
| `model.train()` must be called before `gradient_checkpointing_enable()` | **NOT-BNB** | This is a `transformers`/`torch` concern (HF checks `if self.gradient_checkpointing and self.training`). It's not a bitsandbytes property. The README implied otherwise; that framing has been removed. |
| "~6.25s/step flat across seq lengths → CPU→GPU transfer dominates" | Benchmark observation only | Not a bitsandbytes performance claim. Belongs in the PCIELink research writeup, not here. Removed from the bitsandbytes-adjacent sections. |

## What this changes for existing users

- If you previously applied P1/P2/P9 because you saw a crash on 0.49.x, **please open an issue with the exact traceback** — I want a real reproducer, not a synthetic one. I'll update the patch or withdraw it.
- P11 and all transformers / peft patches remain as-advertised; the reproducers are in the linked issues.
- `apply_patches.py` now emits a warning for HISTORICAL patches and continues. It does not refuse to apply them — there may be environments where they still help.

## Why this file exists

A maintainer (Matthew Douglas, bitsandbytes) pointed out in
[bnb #1927](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1927)
that the P2 fix looks like an AI-generated band-aid on a red herring.
Re-reading my own reproducer showed he was substantively right.

Rather than quietly paper over that, this file is the honest status. If
you're here because you hit a real 0.49.x crash that one of these patches
"fixes" — please open an issue. A confirmed organic reproducer is more
valuable than a fix.
