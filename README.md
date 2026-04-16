feat: 10 patches to train Gemma4 26B on a single RTX 4090
Patches bitsandbytes, transformers, and peft to enable fine-tuning
Gemma4 26B-A4B-it on consumer hardware with BnB INT8 + PEFT LoRA
+ Gradient Checkpointing + CPU offload (no NVLink, no A100).

Key patches:
- P1/P9/P2: BnB INT8 CB/SCB state machine fixes for GC recompute
- P3: BnB nn/modules meta-device SCB AttributeError fix
- P4/P5/P7: Gemma4 cross-device tensor routing (RoPE, inputs, layer_scalar)
- P6: SDPA attention_mask cross-device fix
- P8: LoRA output device normalization (both occurrences in peft/bnb.py)
- P10: Gemma4 text-only training — mm_token_type_ids not required

Critical insight: model.train() MUST precede gradient_checkpointing_enable().
Without this, GC never activates → state.CB accumulates for all layers → OOM.

Empirical benchmark (smoke20, 2026-04-16):
  Step time nearly flat across seq lengths (64→512 tok = 1.06×).
  CPU→GPU transfer dominates (~94% of step time), not compute.
  6.25s/step at 512 tokens → 7K samples × 1 epoch ≈ 12.8h overnight.

Built during FYOS development on RTX 4090 + RTX PRO 2000 Blackwell.
Not enterprise. Not sponsored. Just refusing to accept "unsupported."

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
