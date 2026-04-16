# Patch Files

Each file documents one patch with the exact old/new code.

Apply with `python apply_patches.py --apply` or manually.

| File | Package | Location |
|------|---------|----------|
| [p1_bnb_int8params.patch](p1_bnb_int8params.patch) | bitsandbytes | autograd/_functions.py ~L149 |
| [p2_bnb_backward_scb.patch](p2_bnb_backward_scb.patch) | bitsandbytes | autograd/_functions.py ~L247 |
| [p3_bnb_meta_device.patch](p3_bnb_meta_device.patch) | bitsandbytes | nn/modules.py ~L1021 |
| [p4_gemma4_rope_device.patch](p4_gemma4_rope_device.patch) | transformers | models/gemma4/modeling_gemma4.py ~L751 |
| [p5_gemma4_inputs_device.patch](p5_gemma4_inputs_device.patch) | transformers | models/gemma4/modeling_gemma4.py ~L1366 |
| [p6_sdpa_mask_device.patch](p6_sdpa_mask_device.patch) | transformers | integrations/sdpa_attention.py ~L92 |
| [p7_gemma4_layer_scalar.patch](p7_gemma4_layer_scalar.patch) | transformers | models/gemma4/modeling_gemma4.py ~L1423 |
| [p8_peft_lora_device.patch](p8_peft_lora_device.patch) | peft | tuners/lora/bnb.py ~L267, ~L549 |
| [p9_bnb_gc_scb_restore.patch](p9_bnb_gc_scb_restore.patch) | bitsandbytes | autograd/_functions.py ~L162 |
| [p10_gemma4_text_only.patch](p10_gemma4_text_only.patch) | transformers | models/gemma4/modeling_gemma4.py ~L2040 |
