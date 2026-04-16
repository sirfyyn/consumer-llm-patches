# Benchmark: CPU Offload Step-Time vs Sequence Length

**Date:** 2026-04-16  
**Hardware:** RTX 4090 (24GB, sm_89) + RTX PRO 2000 (16GB, sm_100) + 60GB CPU RAM  
**Model:** Gemma4 26B-A4B-it (unsloth variant)  
**Config:** BnB INT8, PEFT LoRA r=16, Gradient Checkpointing, CPU offload for overflow layers  
**Script:** `smoke20_timing.py`

## Results

| Sequence Length | Step Time | Factor vs 64 tok |
|----------------|-----------|-----------------|
| 64 tokens | 5.89s | 1.00× |
| 128 tokens | 5.93s | 1.01× |
| 256 tokens | 6.01s | 1.02× |
| 512 tokens | 6.25s | **1.06×** |

## Interpretation

Step time is **nearly flat** across a 8× range of sequence lengths.

This means:
- **Compute is not the bottleneck.** 8× more tokens → only 6% more time.
- **CPU→GPU weight transfer dominates** (~94% of step time).
- The 10 CPU-offloaded layers each require a load from RAM to GPU per forward pass.
- Transfer cost: ~10 layers × ~968MB × 2 (load + unload) / PCIe bandwidth ≈ constant per step.

## Implications for PCIELink

This is Phase 0 empirical data for [PCIELink](https://github.com/sirfyyn/pcielink) Lever 2 (Async Pipeline).

If we prefetch the next layer while computing the current:
- Current: load layer → compute → unload → load next layer → ...
- With prefetch: [compute layer N] + [prefetch layer N+1 in parallel]
- Transfer time hidden behind compute time
- Expected: ~6.25s → ~1-2s per step

That's a **3-6× training speedup** from a single patch to accelerate's `AlignDevicesHook`.

## Raw command

```bash
python3 /tmp/ginny_smoke20_timing.py
# Output logged to /tmp/smoke20.log
```
