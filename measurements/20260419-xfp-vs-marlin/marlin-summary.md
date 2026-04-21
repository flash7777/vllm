# Marlin-INT4-AutoRound Run Summary — Qwen3.5-122B-A10B + fp8 KV + fp8 LM-head

**Date:** 2026-04-20
**Host:** DGX Spark (GB10, SM121a, aarch64)
**Image:** `localhost/vllm-multiquant:latest` (42 GB, NGC 26.03-py3 base)
**Run tag:** Marlin-INT4-AutoRound+fp8KV+fp8LMH

## Configuration

- **Model:** `Qwen3.5-122B-A10B-int4-AutoRound` (Intel AutoRound, `packing_format: auto_round:auto_gptq`)
- **Weights:** INT4 (native) via `GPTQMarlinLinearMethod` + `fused_marlin_moe` for MoE
- **KV cache:** fp8 (RTN, via `--kv fp8`)
- **LM head:** fp8 E4M3 (via `--weight-dtype-lm-head fp8`)
- **Attention backend:** **FLASHINFER** (auto-selected, vs XFP run's FLASH_ATTN — see Notes)
- **CUDA graphs:** default (vLLM VLLM_COMPILE mode)

## Stage 1 — bench.py (seed=42, deterministic)

```
Benchmark: Marlin-INT4-AutoRound+fp8KV+fp8LMH
URL: http://localhost:8011  Model: glm-4.7-flash

--- Performance (n=5) ---
  short   :    2.5 tok/s  (20 tok in 7.89s, n=5)
  medium  :   29.6 tok/s  (150 tok in 5.07s, n=5)
  long    :   25.8 tok/s  (400 tok in 15.52s, n=5)

--- Math Accuracy (n=50) ---
  Math: 47/50 (94%)

--- Memory ---
  Memory: KV cache: 0.0%
```

**Headline: long decode = 25.8 tok/s, medium = 29.6 tok/s, math 94 %.**

## Stage 2 — nsys (pending)

Same status as XFP run: torch.profiler via `/start_profile` needs
`--profiler-config` at serve start. Deferred for future run.

## Notes

- **Attention backend differs from XFP run:** Marlin selected FLASHINFER,
  XFP selected FLASH_ATTN. This is vLLM's automatic backend selection
  (dependent on weight format / kernel availability). Affects
  comparability of `attn/core` numbers in the future nsys breakdown.
- **Math is 4 pp lower than XFP** (94 % vs 98 %). Auto-Round-INT4 MoE
  loses more accuracy than XFP's per-channel learned codebook.
- Weight load was fast (14 shards, ~7 min cold — AutoRound is a drop-in
  GPTQ pack, no Lloyd pass needed).
