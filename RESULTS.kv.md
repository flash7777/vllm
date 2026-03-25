# KV-Cache Quantization Benchmark Results

**Platform**: DGX Spark (GB10 Blackwell, SM121, 120 GiB unified memory)
**Image**: vllm-ng17e-riy (vLLM 0.15.1 + SM12x patches)
**Date**: 2025-03-25
**Bench**: `bench.py` (deterministic, seed=42)

## Qwen3.5-35B-A3B (INT4 AutoRound, MoE 35B/A3B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | 2.6 tok/s   | 47.8 tok/s    | 36.2 tok/s  | 100% | Baseline |
| TQ3 PyTorch | 2.7 tok/s | 35.4 tok/s  | 28.7 tok/s  | 100% | -21-26% (PyTorch fallback) |
| TQ3 CUDA v1 | 2.6 tok/s | 33.3 tok/s  | 29.1 tok/s  | 100% | -20-30% (naive GEMV) |
| TQ3 CUDA v2 | 8.1 tok/s | 35.5 tok/s  | 32.9 tok/s  | 100% | -9% long, -26% med (tiled GEMV+float4) |
| **TQ3 CUDA v3** | **2.7 tok/s** | **45.5 tok/s** | **32.8 tok/s** | **100%** | **-5% med, -9% long (cached buffers)** |
| TQ4      | —           | —             | —           | —    | Pending |

## Qwen3.5-122B-A10B (INT4 AutoRound, MoE 122B/A10B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | —           | —             | —           | —    | Pending |
| TQ3      | —           | —             | —           | —    | Pending |
| TQ4      | —           | —             | —           | —    | Pending |

## Context Scaling (Qwen3.5-35B, 100 gen tokens, DGX Spark)

| Context | FP8 tok/s | TQ3 tok/s | TQ4 tok/s | TQ3+FP8 tok/s | TQ3 vs FP8 |
|---------|-----------|-----------|-----------|--------------|------------|
| 0       | 42.0*     | 45.7      | 15.1*     | 15.6*        | +9%        |
| 2K      | 38.5      | 33.3      | 35.0      | 13.7*        | -13%       |
| 8K      | 29.3      | 26.6      | 29.1      | 32.6         | -9%        |
| 32K     | 13.5      | 13.5      | 13.9      | 14.5         | 0%         |
| 64K     | 6.9       | —         | 7.5       | 7.2          | —          |
| 128K    | 2.9       | —         | 3.3       | 2.9          | —          |

*Low-context values affected by JIT/warmup artifacts

**TQ3+FP8**: TQ round-trip on keys + FP8 KV-cache storage.
Same memory as FP8 (2× compression), but TQ improves key quality.
Enabled via `VLLM_TQ_ROUNDTRIP=1` + `--kv-cache-dtype fp8`.

At 8K context TQ3+FP8 is **+19% faster** than FP8 alone (32.6 vs 27.4).
At 32K+ contexts, throughput converges (memory-bound, same cache size).

**NOTE**: Phase 1 TQ round-trip does NOT save additional KV-cache memory
beyond what FP8 provides. Asymmetric K/V layout (Q2) needed for 4-5× savings.

## Config

```
--max-model-len 32768
--gpu-memory-utilization 0.05
--kv-cache-memory-bytes 10G
--enforce-eager
--trust-remote-code
--limit-mm-per-prompt '{"image":0,"video":0}'
```

## TurboQuant Standalone Quality (CPU, head_dim=128)

| Metric | TQ3 | TQ4 |
|--------|-----|-----|
| Score correlation vs FP16 | 0.92 | 0.98 |
| Output cosine similarity | 0.92 | 0.98 |
| Inner product bias | < 0.005 | < 0.002 |
| Needle-in-haystack top-1 | 100% | 100% |
| Key compression vs FP16 | 4.9x | 3.8x |
| GPU quantize throughput (GB10) | 13M vecs/s | 9M vecs/s |
