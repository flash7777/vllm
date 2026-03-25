# KV-Cache Quantization Benchmark Results

**Platform**: DGX Spark (GB10 Blackwell, SM121, 120 GiB unified memory)
**Image**: vllm-ng17e-riy (vLLM 0.15.1 + SM12x patches)
**Date**: 2025-03-25
**Bench**: `bench.py` (deterministic, seed=42)

## Qwen3.5-35B-A3B (INT4 AutoRound, MoE 35B/A3B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | 2.6 tok/s   | 47.8 tok/s    | 36.2 tok/s  | 100% | Baseline |
| **TQ3**  | **2.7 tok/s** | **35.4 tok/s** | **28.7 tok/s** | **100%** | **-21-26% overhead (PyTorch round-trip)** |
| TQ4      | —           | —             | —           | —    | Pending |

## Qwen3.5-122B-A10B (INT4 AutoRound, MoE 122B/A10B)

| KV-Cache | Short (20t) | Medium (150t) | Long (400t) | Math | Notes |
|----------|-------------|---------------|-------------|------|-------|
| FP8      | —           | —             | —           | —    | Pending |
| TQ3      | —           | —             | —           | —    | Pending |
| TQ4      | —           | —             | —           | —    | Pending |

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
