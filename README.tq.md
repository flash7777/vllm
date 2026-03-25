# TurboQuant KV-Cache Quantization for vLLM

TurboQuant compresses the KV-cache to 3-4 bits per coordinate with near-zero quality loss,
enabling longer context lengths and higher throughput within the same GPU memory.

Based on: *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
(ICLR 2026, Zandieh, Daliri, Hadian, Mirrokni — Google Research / NYU / DeepMind)

## How It Works

Two-stage compression per KV vector:

1. **PolarQuant (MSE stage)**: Random orthogonal rotation + per-coordinate Lloyd-Max quantization.
   Uses (b-1) bits per coordinate for near-optimal MSE distortion.

2. **QJL (residual correction)**: 1-bit sign quantization of the residual via the
   Quantized Johnson-Lindenstrauss transform. This corrects the inner-product bias
   inherent in MSE quantizers, yielding **unbiased** attention scores.

```
Storage per KV vector (tq3, head_dim=128):
  MSE indices:  32 bytes  (128 × 2 bits / 8)
  QJL signs:    16 bytes  (128 × 1 bit / 8)
  Norms:         4 bytes  (vec_norm + residual_norm as float16)
  Total:        52 bytes  vs 256 bytes FP16 → 4.9× key compression
```

## CLI Usage

```bash
vllm serve <model> --kv-cache-dtype tq3    # 3-bit (2-bit MSE + 1-bit QJL)
vllm serve <model> --kv-cache-dtype tq4    # 4-bit (3-bit MSE + 1-bit QJL)
```

### Full Example (DGX Spark)

```bash
vllm serve /data/tensordata/Qwen3.5-35B-A3B-int4-AutoRound \
  --served-model-name qwen35-35b-tq3 \
  --host 0.0.0.0 --port 8011 \
  --kv-cache-dtype tq3 \
  --gpu-memory-utilization 0.05 \
  --kv-cache-memory-bytes 10G \
  --max-model-len 131072 \
  --trust-remote-code \
  --enforce-eager
```

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `--kv-cache-dtype tq3` | `tq3`, `tq4` | TurboQuant 3-bit or 4-bit KV-cache |
| `--max-model-len` | int | Can be set higher than FP8/FP16 due to memory savings |
| `--kv-cache-memory-bytes` | e.g. `10G` | Same as FP8; TQ fits more tokens in same memory |

All other vLLM parameters work unchanged (`--tensor-parallel-size`, `--quantization`, etc).

## Compression Comparison

| KV-Cache Type | Bits/coord | Bytes/vector (d=128) | Compression vs FP16 |
|---------------|-----------|---------------------|-------------------|
| FP16          | 16        | 256                 | 1.0×              |
| FP8           | 8         | 128                 | 2.0×              |
| **TQ4**       | **4**     | **68**              | **3.8×**          |
| **TQ3**       | **3**     | **52**              | **4.9×**          |

*Phase 1: Only keys compressed. Values stored as FP16. Net K+V compression ~2× for tq3.*

## Quality

Tested on random vectors (d=128), comparing TurboQuant attention scores vs FP16 reference:

| Metric | TQ3 | TQ4 |
|--------|-----|-----|
| Score correlation | 0.92 | 0.98 |
| Attention output cosine similarity | 0.92 | 0.98 |
| Inner product bias | < 0.005 | < 0.002 |
| Needle-in-haystack top-1 | 100% | 100% |

## Architecture

```
┌─────────────────────────────────────────────────┐
│ vllm serve --kv-cache-dtype tq3                 │
│   │                                             │
│   ├─ config/cache.py: CacheDType "tq3"/"tq4"   │
│   ├─ platforms/cuda.py: TURBOQUANT priority     │
│   ├─ attention backend selector                 │
│   │                                             │
│   └─ TurboQuantAttentionBackend                 │
│       ├─ PREFILL: FlashAttention (FP16)         │
│       │           → quantize → store in cache   │
│       └─ DECODE:  Fused TQ score kernel         │
│                   q_rot = Q @ Pi^T (once)       │
│                   per token: gather + bit-unpack │
└─────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `vllm/turboquant/config.py` | TurboQuantConfig (bits, packed_size) |
| `vllm/turboquant/centroids.py` | Lloyd-Max codebook computation |
| `vllm/turboquant/quantizer.py` | Quantize, dequantize, pack/unpack, attention_scores |
| `vllm/v1/attention/backends/turboquant_attn.py` | Attention backend (PyTorch + Triton) |
| `vllm/v1/attention/ops/triton_tq_reshape_and_cache.py` | Triton: quantize-on-store |
| `vllm/v1/attention/ops/triton_tq_attention_score.py` | Triton: fused decode score |
| `tests/turboquant/test_quantizer.py` | Standalone correctness tests |
| `tests/turboquant/test_cache_pipeline.py` | End-to-end cache pipeline tests |
| `tests/turboquant/patch_container.py` | Runtime patching for container images |

## Container Usage

For existing vLLM container images without TurboQuant built in:

```bash
# Mount TQ code + run patch script
podman run -d --name vllm-tq \
  -v /path/to/vllm-riy/vllm/turboquant:/usr/local/lib/.../vllm/turboquant:ro \
  -v /path/to/vllm-riy/tests/turboquant/patch_container.py:/opt/patch.py:ro \
  ... \
  vllm-ng17e-riy bash -c "python3 /opt/patch.py && vllm serve ... --kv-cache-dtype tq3"
```

## Running Tests

```bash
# Standalone (no GPU needed)
python3 tests/turboquant/test_quantizer.py
python3 tests/turboquant/test_cache_pipeline.py

# Inside container with GPU
bash tests/turboquant/start_qwen35_tq.sh test
```

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Algorithms 1+2, Theorems 1-3
- [QJL CUDA kernels](https://github.com/amirzandieh/QJL) — Sign-bit quantization (same first author)
- [PolarQuant](https://github.com/ericshwu/PolarQuant) — Triton fused attention kernel reference
- [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — PyTorch reference implementation

## Status

**Phase 1** (current): PyTorch reference + Triton kernel drafts. Keys compressed, values FP16.
GPU quantization benchmark: 13M vecs/s on DGX Spark GB10.

**Phase 2** (planned): Triton kernels validated on GPU, value compression, full MetadataBuilder.

**Phase 3** (planned): CUDA fused kernels based on QJL CUDA code, production-ready.
