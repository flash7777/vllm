# MultiQuant Benchmark Results

Platform: DGX Spark (GB10, SM121, 128 GB Unified Memory)

## GLM-4.7-Flash (31B MoE, D=128, rein Transformer)

Model: `unsloth/GLM-4.7-Flash-FP8-Dynamic` (FP8) / `GLM-4.7-Flash-int4-AutoRound` (INT4)

### Vanilla (kein MTP)

| KV-Dtype | Bits | Packed B/Key | tok/s | Math% | KV B/Block | Notes |
|----------|------|-------------|-------|-------|------------|-------|
| fp8      | 8    | 128         |       |       |            | Baseline |
| tq3      | 3    | 52          |       |       |            | TurboQuant |
| tq4      | 4    | 68          |       |       |            | TurboQuant |
| rq2      | 2    | 36          |       |       |            | RotorQuant |
| rq3      | 3    | 52          |       |       |            | RotorQuant |
| rq4      | 4    | 68          |       |       |            | RotorQuant |

### +MTP NST=1

| KV-Dtype | tok/s | Math% | Notes |
|----------|-------|-------|-------|
| fp8      |       |       |       |
| tq3      |       |       |       |
| tq4      |       |       |       |

## Qwen3.5-35B-A3B (35B MoE, D=256, Hybrid Mamba+Attention)

Model: `Qwen3.5-35B-A3B-int4-AutoRound`

Flags: `--compilation-config '{"cudagraph_mode":"none"}' -e TORCH_CUDNN_V8_API_DISABLED=1`

### Vanilla

| KV-Dtype | Bits | Packed B/Key | tok/s | Math% | KV B/Block | Notes |
|----------|------|-------------|-------|-------|------------|-------|
| fp8      | 8    | 256         |       |       |            | Baseline |
| tq3      | 3    | 100         |       |       |            | TurboQuant |
| tq4      | 4    | 132         |       |       |            | TurboQuant |
| rq2      | 2    | 68          |       |       |            | RotorQuant |
| rq3      | 3    | 100         |       |       |            | RotorQuant |
| rq4      | 4    | 132         |       |       |            | RotorQuant |

## Compression Ratios

| D | Dtype | Packed | vs FP8 | vs BF16 |
|---|-------|--------|--------|---------|
| 128 | fp8  | 128 B  | 1.0×   | 2.0×    |
| 128 | tq3  | 52 B   | 2.5×   | 4.9×    |
| 128 | tq4  | 68 B   | 1.9×   | 3.8×    |
| 128 | rq2  | 36 B   | 3.6×   | 7.1×    |
| 128 | rq3  | 52 B   | 2.5×   | 4.9×    |
| 128 | rq4  | 68 B   | 1.9×   | 3.8×    |
| 256 | fp8  | 256 B  | 1.0×   | 2.0×    |
| 256 | tq3  | 100 B  | 2.6×   | 5.1×    |
| 256 | tq4  | 132 B  | 1.9×   | 3.9×    |
| 256 | rq2  | 68 B   | 3.8×   | 7.5×    |
| 256 | rq3  | 100 B  | 2.6×   | 5.1×    |
| 256 | rq4  | 132 B  | 1.9×   | 3.9×    |

## Bisherige TQ-Ergebnisse (Q4, GLM-4.7-Flash, D=128)

Referenz aus früheren Benchmarks:

| KV-Dtype | tok/s (short/med/long) | Math% | KV B/Block |
|----------|----------------------|-------|------------|
| fp8      | 37.2 / 40.7 / 40.3  | 100   | 40,960     |
| tq3      | 45.2 / 47.2 / 42.8  | 100   | 17,920     |
| tq4      | 40.6 / 42.5 / 45.9  | 100   | 23,040     |
