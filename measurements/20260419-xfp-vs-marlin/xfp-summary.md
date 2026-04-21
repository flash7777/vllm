# XFP Run Summary — Qwen3.5-122B-A10B + fp8 KV + fp8 LM-head

**Date:** 2026-04-20
**Host:** DGX Spark (GB10, SM121a, aarch64)
**Image:** `localhost/vllm-multiquant:latest` (42 GB, NGC 26.03-py3 base)
**Run tag:** XFP+fp8KV+fp8LMH

## Configuration

- **Model:** `Qwen3.5-122B-A10B` (on-the-fly MultiQuant packing from BF16)
- **Weights:** XFP auto (bits=2/3/4 per layer; MoE experts `bits=4`, shared/linear `bits=3`)
- **KV cache:** fp8 (RTN, via `--kv fp8`)
- **LM head:** fp8 E4M3 (via `--weight-dtype-lm-head fp8`)
- **Attention backend:** FlashAttention (fp8 KV → non-MULTIQUANT path per `cuda.py:59-62`)
- **Cache:** warm-start from `/data/tensordata/mq-cache/` (`← cache (skip Lloyd)` on every layer)
- **CUDA graphs:** default (vLLM VLLM_COMPILE mode, full + piecewise)

## Stage 1 — bench.py (seed=42, deterministic)

```
Benchmark: XFP+fp8KV+fp8LMH
URL: http://localhost:8011  Model: glm-4.7-flash

--- Performance (n=5) ---
  short   :    2.5 tok/s  (20 tok in 8.08s, n=5)
  medium  :   18.9 tok/s  (150 tok in 7.92s, n=5)
  long    :   17.3 tok/s  (400 tok in 23.12s, n=5)

--- Math Accuracy (n=50) ---
  Math: 49/50 (98%)

--- Memory ---
  Memory: KV cache: 0.0%
```

**Headline: long decode = 17.3 tok/s, medium = 18.9 tok/s, math 98 %.**

Matches prior Tagebuch-04-16 number (~16–18 tok/s) within noise.

## Stage 2 — nsys (pending)

Attempted capture with `nsys profile --capture-range=cudaProfilerApi`
failed: the in-container client sent the HTTP request to the vllm server
process, and nsys only profiled the client (no CUDA activity, 266 KB trace,
zero NVTX/kernel data in stats).

Also tried vLLM's `/start_profile` HTTP endpoint (torch.profiler-based,
writes Chrome trace) — returns HTTP 404 because `--profiler-config` was
not passed at serve start.

**Next capture approach** (for the Marlin run AND an XFP re-run):
pass `--profiler-config='{"profiler":"torch","output_dir":"/measurements"}'`
via `EXTRA_ARGS` so `/start_profile` + `/stop_profile` are wired. No nsys
needed — torch.profiler traces include NVTX (from `--enable-layerwise-nvtx-tracing`
hooks) plus GPU kernel timings.

## Notes / gotchas hit this run

- `@contextlib.contextmanager`-based custom NVTX CMs trip dynamo
  (gb0208, "_GeneratorContextManagerBase").
- Class-based custom CMs trip dynamo too (gb0142, "Dynamo does not know
  how to enter a `_NoOp` context manager").
- Custom `with _nvtx(...)` inserts in `Qwen3NextDecoderLayer.forward` etc.
  were fully reverted. vLLM's built-in `--enable-layerwise-nvtx-tracing`
  (hooks-based) is the right path — hooks fire outside the compiled
  region so dynamo doesn't see them.
