# Comparison — XFP vs Marlin-INT4-AutoRound (Qwen3.5-122B-A10B, DGX Spark SM121a)

**Date:** 2026-04-20
**Identical config:** fp8 KV + fp8 LM-head + same model architecture + same host.
**Only difference:** weight-GEMM path (XFP codebook-SHFL vs Marlin tensor-core INT4).

## Headline (bench.py, seed=42, n=5 decode)

| Metric | XFP v1 (attn-only quant) | Marlin INT4 AutoRound | XFP v2 (+linear_attn quant) | **Best / Baseline XFP** |
|---|---:|---:|---:|---:|
| **long** (400 tok) | 17.3 tok/s | 25.8 tok/s | **29.9 tok/s** | **1.73×** |
| **medium** (150 tok) | 18.9 tok/s | 29.6 tok/s | **34.8 tok/s** | **1.84×** |
| **short** (20 tok) | 2.5 tok/s | 2.5 tok/s | 2.6 tok/s | ~1× (latency-bound) |
| **Math accuracy** | 98 % | 94 % | **98 %** | unchanged |
| LM head | fp8 E4M3 | fp8 E4M3 | fp8 E4M3 | same |
| KV cache | fp8 | fp8 | fp8 | same |

**XFP v2 beats Marlin by +16 % long / +18 % medium AND has +4 pp math.** The
single fix (policy.py:141 — add `"linear_attn" in p` to the `attn` classifier)
unlocked 12.6 tok/s on long decode by quantising the 108 linear_attention
(GatedDeltaNet) projections that Qwen 3.5 uses for its hybrid attention stack
— previously falling through to bf16 because the classifier only matched
`self_attn`/`attention` substrings, not `linear_attn`. (The GatedDeltaNet
module class is reused from the Qwen3-Next implementation in vLLM; the model
itself is Qwen 3.5-122B-A10B.)

## Interpretation

**Marlin decode is ~1.5× faster than XFP, not 2–3×.**

The prior rumored gap (50 tok/s Marlin + MTP vs 16 tok/s XFP) conflated
speculative decoding gains with weight-GEMM gains. On a like-for-like
single-token decode (no MTP, same KV, same LM head) the weight-path gap is
**~50 %, not 200 %**.

XFP trades a fixed, modest decode penalty for **+4 pp math accuracy** —
worth considering as a quality-quantity knob, not just a speed deficit.

## Per-kernel GPU breakdown (torch.profiler, 60 decode steps each)

Both runs captured via vLLM `/start_profile` + `/stop_profile` (torch.profiler
with CUDA activities). Kernels classified into categories, per-step averages:

| Category | XFP (ms/tok) | Marlin (ms/tok) | Δ | Ratio M/X |
|---|---:|---:|---:|---:|
| `cublas_gemvx_bf16` | 11.22 | 5.18 | **-6.03** | **0.46×** |
| `weight_gemm_moe` (xfp_gemm MoEPolicy / marlin_moe_wna16) | 3.88 | 3.73 | -0.16 | **0.96×** |
| `weight_gemm_linear` (xfp_gemm LinearPolicy / marlin::Marlin) | 1.32 | 3.48 | +2.16 | 2.64× |
| `cutlass_wmma` | 0.36 | 2.46 | +2.11 | 6.92× |
| `attn_core` (flashinfer/flash_fwd) | 1.20 | 0.00 | -1.20 | 0× |
| `silu` (triton_poi / act_and_mul) | 0.03 | 0.44 | +0.41 | 14.3× |
| `scatter` | 0.85 | 0.01 | -0.84 | 0.01× |
| `elementwise` | 0.91 | 0.11 | -0.80 | 0.12× |
| `mamba_gdn` (fused_recurrent_gated_delta) | 0.26 | 0.26 | -0.00 | 0.98× |
| `splitk_reduce` | 0.14 | 0.15 | +0.01 | — |
| `reduce` | 0.00 | 0.10 | +0.10 | — |
| `moe_topk` | 0.08 | 0.26 | +0.18 | 3.33× |
| `other` | 0.56 | 0.42 | -0.14 | 0.75× |
| **TOTAL GPU** | **20.79** | **16.59** | **-4.20** | **1.25×** |

**bench.py said Marlin 1.49× faster. Kernel-only says 1.25×.** The extra
20 % gap (~3–4 ms/tok) is launch overhead, Python dispatch, HTTP roundtrip,
and scheduler — not on the GPU.

### Where the decode-speed delta *actually* lives

1. **Marlin MoE GEMM is NOT faster than XFP MoE GEMM.** Both ~3.8 ms/tok.
   XFP's 3-/4-bit SHFL-codebook dequant matches Marlin's 4-bit LOP3-dequant
   kernel-for-kernel. **The weight-GEMM gap is not the bottleneck.**

2. **-6 ms bf16 gemvx gap is the single biggest contributor.** XFP spends
   11 ms/tok in cuBLAS BF16 GEMV; Marlin spends 5 ms. About 2 ms of that
   reappears as Marlin's cutlass_wmma (the fp8 LM-head takes the cutlass
   path in Marlin, the bf16 gemvx path in XFP). The remaining ~4 ms/tok
   gap in gemvx is unexplained — needs `nvtx_gpu_proj_sum` / stack-frames
   to trace which layers dispatch bf16 GEMV in XFP but don't in Marlin.
   Candidates: Mamba/GatedDeltaNet in/out-proj, router linear,
   shared_expert_gate.

3. **scatter (0.84 ms) + elementwise (0.80 ms) = 1.6 ms fused-MoE-overhead.**
   Marlin's `moe_wna16_marlin_gemm` fuses scatter+silu+gemm; XFP does them
   as 3 separate dispatches. **This IS a clean optimization target** — fuse
   our `xfp_moe_gemm` with scatter_add inside the kernel.

4. **Marlin uses FLASHINFER attention; XFP uses flash_fwd.** The 1.2 ms
   XFP attention time doesn't exist in Marlin (different backend — likely
   zero-cost MLA cache path). Marginal contribution.

### Optimization results

| # | Target | Est. saving (ms/tok) | Status | Actual impact |
|---|---|---:|---|---|
| 1 | Quantise linear_attention (GatedDeltaNet) layers used by Qwen 3.5 hybrid stack | up to 10 | **DONE** (policy.py:141) | **+12.6 tok/s (1.73×) on long decode** |
| 2 | Fuse scatter_add into xfp_moe_gemm | ~0.8 | DONE (kernel already supports `topk_weights`) | below bench.py noise floor |
| 3 | Fuse silu/elementwise into xfp_moe_gemm | ~0.8 | DONE (`torch.ops._C.silu_and_mul`) | below bench.py noise floor |
| 4 | Align fp8 LM-head to cutlass path | ~2 | not started | — |
| 5 | Match attention backend across both | 0.5 | not started | — |

Item 1 was the dominant lever. Without it, XFP was running 36 × 3 = 108
bf16 linears per token for the Qwen 3.5 hybrid attention stack —
*unquantized* because `classify_layer()` looked for `self_attn`/`attention`
substrings but Qwen 3.5 uses `linear_attn` for the Mamba-style layers.
One-line fix unlocked 73 % speedup.

Items 2+3 are still correct wins (less Python glue, less elementwise
kernel pressure) but inside a CUDA-graph replay their wall-clock effect
is amortised below the bench.py resolution (0.1 tok/s ≈ 0.3 ms/step).
Worth keeping for clarity.

Remaining items 4+5 are fringe: Marlin's `cutlass_wmma` path is
~2 ms/tok faster than XFP's `gemvx` for LM-head, but that's now a small
fraction of the (much reduced) total.

## Artefacts

- `xfp.trace.json.gz` — torch.profiler chrome-trace (3.9 MB, 60 decode steps)
- `marlin.trace.json.gz` — torch.profiler chrome-trace
- Re-analysis:
  ```
  python3 -c "import gzip,json,collections; ev=json.load(gzip.open('PATH'))['traceEvents']; ..."
  ```
- Raw bench: `xfp-bench.txt`, `marlin-bench.txt`

## Still open

- **MTP / speculative decoding not measured.** Marlin's advertised 50 tok/s
  came with MTP 1..5. If we wire XFP + ngram-spec-4 we'd close more of the
  total-throughput gap regardless of per-kernel improvements.
- **Attention backend mismatch** (FLASH_ATTN vs FLASHINFER) not yet
  normalized. Small (1.2 ms) but should be made identical for clean
  comparison.
- **Per-layer NVTX breakdown** — `--enable-layerwise-nvtx-tracing` hooks
  produced only top-level `execute_context` + `ProfilerStep` annotations
  in the trace; no per-module ranges. vllm's hook-walker might need
  different CompilationMode to propagate into the inner trace.

## Config traceability

- **XFP run log:** `xfp-bench.txt`, `xfp-summary.md`
- **Marlin run log:** `marlin-bench.txt`, `marlin-summary.md`
- **bench.py:** `/home/flash/vllm-riy/bench.py` (seed=42, deterministic, per MEMORY.md "NICHT ÄNDERN")
- **Model files:**
  - XFP: `/data/tensordata/Qwen3.5-122B-A10B` (BF16 source, on-the-fly packed; cache in `/data/tensordata/mq-cache/`)
  - Marlin: `/data/tensordata/Qwen3.5-122B-A10B-int4-AutoRound` (14 shards, Intel Auto-Round w/ `auto_round:auto_gptq` packing)
- **Start flags used:**
  - XFP: `./start.multiquant --model Qwen3.5-122B-A10B --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 --nvtx`
  - Marlin: `./start.multiquant --model Qwen3.5-122B-A10B-int4-AutoRound --kv fp8 --weight-dtype-lm-head fp8 --nvtx`
