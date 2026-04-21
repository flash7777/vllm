# Methodology: XFP vs Marlin-INT4 Source-Traced Profiling

**Date:** 2026-04-19
**Host:** DGX Spark (flash@192.168.1.117), GB10 Blackwell (SM121a, aarch64)
**Base image:** `nvcr.io/nvidia/vllm:26.03-py3` → `localhost/vllm-multiquant:mq_2603_quantcache`

## Goal

**Why is Marlin-INT4 decode 2–3× faster than XFP on the same model?**

Side-by-side per-phase timing of the XFP and Marlin-INT4 decode paths on the
identical model architecture (Qwen3.5-122B-A10B, 48 layers, 47 MoE + 1 dense),
with identical KV-cache dtype (fp8), LM-head dtype (fp8), and attention
backend. **Only the weight-GEMM path differs.** Every timing bucket is mapped
to an exact source location via NVTX ranges. Output: a `COMPARISON.md` whose
rows point at the specific lines of code responsible for the observed cost, so
the next optimization step is a *directed* source edit, not a guess.

The Marlin path is the speed ceiling we're aiming for. The XFP path is what
we're trying to pull up. The per-range ratio in COMPARISON.md tells us WHERE
the gap actually lives — attention (unlikely, same backend), MoE-GEMM (likely
dominant), scatter (possibly), router/sort (possibly). Without this, every
kernel optimisation is a guess.

## Why NVTX + nsys (not `torch.cuda.Event`)

`Qwen3_5Model` is decorated with `@support_torch_compile`. `torch.cuda.Event`
objects inserted inside the compiled region would graph-break (they're not
tracer-friendly). `torch.cuda.nvtx.range_push/pop` is recognized by the
compiler as a side-effect op that does not alter graph shape and stays inline
with the captured kernels. In a non-profiled run the calls become no-ops by
virtue of the `VLLM_NVTX_PROFILE` gate in `vllm/multiquant/_profiler.py`.

Nsight Systems groups timelines by NVTX range name via
`nsys stats --report nvtxsum,cudaapisum,gpukernsum`. We read the NVTX ranges
to get phase-level budgets and the `gpukernsum` to verify kernel-to-phase
attribution.

## Runs

All runs use the same prompt (fixed seed in `bench.py`) and decode 200
tokens. nsys captures only the last 100 tokens (after CUDA Graph warmup) via
`--capture-range=cudaProfilerApi`.

| Run | Model | Weight dtype | KV dtype | LM head | Attention backend |
|---|---|---|---|---|---|
| A: XFP   | `Qwen3.5-122B-A10B` (our MultiQuant pack, on-the-fly) | XFP auto (bits=2/3/4 per layer) | fp8 | fp8 (via `--weight-dtype-lm-head fp8`) | FlashAttention |
| B: Marlin| `Qwen3.5-122B-A10B-int4-AutoRound` (`packing_format: auto_round:auto_gptq` → `fused_marlin_moe`) | INT4 AutoRound (Marlin) | fp8 | fp8 (via `--weight-dtype-lm-head fp8`) | FlashAttention |

**Only the weight quantisation differs between runs.** KV, LM head, attention
backend and model architecture are identical — so any per-range delta in
COMPARISON.md is attributable to the weight GEMM path (Marlin tensor-core +
LOP3 dequant vs XFP scalar + SHFL codebook dequant).

### Identical constants

- Seed: 42 (bench.py)
- `max_model_len` 32768
- `gpu_memory_utilization` 0.05 + `kv_cache_memory_bytes 10G`
- `VLLM_MLA_DISABLE=1`
- `FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"`
- `VLLM_NVTX_PROFILE=1`
- No speculative decoding (to isolate raw per-token cost)

## NVTX range name convention

Flat names, `/` for hierarchy. Each name maps 1:1 to a line range in
`SOURCE_MAP.md`.

- `layer_{N}` — whole decoder layer body (0..47)
- `layer_{N}/input_norm`, `/attn`, `/post_norm`, `/mlp` — layer-level phases
- `attn/qkv_proj`, `/qk_norm`, `/rope`, `/core`, `/o_proj` — attention internals
- `moe/xfp/sort`, `/gate_up_gemm`, `/silu`, `/down_gemm`, `/scatter` — XFP MoE
- `moe/marlin/align_block`, `/fused_kernel`, `/reduce` — Marlin outer dispatch
- `moe/marlin/gate_up_gemm`, `/silu`, `/down_gemm` — Marlin internals (parity with XFP)

Rationale for parity splits: even though Marlin has a single C++ call outer,
inside it calls two `moe_wna16_marlin_gemm` launches + an activation_func.
Marking them separately lets us compare `moe/*/gate_up_gemm` on both paths
directly.

## Capture workflow per run (two-stage)

**Stage 1 — E2E throughput via bench.py (deterministic, seed=42):**

```bash
# Documents tok/s across short/medium/long decode + math-%, no profiler overhead
python3 bench.py --url http://localhost:8011 --model <served_name> --label "<run_tag>"
```
Output: `tok/s long`, `tok/s medium`, `tok/s short`, `math-%`. This is the
**reliable throughput metric** — a single profiled call cannot give a stable
tok/s because nsys itself adds per-kernel sync overhead (~5-10%).
`bench.py` is NOT modified (per MEMORY: deterministic seed=42).

**Stage 2 — Per-phase kernel breakdown via nsys:**

```bash
# One 100-token decode captured with NVTX ranges
podman exec mq-serve bash -c 'nsys profile -t cuda,nvtx,osrt \
  -o /measurements/$RUN_TAG --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 -c "
import ctypes, requests
cudart = ctypes.CDLL(\"libcudart.so\")
cudart.cudaProfilerStart()
requests.post(\"http://localhost:8000/v1/completions\", json={\"model\":\"<served_name>\",\"prompt\":\"<fixed>\",\"max_tokens\":100}, timeout=120)
cudart.cudaProfilerStop()
"'
```

Output (`nsys stats --report nvtxsum,gpukernsum`):
- NVTX range totals — ms per range summed across all 100 decode steps
- GPU kernel summary — which kernels run inside each range

**Both stages identical for XFP and Marlin runs.** Run-summary markdown
records:
- bench.py numbers (tok/s + math-%) — headline throughput
- NVTX per-token averages (ms) — where the time goes
- Kernel-to-range attribution — what kernels deliver that time

## Post-processing

```bash
for tag in xfp marlin; do
  nsys stats --report nvtxsum,cudaapisum,gpukernsum $tag.nsys-rep \
      > $tag-stats.txt
done
```

Summarized into `xfp-summary.md` / `marlin-summary.md`: tables of
(range, total ms, ms/call, #calls, mean µs).

## Sanity checks (pre-publication)

1. All NVTX range names we inserted appear in `nvtxsum`.
2. Σ `layer_{0..47}` ≥ 0.90 × decode wall-clock time
   (else kernels are running outside our tagged ranges).
3. `gpukernsum` shows `xfp_moe_gemm_kernel` only inside `moe/xfp/*_gemm`
   ranges — and `moe_wna16_marlin_gemm` only inside `moe/marlin/*_gemm`.
4. Model produces coherent output on the test prompt in both runs.

## Out-of-scope

- Speculative decoding effects (MTP / ngram)
- Prefill cost — only decode steady-state
- CPU-side overhead (not what we're optimizing)
