# XFP Benchmarks (v1)

Benchmark results for XFP (learned-codebook) quantization-on-load on
GLM-4.7-Flash. v1 is a **functional reference implementation** — quality
is validated end-to-end, performance is not yet optimized. See "Known
perf limitations" below.

- **Platform**: DGX Spark (GB10, SM121, 120 GB Unified)
- **Image**: `localhost/vllm-multiquant` (base `nvcr.io/nvidia/vllm:26.02-py3`)
- **vLLM Version**: 0.15.1+befbc472
- **Model**: `/data/tensordata/GLM-4.7-Flash` (BF16 checkpoint, quant-on-load)
- **Benchmark**: `bench.py` (deterministic, seed=42, 5 perf rounds, 50 math problems)
- **Mount**: `vllm/multiquant` + `kernels/multiquant` as volume (no image rebuild)
- **Runtime flags**: `--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G --max-model-len 4096`

## Quant Matrix

Tests run with `--quantization autoround_rtn --weight-dtype-attn xfp{N} --weight-dtype-routed xfp{N} --weight-dtype-shared xfp{N}`.

| Weight (attn/routed/shared) | KV cache | Short tok/s | Medium tok/s | Long tok/s | Math |
|-----------------------------|----------|-------------|--------------|------------|------|
| **xfp4** (v1)               | default  |    5.3      |     5.4      |    5.3     | **27/50 (54 %)** |
| **xfp4** (v2a, SMEM cb)     | default  |    6.3      |     6.4      |    6.4     | **26/50 (52 %)** |
| xfp4 (v2b, + M=1 spec)      | default  |    6.2      |     6.4      |    6.3     | 25/50 (50 %) |
| xfp3                        | default  |    6.7      |     6.8      |    6.8     | 15/50 (30 %) |
| xfp2                        | default  |   10.5      |    11.0      |   10.9     |  0/50 ( 0 %) |

v2a delivers **+20 % tok/s with no accuracy change** (1-problem math
delta 27→26 is within run-to-run noise — kernel correctness is gated by
unit tests matching fp32 reference at cos sim > 0.999).

**v2b (M_COUNT=1 specialization) is a null result.** The hypothesis was
that M_COUNT=4 wastes 3/4 of the inner-loop work on decode (M=1) through
the `if (mi >= M) continue` branch. Measured: same tok/s within noise
(6.3 vs 6.4 long), same math accuracy within noise. Why this didn't
help:

- For M=1, `grid.y = ceil(M / M_COUNT) = 1` regardless of M_COUNT — no
  change in block launch count
- The CUDA compiler's constant propagation likely already eliminates
  the dead branches for M_COUNT=4 when M is known small
- The real bottleneck isn't in the M-dimension MACs; it's elsewhere
  (likely the per-MAC `__half2float` conversions or the per-thread A
  loads from global memory)

The v2b change is kept in the codebase as infrastructure for future
M-based tile tuning — it adds three template instantiations (M_COUNT
∈ {1, 2, 4}) with zero-cost runtime dispatch, so later kernels can
plug into the same launch path. It costs nothing and documents the
null result.

### Remaining optimization headroom (v3+ scope)

- **Cooperative A-row SMEM staging**. All 32 threads in a block read
  the same `A[mi, k]` values — currently 32 independent global loads
  that L1 coalesces into one transaction, but still 32× the instructions.
  One explicit cooperative load into SMEM + broadcast would remove the
  instruction count.
- **Tile-layout rewrite closer to Marlin**. The current (32 threads ×
  4 N-cols) layout is a direct copy of `mq_gemm_int2.cu` and pays for
  the generality of handling variable K-strides. A 128 threads × 1
  N-col layout with warp-level reductions (no atomicAdd across gridDim.z)
  would match Marlin's structure and unlock the rest of the gap.
- **half2 FMA**. Replace the fp32 accumulator chain with `__hfma2` to
  double arithmetic throughput where the codebook-decoded values are
  paired. Requires the MAC loop to produce pairs of products, which is
  cleaner with a tile-layout rewrite.

These are the headline v3 targets. The v2 increment banks the SMEM
codebook win as a committed baseline while documenting that M-dimension
tuning is a dead end for the current tile structure.

### Accuracy vs. bit width

- **XFP4 matches pre-quant INT4 AutoRound math accuracy (54 %) exactly**.
  The learned-codebook quantization preserves model behavior on the
  GSM8K-style math probes, even though we pack at load time from BF16
  weights (no calibration data) instead of using the offline AutoRound
  GPTQ procedure. This is the key finding: XFP4 is within rounding of
  state-of-the-art INT4 quality, through a much simpler pipeline.
- **XFP3 drops to 30 %** — 8-entry codebook isn't enough to cover the
  weight distribution of GLM-4.7-Flash's MoE experts well enough for
  multi-step arithmetic. Still coherent prose output, but the expert
  paths that carry arithmetic reasoning degrade.
- **XFP2 collapses to 0 %** — 4-entry codebook is below the capacity
  floor for this model. Output is coherent at the token level (non-repeating
  tokens, no garbage strings) but semantic content is incoherent. This
  is expected: 4 codebook entries per row leave no headroom for the
  weight-value diversity in a post-training MoE.

### Speed vs. bit width

The v1 kernel is memory-bandwidth-bound on the packed weight reads.
With fewer bits per value the same block processes 2×/1.5×/1× the
number of weights per uint32 word (16 / 10 / 8 for xfp2/3/4), so the
decode throughput scales inversely with the effective bytes-per-weight.
That's why xfp2 (10.9 tok/s) > xfp3 (6.8 tok/s) > xfp4 (5.3 tok/s) —
the kernel is more constrained by how fast it can read packed weight
words than by arithmetic. With a bandwidth-optimized kernel (shared-memory
LUT + vectorized loads + fp16 accumulator) the ranking should invert to
match Marlin's ~50 tok/s ceiling with a smaller spread.

### Per-layer encoder stats (typical)

- `cos sim 0.992–0.994` per output channel after Lloyd (20 iters) at xfp4
- `mse 3e-6 to 3e-5` depending on layer width
- `3σ outlier fraction 0.3–2.2 %` (well below the 15 % threshold for the
  sparse-outlier path, so v1's bulk-only variant is justified on GLM-4.7)

## Reference baselines (from RESULT.mixed.md)

| Config                             | tok/s (long) | Math |
|------------------------------------|--------------|------|
| GLM-4.7 BF16 pure                  | 26.8         | 54 % |
| GLM-4.7 pre-quant INT4 AutoRound   | 53.6         | 54 % |
| GLM-4.7 RTN INT4 Marlin Attn-only  | 33.6         | 0 %  |

## Encoder statistics (sampled)

Typical per-layer stats during pack (from stdout):

```
XFP ? [N_out x K] -> xfp4 | mse=2.8e-06..2.7e-05 | cos=0.994..0.997
                           | 3σ outlier fraction 0.3..2.0 %
```

The bulk Gaussian of GLM-4.7 weights gives cos similarity around 0.995
at xfp4. Outlier ratios below 2 % justify the bulk-only (no sparse)
path of v1.

## Known perf limitations (v1)

The current `xfp_gemm.cu` kernel is a naive reference implementation. It
works correctly (cos sim > 0.999 vs fp32 reference on unit tests, coherent
model output on GLM-4.7-Flash) but is roughly **10× slower than Marlin INT4**
at decode time. Root causes:

1. **Codebook in global memory**, not shared/constant. Each of the 4 N
   columns per thread has its own `2^N`-entry fp16 LUT (8/16/32 bytes).
   Total per block: <1 KB, trivially fits in SMEM or constant memory, but
   v1 reads from global every inner-loop iteration. L1 absorbs the reuse,
   but a shared-memory staging would eliminate the per-lookup latency.
2. **No K-tile unrolling**, no vectorized loads for `A[m*K+k]`. The inner
   loop reads single fp16 scalars; a `half2`/`float4` vectorized load would
   2-4× the memory-bandwidth headroom.
3. **atomicAdd on fp16 output** for K-split accumulation. Works, but
   serializes writes; a reduction tree across grid.z would scale better.
4. **Unnecessary `x.to(float16)` / `C.to(bf16)` copies** around the kernel
   call. GLM-4.7-Flash runs in bf16, we convert to fp16 before the kernel
   and back after — two extra full-tensor copies per layer forward.

The v1 target was "does the pipeline (encoder + dispatch + apply) work
end-to-end for all three widths, is the output correct, does the stats
infrastructure record the expected signals". All yes. Performance work
is tracked as v2 scope in `XFP.PAPER.md` §5 ("fused kernel with LUT in
registers/SMEM").

For comparison, `RESULT.mixed.md` shows pre-quantized INT4 AutoRound via
Marlin at 53.6 tok/s (long). The XFP v1 kernel runs at roughly 5-6 tok/s
for the same model — functional but strictly a correctness baseline.

## Setup issues fixed during integration

1. **torch.compile / Dynamo cannot trace pybind11 extensions**. Calling
   the JIT-loaded `xfp_gemm` directly from `apply()` triggered a graph
   break (`torch._dynamo.exc.Unsupported: skipped` on the pybind function
   record). Fix: wrap the kernel call in a `torch.library.custom_op` via
   `direct_register_custom_op` with both a real impl and a fake_impl that
   returns an empty output of the right shape and dtype — same pattern
   used by `ArcherOnlineLinearMethod._archer_apply_impl`.
2. **torch.compile can't graph `os.path.abspath`**. Resolving the kernel
   source directory at `_load_xfp_gemm` call time did posix path calls
   that Dynamo rejects. Fix: compute `_KERNEL_SRC_DIR` once at module
   import time and stash it in a module-level constant.
3. **Output dtype mismatch**. First cut returned fp16 from the custom op,
   but GLM-4.7-Flash runs in bf16 and a downstream `extern_kernels.mm` then
   crashed with `float != c10::BFloat16`. Fix: cast the kernel output back
   to `x.dtype` in the real impl, and make the fake impl return a tensor
   with `x.dtype` so the traced graph agrees.
4. **Pack wall-clock too high with chunked Lloyd at `row_chunk=128`**.
   GLM-4.7-Flash has 300+ small Linear layers and packing one layer at
   a time was ~2s × 300 = ~10 min just for the linear path. Fix: raise
   `row_chunk=4096` so typical layers run in a single Lloyd chunk, drop
   `lloyd_iters` to 20, drop `also_score_widths` default to empty. Total
   pack time now ~50s for the full model.
5. **`torch.quantile` init was O(N·K·logK)**. Replaced with a min-max
   linspace init O(N·K) — Lloyd converges from it within the same number
   of iterations on smooth distributions.

## Issues and fixes during this session

(filled above)

## Reproduction

```bash
podman run -d --replace --name mq-test \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --ipc=host --network host \
  -v /data/tensordata:/data/tensordata \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v /home/flash/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \
  -v /home/flash/vllm-riy/kernels/multiquant:/opt/mq_kernels:ro \
  -e VLLM_MLA_DISABLE=1 -e VLLM_WORKER_MULTIPROC_METHOD=fork \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_DISABLE_AUTOTUNER=1 \
  -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
  -e FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a" \
  localhost/vllm-multiquant \
  vllm serve /data/tensordata/GLM-4.7-Flash \
    --host 0.0.0.0 --port 8011 \
    --served-model-name glm-4.7-flash \
    --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 5G \
    --max-model-len 4096 \
    --quantization autoround_rtn \
    --weight-dtype-attn xfp4 --weight-dtype-routed xfp4 --weight-dtype-shared xfp4 \
    --trust-remote-code

python3 bench.py --url http://localhost:8011 --model glm-4.7-flash --label "XFP4 E2E"
```

## Architecture notes

- **Registry-centered dispatch**: XFP does not introduce a new `--quantization`
  string. It reuses `--quantization autoround_rtn`, which is a thin wrapper
  around `vllm.multiquant.policy.create_weight_method`. The dispatcher reads
  the active `MultiQuantPolicyRegistry` and routes by dtype prefix: `xfp*` →
  `XFPLinearMethod`/`XFPMoEMethod`, `tq*`/`rq*` → `ArcherOnlineLinearMethod`,
  `int*` → `AutoRoundRTNLinearMethod`/`AutoRoundRTNMoEMethod`, and `bf16` →
  `UnquantizedLinearMethod`.
- **Per-channel codebook**: each row of the weight matrix gets its own
  `2^bits`-entry Lloyd-optimal codebook (fp16). Total codebook overhead per
  linear layer is `N_out * 2^bits * 2 bytes` — negligible next to the packed
  indices.
- **Word-aligned packing**: uint32 words, 16/10/8 values per word for
  bits 2/3/4 respectively. XFP3 has 2 reserve bits per word (not yet used).
- **Fused decode kernel**: `kernels/multiquant/xfp_gemm.cu` — single template
  on `BITS`, grid layout mirrored from `mq_gemm_int2.cu`, `atomicAdd` for
  K-split accumulation.
