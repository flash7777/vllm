# XFP v12 Kernel — E2E Results

**Image:** `localhost/vllm-multiquant:xfp_speed` (NGC 26.03-py3 base,
vLLM 0.17.1, MultiQuant)
**Kernel:** `kernels/multiquant/xfp_moe_gemm_v12.cu` (bf16-native,
A-row SMEM cache K_SMEM_MAX=4096, `MoEPolicy`/`LinearPolicy` templates)
**Policy:** includes `linear_attn` classification fix
(`vllm/multiquant/policy.py:141`)
**Bench:** `bench.py` seed=42, n=5 decode, GSM8K 50 problems
**Config (all runs unless noted):** fp8 KV-cache + fp8 LM-head

Three models get the same treatment: XFP auto-mode + Marlin INT4 AutoRound
on identical hardware, identical fp8 KV/LM-head, same bench.

---

## 1 — Qwen3.5-122B-A10B (DGX Spark, GB10, SM121a)

**Architecture:** hybrid MoE. 48 decoder layers (36 linear_attention
GatedDeltaNet + 12 full_attention) + 47 MoE blocks (256 routed experts +
BF16 shared expert per layer) + 1 dense MLP at layer 0 + fp8 LM-head
(248,320 vocab). Served from BF16 checkpoint with quantize-on-load.

### 1.1 Throughput + Math

| Configuration | long (400t) | medium (150t) | short (20t) | Math | Eff. bits | Notes |
|---|---:|---:|---:|---:|---:|---|
| XFP v1 (attn-only quant, pre-fix) | 17.3 | 18.9 | 2.5 | 98% | ~3.97 | linear_attn in BF16 — 11.22 ms/tok bf16-GEMV |
| **XFP v2 (linear_attn quantized)** | **29.9** | **34.8** | **2.6** | **98%** | **~3.97** | classifier fix unlocks 108 GatedDeltaNet projections |
| Marlin INT4 AutoRound | 25.8 | 29.6 | 2.5 | 94% | 4.00 | Intel `auto_round:auto_gptq`, fused_marlin_moe path |
| **XFP v2 / Marlin ratio** | **1.16×** | **1.18×** | ~1× (latency-bound) | **+4 pp** | — | — |
| **XFP v2 + MTP NST=3** | **32.7** | **37.2** | **2.5** | **98%** | **~3.97** | qwen3_5_mtp.py load_weights fix + q/k/v shard fuser |
| **XFP v2 + MTP / Marlin** | **1.27×** | **1.26×** | ~1× | **+4 pp** | — | — |

XFP v2 **beats Marlin on both throughput AND math accuracy** at the same
average bit width (routed experts dominate at xfp4 either way; XFP's win
is the learned per-channel codebook + xfp3 for the hybrid attention stack
and shared experts). **MTP NST=3** adds another +9.4 % long-context
throughput with math fully preserved — volle Matrix (NST=1..5) in
`measurements/20260421-xfp-mtp-verify/COMPARISON-MTP.md`.

### 1.2 Per-kernel profile (torch.profiler, 60 decode steps)

Profile captured pre-linear_attn-fix (XFP v1) vs Marlin, to isolate *where*
the decode-time delta lives. Kernels classified into categories, per-step
averages:

| Category | XFP v1 (ms/tok) | Marlin (ms/tok) | Δ |
|---|---:|---:|---:|
| cuBLAS BF16-GEMV (unquant layers) | 11.22 | 5.18 | −6.04 |
| Weight-GEMM MoE | 3.88 | 3.73 | −0.15 |
| Weight-GEMM Linear | 1.32 | 3.48 | **+2.16** |
| cutlass_wmma (fp8 LM-head on Marlin) | 0.36 | 2.46 | +2.10 |
| Attention-Core (flash_fwd / flashinfer) | 1.20 | 0.00 | −1.20 |
| SiLU (triton_poi / act_and_mul) | 0.03 | 0.44 | +0.41 |
| Scatter | 0.85 | 0.01 | −0.84 |
| Elementwise | 0.91 | 0.11 | −0.80 |
| Mamba/GDN recurrent | 0.26 | 0.26 | 0.00 |
| Top-k + reduce | 0.22 | 0.51 | +0.29 |
| Other | 0.56 | 0.42 | −0.14 |
| **Total GPU** | **20.79** | **16.59** | **−4.20** |

**Readout:**
1. **XFP MoE GEMM ≈ Marlin MoE GEMM** (3.88 vs 3.73 ms/tok) — the expected
   compute-density gap from SHFL+codebook vs LOP3+MMA does not materialize
   at M=1 decode. Both paths are HBM-bandwidth-bound on the weight read.
2. **XFP Linear GEMM is 2.64× FASTER** than Marlin's linear kernel
   (1.32 vs 3.48 ms/tok) — 3-bit weights read ~25% fewer bytes than
   Marlin's 4-bit and the memory wall translates directly into throughput.
3. **The 11.22 ms/tok bf16-GEMV in XFP v1 is entirely the unquantized
   linear_attn layers** (confirmed via kernel-shape analysis). The one-line
   classifier fix moves these into `xfp_gemm` and closes the bench-level
   gap to Marlin + gives XFP the lead.

### 1.3 Math accuracy detail

XFP v2 math failures (98 %, 49/50 correct, 1 error):
```
90 − 575 = −485, got: 485.
```
(single sign error, no arithmetic mistake)

Marlin math failures (94 %, 47/50 correct, 3 errors):
```
469 − 613 = −144, got: 144
859 × 653 = 560927, got: 561927
243 − 801 = −558, got: 558
```
(Marlin loses both signs AND one multiplication.)

### 1.4 Reproduction commands

```bash
# XFP v2 (linear_attn quantized — current production)
./start.multiquant --model Qwen3.5-122B-A10B \
    --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8

# Marlin INT4 AutoRound (direct comparison)
./start.multiquant --model Qwen3.5-122B-A10B-int4-AutoRound \
    --kv fp8 --weight-dtype-lm-head fp8

# Bench (seed=42 deterministic)
python3 bench.py --url http://localhost:8011 --model glm-4.7-flash \
    --label "<tag>"
```

### 1.5 Artefacts

- `measurements/20260419-xfp-vs-marlin/COMPARISON.md` — full analysis
- `measurements/20260419-xfp-vs-marlin/xfp.trace.json.gz` — XFP v1
  torch.profiler chrome trace (3.9 MB, 60 decode steps, pre-fix)
- `measurements/20260419-xfp-vs-marlin/marlin.trace.json.gz` — Marlin
  torch.profiler chrome trace
- `measurements/20260419-xfp-vs-marlin/xfp-bench.txt`,
  `xfp-bench-linearattn.txt`, `marlin-bench.txt` — raw bench.py outputs
- Source fix: `vllm/multiquant/policy.py:141` (commit `683d80d8b`)

---

## 2 — Qwen3.5-35B-A3B (DGX Spark, GB10, SM121a)

**Architecture:** hybrid MoE smaller variant. To be filled.

### 2.1 Throughput + Math

| Configuration | long | medium | short | Math | Eff. bits | Notes |
|---|---:|---:|---:|---:|---:|---|
| XFP v2 (linear_attn quantized) | TBD | TBD | TBD | TBD | TBD | |
| Marlin INT4 AutoRound | TBD | TBD | TBD | TBD | 4.00 | |
| Ratio | | | | | | |

### 2.2 Artefacts

- Pending.

---

## 3 — GLM-4.7-Flash (DGX Spark, GB10, SM121a)

**Architecture:** 30B parameters, MoE 64 experts × 46 layers, pure
Transformer self-attention (no hybrid linear_attn — policy coverage
already complete on this model).

### 3.1 Throughput + Math

| Configuration | long | medium | short | Math | Eff. bits | Notes |
|---|---:|---:|---:|---:|---:|---|
| XFP v2 auto (cos≥0.98) | TBD | TBD | TBD | TBD | ~3.0 | GLM-specific: auto converges to xfp3 for 99% of layers |
| XFP4 uniform (v8 legacy) | 32.7 | — | — | 66% (33/50) | 4.0 | historical, v8 kernel, pre-fp8-LMH |
| Marlin INT4 AutoRound (historical) | 53.6 | — | — | 54% | 4.0 | v8-era, different config |
| FP8 pre-quantized (historical) | 28.2 | — | — | 60% | 8.0 | |
| BF16 unquantized (historical) | 26.8 | — | — | 54% | 16.0 | |

The historical GLM Marlin 53.6 tok/s was on a different config (no
fp8-LMH, v8-era kernel). A direct apples-to-apples re-run with v12 +
fp8 KV + fp8 LM-head is **pending** and will replace these rows.

### 3.2 Artefacts

- Pending (v12 re-run).

---

## 4 — Cross-model summary (populated as runs complete)

| Model | Params | Arch | XFP v2 long | Marlin long | XFP / Marlin |
|---|---:|---|---:|---:|---:|
| Qwen3.5-122B-A10B | 122B | hybrid MoE | **29.9** | 25.8 | **1.16×** |
| Qwen3.5-35B-A3B | 35B | hybrid MoE | TBD | TBD | TBD |
| GLM-4.7-Flash | 30B | transformer MoE | TBD | TBD | TBD |

Math-accuracy column to be added when full matrix is populated.
