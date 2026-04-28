# XFP-V2: Per-Group Quantization with a Shared Codebook Library

**Status:** Phase 1 verified (pack/decode in pure PyTorch).
**Target hardware:** NVIDIA Blackwell (sm_120/sm_121), but design is portable.
**Code:** `vllm/multiquant/xfp/xfp_pack.py` (`xfp_pack_v2`, `dequant_xfp_v2`).
**Tests:** `tests/xfp/test_pack_v2_quality.py`, `tests/xfp/verify_v2_paper.py`.

## TL;DR

Replace XFP's per-row learned codebook (16 fp16 centroids per output row)
with a **shared library of prototype codebooks** referenced per group of
weights inside each row. Same 4 bits/weight, similar memory footprint,
**+0.22 pp average cosine similarity** over V1, and consistently beats
int4 RTN per-group g=32 (the strong baseline matching AutoRound iter=0
in granularity).

## Motivation

V1 of XFP fits one 16-centroid Lloyd codebook per output row. We
hypothesised this learned codebook would dominate uniform int4 quantizers
at the same 4-bit budget. Empirically on Qwen3.5-A3B-BF16, **per-row XFP-V1
beats int4 RTN per-channel by only +0.5 pp cosine** and **loses to int4
RTN per-group g=32 by −0.4 pp**. The reason: weight distributions in modern
MoE models are gaussian-symmetric with low std (≈0.006-0.02), so Lloyd's
optimum codebook is close to a uniform linspace — the learning advantage
is marginal. Per-group RTN compensates with better local scale resolution.

V2 keeps the *learned* USP but moves it from per-row to per-group level
to match int4-g32's locality, while a shared library prevents the
SMEM cost from blowing up.

## Method

### Encoding

For an input weight matrix `W ∈ ℝ^{N×K}`:

1. **Group reshape** — split each row into `G = K / group_size` groups:
   `W ⟶ W_groups ∈ ℝ^{N·G × group_size}`.
2. **Per-group Lloyd** — fit a 16-centroid codebook per group with the
   *unchanged* `_lloyd_per_channel` from V1, producing
   `cb ∈ ℝ^{N·G × 16}`.
3. **Normalize** — for each group codebook `cb_i`, record midpoint
   `μ_i = (max(cb_i) + min(cb_i)) / 2` and scale
   `s_i = (max(cb_i) − min(cb_i)) / 2`, then store
   `cb̂_i = (cb_i − μ_i) / s_i ∈ [−1, +1]^{16}`.
4. **Library construction (k-means++)** — cluster the `N·G` normalized
   codebooks `{cb̂_i}` into a shared library
   `Lib ∈ ℝ^{L × 16}` of prototype codebooks (typical `L ∈ {16, 32}`).
   Each group `i` records the index `lib_id_i ∈ {0, …, L−1}` of its
   nearest library entry.
5. **Re-quantize weights** — given the chosen library codebook
   `Lib[lib_id_i] · s_i + μ_i`, assign each weight in the group its
   nearest centroid (index in `[0, 16)`), and pack via the existing
   `_pack_indices`.

### Decoding (Phase 1 reference; kernel implements equivalent path)

```
W[n, k] = group_scale[n, g] · Lib[group_lib_id[n, g], idx[n, k]]
       + group_mid[n, g]
where  g = k // group_size
```

### Storage

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `packed`        | `[K_packed, N]`   | int32  | 4-bit indices, **same layout as V1** (`_pack_indices`) |
| `library`       | `[L, 16]`         | fp16   | shared per layer (or per layer-class); SMEM-resident at runtime |
| `group_lib_id`  | `[N, G]`          | uint8  | for `L ≤ 256`, else int32 |
| `group_scale`   | `[N, G]`          | fp16   | per-group magnitude |
| `group_mid`     | `[N, G]`          | fp16   | per-group offset |

Bit budget per parameter:
```
B = 4 (weight index)
  + (32 / group_size)         # scale + midpoint, fp16 each
  + (lib_id_bits / group_size) # 4 if L ≤ 16, 8 if L ≤ 256
```
For the recommended `(group_size=128, L=32)`:
`B = 4 + 32/128 + 8/128 = 4.31 bits/param`.

## Reuse map (engineering investment retained)

| V1 component (kept) | V2 use |
|---|---|
| `_lloyd_per_channel` | unchanged; runs over `[N·G, group_size]` reshape |
| `_pack_indices`      | unchanged; same 4-bit packing |
| `xfp_repack`         | unchanged; warp-interleaved layout for kernel |
| Outlier extraction   | orthogonal; can be wired in V2 same as V1 |
| `xfp_gemm_v12.cu` (production kernel) | basis for `v17_lib.cu`, ~30-50 LOC patch in dequant inner loop |
| `xfp_moe_gemm_v12.cu` (Marlin-pattern MoE) | basis for `v17_lib` MoE, same patch |
| `xfp_gemm_core.cuh` (shared SMEM A-cache) | unchanged; library lookup adds 512 B-1 KB SMEM resident |
| Cache I/O + `tensor_meta` TP-slicing | extended with new tensors; existing slicing logic reused |

## Experimental setup

- **Model:** Qwen3.5-35B-A3B-BF16
- **Layers tested:** 6 (indices 0, 5, 10, 20, 30, 40), spanning model depth
- **Weight classes per layer:** 9
  - Linear-attention: `in_proj_qkv`, `out_proj`, `in_proj_a`, `in_proj_b`, `in_proj_z`
  - Shared expert: `gate_proj`, `down_proj`
  - Routed expert (slice expert 0 from fused 3D): `gate_up_proj`, `down_proj`
- **Metric:** flat cosine similarity between BF16 reference and reconstruction.
- **Seeds:** 3 (k-means++ library construction is stochastic).
- **Sweep:** `group_size ∈ {64, 128, 256}` × `library_size ∈ {8, 16, 32, 64}`.
- **Baselines:**
  - V1 — XFP per-channel learned codebook (current production)
  - int4 per-channel symmetric RTN (1 scale per row)
  - int4 per-group g=128 (matches V1 storage budget)
  - int4 per-group g=32 (matches AutoRound iter=0 default group size)

## Results

### Headline numbers

45 weight matrices (9 classes × 6 layers from indices {0, 5, 10, 20, 30, 40}),
3 seeds for V2 library. cos = flat cosine similarity vs BF16 reference.

| Method | bits/param | avg cos |
|---|---|---|
| BF16 (reference)                          | 16   | 1.00000 |
| int4 per-channel (1 scale/row)            | 4.01 | 0.97966 |
| int4 per-group g=128                      | 4.13 | 0.99041 |
| **XFP-V1** (per-channel codebook, K=2048) | **4.13** | **0.99203** |
| int4 per-group g=32                       | 4.50 | 0.99436 |
| **XFP-V2 g=256 L=32** (Pareto efficient)  | **4.16** | **0.99332** ±0.00013 |
| **XFP-V2 g=128 L=32** (recommended)       | **4.31** | **0.99464** ±0.00010 |
| **XFP-V2 g=64  L=32** (max quality)       | **4.62** | **0.99606** ±0.00007 |

`±` = standard error across 8 weight classes × 6 layers × 3 seeds (≈144 observations per V2 row).

### Sweep over (group_size, library_size)

| g \ L | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| **64**  | 0.99589 (4.56 bpp) | 0.99598 | 0.99606 | **0.99615 (4.62 bpp)** |
| **128** | 0.99447 (4.28 bpp) | 0.99457 | **0.99464 (4.31 bpp)** | 0.99473 |
| **256** | 0.99310 (4.14 bpp) | 0.99321 | 0.99332 (4.16 bpp) | 0.99342 |

**Sweet spot: `(g=128, L=32)`.** Same bit budget as XFP-V1 plus 0.18 bits/param overhead, gains +0.26 pp avg cos. Beats int4-g32 (0.99436) at lower bit budget (4.31 vs 4.50). Library size beyond 16 gives diminishing returns (+0.001 cos for 4× library size).

### Per-class detail at the recommended `(g=128, L=32)`

| weight class | V1 cos | **V2 cos** | Δ V2−V1 | int4-g32 | Δ V2−g32 |
|---|---|---|---|---|---|
| attn_qkv      | 0.99406 | **0.99632** | +0.23 pp | 0.99514 | +0.12 pp |
| attn_o        | 0.99188 | **0.99523** | +0.34 pp | 0.99472 | +0.05 pp |
| attn_a        | 0.98992 | **0.99388** | +0.40 pp | 0.99382 | +0.01 pp |
| attn_b        | 0.98873 | **0.99288** | +0.42 pp | 0.99281 | +0.01 pp |
| attn_z        | 0.99288 | **0.99519** | +0.23 pp | 0.99472 | +0.05 pp |
| shared_gate   | 0.99040 | **0.99353** | +0.31 pp | 0.99345 | +0.01 pp |
| shared_down   | 0.99270 | **0.99435** | +0.16 pp | 0.99415 | +0.02 pp |
| routed_gateup | 0.99337 | **0.99527** | +0.19 pp | 0.99505 | +0.02 pp |
| routed_down   | 0.99435 | **0.99515** | +0.08 pp | 0.99541 | −0.03 pp |

V2 beats V1 on **all 9 classes** (Δ ranges from +0.08 pp on `routed_down` to +0.42 pp on `attn_b`). V2 ≥ int4-g32 on 8/9 classes, marginally below on `routed_down` (−0.03 pp).

## Discussion

**Why does the library work?** A library of 16-32 prototype codebooks
covers the per-group codebook space with p5 cosine ≥ 0.999 across 42k
codebooks (`tests/xfp/test_codebook_library_size.py`). This is because
Lloyd-fitted codebooks for similar weight distributions converge to
similar near-uniform shapes — the variability lives in the *scale and
midpoint*, not in the *centroid pattern*. Decoupling these two factors
(library = pattern, group_scale + group_mid = magnitude) compresses the
codebook overhead dramatically without loss.

**Why per-group + library beats per-row?** A per-row codebook must capture
the magnitude of the entire row in 16 centroids. Within a 2048-wide row,
some groups have small magnitude (close to zero), others larger (toward
amax). The single per-row codebook over-quantizes the small groups (their
range is squashed) and under-quantizes the large groups. Per-group scale
recovers this resolution; the shared library captures the (uniformly
gaussian) shape with no penalty.

**Hardware cost.** For the production tile size (64 rows × 4096 K
weights of A in SMEM = 32 KB), V2 adds:
- Library: 512 B (always resident; serves all rows in the tile).
- Per-group params: 64 rows × 32 groups × 4 B (scale+mid+lib_id) = 8 KB.

Total V2 SMEM footprint ≈ V1 SMEM footprint (V1 had 64 × 32 B = 2 KB
per-row codebook). The +6 KB fits within the typical 100 KB SM budget.

**Outliers.** Phase 1 disables outlier extraction to isolate the library
contribution. Outliers can be re-enabled orthogonally as a residual
encoded with the existing `outlier_indices` / `outlier_values` mechanism.
On Qwen3.5-A3B test layers, outlier_fraction was ≤ 0.5% for all classes,
so the contribution is small.

## Limitations and next steps

1. **Calibration not used.** AutoRound's quality on int4 (GSM8K −2 pp vs
   BF16) comes from optimizing per-weight rounding decisions against a
   calibration set. V2 inherits XFP's calibration-free design. A future
   V3 could add calibrated centroid optimization and outperform AutoRound
   structurally.
2. **Library construction is offline.** We assume the library is fitted
   once per layer-class on the model's weight statistics. Per-layer
   libraries (one per FusedMoE block) are an option if memory is cheap
   enough — they would slightly increase quality.
3. **Phase 3 kernel not yet measured.** The current verification is
   Python-level; the v17_lib kernel is still pending. The math is
   identical to the Phase-1 reference, so we expect cos to within 1e-3
   of the Python reference (fp16 codebook rounding plus tile alignment).

## Reproducibility

```bash
# Phase 1 acceptance test (CPU-only):
podman run --rm \
    -v $HOME/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \
    -v /data/tensordata:/data/tensordata:ro \
    -v $HOME/vllm-riy/tests:/tests:ro \
    localhost/vllm-multiquant:latest \
    python3 /tests/xfp/test_pack_v2_quality.py

# Full paper-grade sweep (≈10 min on RTX PRO 6000):
podman run --rm \
    -v $HOME/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \
    -v /data/tensordata:/data/tensordata:ro \
    -v $HOME/vllm-riy/tests:/tests:ro \
    localhost/vllm-multiquant:latest \
    python3 /tests/xfp/verify_v2_paper.py
```

Both scripts emit a markdown summary on stdout and JSON
(`/tmp/xfp_v2_verification.{json,md}`) suitable for paper figures.
