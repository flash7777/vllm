# XFP Auto — Cosine-Similarity Gate: What, How, Why

This documents the quality floor used by `xfp_auto_select()` to pick the
minimum bit width per layer. Three questions, answered by reading the
source.

## TL;DR

- **Metric:** per-output-channel cosine similarity between the original
  weight row and its codebook reconstruction.
- **Aggregator:** **median** across all output channels of the layer. Not
  mean. Not min.
- **Gate:** `median_cos >= 0.98` (default). Iterate candidates
  `(2, 3, 4)` in order, return first that passes; fall back to the
  highest (4) if none does.
- **MoE:** compute on a sample of up to 4 experts concatenated along the
  K axis (not the full 256-expert stack).
- **Outlier split is applied before Lloyd** so the codebook fits the bulk
  distribution; outliers are put back into the reconstruction before the
  cos is measured.

## 1 — What is aggregated

### 1.1 Per-layer linear

For a linear weight `W ∈ R^{N_out × K}` (e.g. `qkv_proj`, `in_proj_qkvz`,
`gate_up_proj`):

```python
# vllm/multiquant/xfp/xfp_pack.py:405–409
cos_per_ch = F.cosine_similarity(W, rec, dim=1)  # [N_out]
median_cos = float(cos_per_ch.median().item())
if median_cos >= min_cos:
    return bits
```

- `W`, `rec` are both `[N_out, K]`.
- `F.cosine_similarity(..., dim=1)` produces one cosine **per output
  channel** (one value per row of the weight matrix).
- The aggregator is `.median()`. On a layer with 8192 output channels,
  the median is the 4096-th ranked channel's cosine.

### 1.2 MoE experts (sample-based)

MoE layers have `[E, N, K]` stacked experts. The auto-select runs on a
sample concatenated along the K axis:

```python
# vllm/multiquant/xfp/online_moe.py:251–263
if bits == 0:
    sample_experts = min(4, E)  # 4 of up to 256
    sample = w13[:sample_experts].reshape(-1, w13.shape[2]).float()
    bits = xfp_auto_select(
        sample,
        candidates=(2, 3, 4),
        min_cos=self.quant_config.auto_min_cos,
        lloyd_iters=moe_lloyd_iters,
    )
```

The sampled matrix is `[sample_experts × N, K]` (e.g.
`[4 × 2048, 3072] = [8192, 3072]` for Qwen3.5-122B routed experts). The
per-channel cos is still computed per output row — now the rows are
expert-0-row-0 … expert-0-row-N, expert-1-row-0 …, so the median
effectively spans experts too. Expert-to-expert homogeneity is high
(§6.1 of PAPER_XFP: inter-expert codebook cos 0.99999), so 4 experts
is a good-enough sample and the whole 256-expert group gets the same
`bits` decision (MoE auto-mode is *group-wise*, not per-expert).

### 1.3 Total element count used to decide

| Example layer | Elements going into cos |
|---|---|
| Qwen3.5-122B full-attn `qkv_proj` (3072×8192) | 25.2 M |
| GatedDeltaNet `in_proj_qkvz` (20480×3072) | 62.9 M |
| Qwen3.5-122B MoE routed (sample 4/256 × 2048×3072) | 25.2 M |
| Qwen3.5-122B MoE routed (if we used all 256) | 1.6 G |

The 4-expert sampling reduces the MoE decision time by ~64× with no
measured quality difference.

## 2 — How cos is computed

### 2.1 The reconstruction path

```python
# vllm/multiquant/xfp/xfp_pack.py:391–403
for bits in candidates:                    # (2, 3, 4)
    n_centroids = 1 << bits                # 4, 8, 16
    cb = _lloyd_per_channel(W_bulk, n_centroids, lloyd_iters)  # [N_out, 2^bits]
    idx = _assign_indices(W_bulk, cb)      # [N_out, K] int64
    rec = torch.gather(cb, 1, idx)         # [N_out, K]
    if mask is not None:
        # put outliers back into the reconstruction
        flat_r = rec.reshape(-1).clone()
        flat_r[mask.reshape(-1)] = W.reshape(-1)[mask.reshape(-1)]
        rec = flat_r.reshape_as(W)
    # ...
    cos_per_ch = F.cosine_similarity(W, rec, dim=1)
```

Steps, in order:

1. **Outlier split** (once, shared across all candidate bit widths).
   `W → W_bulk` where outliers (`|w − μ| > 4σ`, capped at 2% of elements)
   are replaced by `μ`. The outlier mask is stored.
2. **Lloyd-Max per output channel**, 20 iterations (5 for MoE), minimax
   linspace init with 1e-6 jitter.
   Produces `cb: [N_out, 2^bits]`.
3. **Assignment** via 1-D argmin → `idx: [N_out, K]`.
4. **Reconstruction** via `torch.gather(cb, 1, idx)`.
5. **Outlier patch-in:** `rec[outlier_mask] = W[outlier_mask]`. This is
   what the **sparse-fp8 path** will carry at inference — the cos the
   gate sees is the *final* reconstruction quality, not the codebook-only
   quality.
6. **Cosine** along axis 1 → per-channel similarity.

### 2.2 Why per-channel, not flat

An alternative would be `cosine_similarity(W.flatten(), rec.flatten())` —
one scalar for the whole matrix. We reject this:

- One scalar is dominated by the largest-magnitude channels. A small
  cluster of well-reconstructed high-norm channels can mask a majority
  of badly-reconstructed low-norm channels.
- `cos_per_ch` preserves the per-channel structure that actually matters
  for downstream quality: each output channel maps to one GEMM column at
  inference, so preserving *directional* alignment per column is what
  keeps the next layer's activations on-distribution.

### 2.3 Why median, not mean

- **Robust to outlier channels.** A single catastrophically bad channel
  (cos ~0.5) drags `mean_cos` down hard; the rest of the layer may be
  fine. The median rejects that.
- **Robust to lucky channels.** Conversely, a few trivially easy
  channels (cos ~0.9999) can inflate the mean on an otherwise mediocre
  fit. The median rejects that too.
- **Matches the decision we want.** The question is "do the majority of
  channels reconstruct well?" — that's literally the median.

Min was considered and rejected: at N=2 a handful of pathological
channels always exist; gating on min would force xfp3 everywhere and
defeat the auto-mode.

### 2.4 MSE was considered and rejected

An earlier version used MSE ratio (MSE@bits vs MSE@4bits) as the gate.
Rejected because:

> On real MoE models, XFP2 has ~12× higher MSE than XFP4 but identical
> math accuracy (the error is spread uniformly and doesn't concentrate
> in model-critical channels). Cos similarity captures this: it measures
> directional preservation per channel, which is what matters for
> downstream quality.

— `xfp_pack.py:346–351` comment.

MSE is magnitude-sensitive, cos is direction-sensitive. In a GEMM, the
output is `x · W_col` — directional preservation of `W_col` matters more
than exact magnitude preservation (activations can absorb
magnitude-scaling via learned norms).

## 3 — Why 0.98 as the default

### 3.1 Configurable

```python
# vllm/multiquant/xfp/online_linear.py:185–188
# Auto bit-width selection: minimum per-channel cosine similarity
# for a candidate bit width to be accepted. Configurable via
# XFP_MIN_COS environment variable. Default 0.98 — calibrated on
# GLM-4.7-Flash where it separates routed experts (xfp3 OK at
# cos 0.982) from attention (needs xfp4 at cos 0.994).
auto_min_cos: float = float(os.environ.get("XFP_MIN_COS", "0.98"))
```

Set a different gate via `XFP_MIN_COS=0.985` at launch time.

### 3.2 Empirical calibration (GLM-4.7-Flash)

The 0.98 default was picked after the 942K-codebook analysis showed the
bimodal landing of `median_cos` across layer classes:

| Component | Median cos at xfp3 (GLM) | xfp3 acceptable? |
|---|---:|---|
| routed experts | 0.982 | yes — 0.002 over threshold |
| shared experts (gate/up/down) | 0.982–0.984 | yes |
| dense MLP | 0.982–0.984 | yes |
| attention (most) | 0.983–0.988 | yes |
| attention `attn_qb` (2 layers) | < 0.98 | **no** → escalates to xfp4 |

Lower than 0.98 would admit `attn_qb` at xfp3 and collapse GLM math
accuracy (66% → 30% in uniform-xfp3 runs). Higher than 0.98 (e.g. 0.985)
would push routed experts to xfp4 and lose the 25% memory win.

**0.98 is the knee point** where 99% of GLM layers land at xfp3 and only
the ~2 sensitive ones get the xfp4 escalation they genuinely need.

### 3.3 Transfer to other models

- On **Qwen3.5-122B-A10B**, the same 0.98 picks xfp3 for 36 GatedDeltaNet
  layers + 12 full-attention + 48 shared-expert blocks, and **xfp4** for
  routed experts (which have broader distributions on Qwen than on GLM).
  Math stays at 98%. Effective bits ≈ 3.97 (routed experts dominate the
  parameter mass).
- On **GLM-4.7-Flash**, the same 0.98 picks xfp3 for 99% of layers with
  ~3.0 effective bits and 54% GSM8K (BF16 baseline).

The threshold is calibrated, but the **algorithm is identical** across
model families. The auto-mode's job is to find the bit width that meets
the gate, not to decide the gate.

### 3.4 Knob for quality-conservative deployment

If quality must stay closer to BF16 than the default:

```bash
XFP_MIN_COS=0.99 ./start.multiquant --model <m> --weight-dtype xfp ...
```

This drives most GLM layers to xfp4 (avg ≈ 3.8 bits), Qwen routed
experts to xfp4, and attention to either xfp3 or xfp4 depending on the
class. Quality should rise 1–2 pp on accuracy benchmarks at the cost of
~25% more weight memory.

Conversely, to push harder toward compression:

```bash
XFP_MIN_COS=0.975 ./start.multiquant ...
```

May drop GLM attention to xfp3 universally (including `attn_qb`) and
accept quality loss. Not recommended without a task-specific re-measure.

## 4 — Sanity checks / gotchas

1. **Shape contract.** `xfp_auto_select` requires `W.dim() == 2`. MoE
   stacks `[E, N, K]` are reshaped to `[sample_experts*N, K]` before the
   call — see §1.2.

2. **Lloyd iteration count matters for the cos value.** The default is
   `lloyd_iters=20` for linear layers and `5` for MoE sample. Using `3`
   or less produces a cos that's 0.003–0.005 pessimistic; using `40`
   gives essentially the same cos as `20`. 20/5 is the sweet spot — one
   full pass already converges the codebook within 1e-4 relative change.

3. **Outlier split is baked in.** The cos is measured *after*
   putting outliers back into `rec` (line 400–403 of `xfp_pack.py`). If
   you disable outlier handling by passing `outlier_sigma=None`, the cos
   will drop by ~0.002–0.004 on attention layers (they rely on the sparse
   path for that last tenth of a percentage point).

4. **MoE sample size matters only for heterogeneous MoE.** 4 experts is
   enough because inter-expert codebook cos is 0.99999 on GLM/Qwen — the
   codebook learned on 4 is essentially the same as on 256. If you
   observe heterogeneity in a new model, increase `sample_experts`.

   **Empirical validation 2026-04-22** (see
   `measurements/20260421-moe-sample-validation/VALIDATION_REPORT.md`):
   | Model | Experts | Disagreement first-4 vs full |
   |---|---:|---:|
   | Qwen3.5-35B-A3B | 256 | **0 %** |
   | GLM-4.7-Flash | 64 | **0 %** |
   | Qwen3.5-122B-A10B | 256 | **11.5 %** (10 Fall-A + 1 Fall-B) |

   For the 122B case the disagreement is Fall-A (sample too tame,
   under-escalates to xfp3 where full would pick xfp4). Math on GSM8K
   stays at 98 % regardless, because the outlier-fp8 path absorbs the
   single-bit loss. If you see math regression on a new >128-expert
   model, raise `sample_experts` to 16 or switch to stratified sampling
   (norm-quantile selection).

## 5 — Where to look in the code

| File | Lines | Role |
|---|---:|---|
| `vllm/multiquant/xfp/xfp_pack.py` | 332–412 | `xfp_auto_select()` — the gate |
| `vllm/multiquant/xfp/xfp_pack.py` | 81–152 | `_lloyd_per_channel()` — codebook fit |
| `vllm/multiquant/xfp/xfp_pack.py` | 155–200 | `_assign_indices()` — RTN assignment |
| `vllm/multiquant/xfp/xfp_pack.py` | 261–271 | `_reconstruction_stats()` — MSE/cos helper (non-auto path) |
| `vllm/multiquant/xfp/online_linear.py` | 180–188 | `auto_min_cos` default + env var |
| `vllm/multiquant/xfp/online_moe.py` | 247–265 | MoE auto-mode: sample 4 experts, call auto-select |

Commit `683d80d8b` (2026-04-21) is the current snapshot these references
track.
