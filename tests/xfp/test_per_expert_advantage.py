"""Verify XFP's per-expert codebook advantage on real MoE weights.

Hypothesis (user): per-expert codebook is the KILLER feature of XFP —
each expert's weight distribution can have its own optimized centroids,
which should DOMINATE a uniform int4 quantizer at the same bit budget.

If XFP-4 per-expert ≈ int4-RTN per-channel (within 1-2% cos), then the
per-expert codebook isn't actually doing what it's advertised to do —
implementation bug, possibly in the GEMM kernel that uses these
codebooks differently than dequant_xfp does.

Test setup (35B-A3B routed_experts):
  Pick first 8 routed-MoE experts from layer 0.
  For each expert (a separate weight matrix per HF tensor):
    - XFP-4 pack: learns its own per-channel codebook
    - int4-RTN: uniform symmetric per-channel
    - measure cos(W, W_rec) and y = x @ W.T cos
  Aggregate over experts → mean cos, max cos drop.

Result interpretation:
  XFP cos > int4 cos by ≥ 1pp on average → per-expert advantage real
  XFP cos ≈ int4 cos                       → no advantage, bug suspected
  XFP cos < int4 cos                       → broken, definite bug
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.multiquant.xfp.xfp_pack import xfp_pack, dequant_xfp


MODEL_DIR = os.environ.get(
    "TEST_MODEL_DIR", "/data/tensordata/Qwen3.5-35B-A3B-BF16"
)
N_EXPERTS_TO_TEST = int(os.environ.get("N_EXPERTS", "8"))
PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def _find_tensor(model_dir: str, key: str) -> torch.Tensor | None:
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def rtn_int4(W: torch.Tensor) -> torch.Tensor:
    """Per-channel symmetric int4 RTN. Same bit budget as XFP-4."""
    W = W.float()
    amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    return (W / scale).round().clamp(-7, 7) * scale


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0), dim=1
    ).item()


def _per_channel_cos(W: torch.Tensor, W_rec: torch.Tensor) -> tuple[float, float]:
    cs = F.cosine_similarity(W.float(), W_rec.float(), dim=1)
    return float(cs.median().item()), float(cs.min().item())


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"model_dir: {MODEL_DIR}")
    print(f"testing {N_EXPERTS_TO_TEST} routed experts × {len(PROJECTIONS)} projections")
    print()

    rows = []  # (expert, proj, xfp_cos, int4_cos, xfp_chmin, int4_chmin, fwd_xfp, fwd_int4)

    # 35B stores routed experts as 3D fused tensors:
    #   experts.gate_up_proj  shape=(E, N_combined, K)
    #   experts.down_proj     shape=(E, K, N_intermediate)
    fused_keys = {
        "gate_up_proj": "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "down_proj":    "model.language_model.layers.0.mlp.experts.down_proj",
    }
    fused_tensors: dict[str, torch.Tensor] = {}
    for proj_name, key in fused_keys.items():
        t = _find_tensor(MODEL_DIR, key)
        if t is not None:
            fused_tensors[proj_name] = t
            print(f"  {proj_name}: {tuple(t.shape)}")
    if not fused_tensors:
        print("ERROR: no fused expert tensors found")
        sys.exit(1)
    print()

    for e in range(N_EXPERTS_TO_TEST):
        for proj_name, fused in fused_tensors.items():
            W = fused[e]  # 2D slice for this expert
            Wf = W.float()
            proj = proj_name

            # XFP-4 per-channel learned codebook
            packed, codebook, o_idx, o_val, stats = xfp_pack(
                Wf, bits=4, also_score_widths=()
            )
            W_rec_xfp = dequant_xfp(packed, codebook, K=Wf.shape[1], bits=4)
            if o_idx is not None and o_idx.numel() > 0:
                flat = W_rec_xfp.reshape(-1).clone()
                flat[o_idx] = o_val.to(flat.dtype)
                W_rec_xfp = flat.reshape(Wf.shape)

            # int4-RTN per-channel symmetric (no learned centroids)
            W_rec_int4 = rtn_int4(Wf)

            xfp_cos = _cos(Wf, W_rec_xfp)
            int4_cos = _cos(Wf, W_rec_int4)
            xfp_med, xfp_min = _per_channel_cos(Wf, W_rec_xfp)
            int4_med, int4_min = _per_channel_cos(Wf, W_rec_int4)

            torch.manual_seed(e)
            x = torch.randn(32, Wf.shape[1], dtype=torch.float32) * 0.5
            y_ref = x @ Wf.T
            y_xfp = x @ W_rec_xfp.float().T
            y_int4 = x @ W_rec_int4.float().T
            fwd_xfp = _cos(y_ref, y_xfp)
            fwd_int4 = _cos(y_ref, y_int4)

            rows.append((e, proj, xfp_cos, int4_cos, xfp_med, int4_med,
                         xfp_min, int4_min, fwd_xfp, fwd_int4,
                         tuple(Wf.shape), 100*stats.outlier_fraction))

    if not rows:
        print("ERROR: no expert weights found")
        sys.exit(1)

    # Per-row report
    print(f"{'expert':>6} {'proj':>10} {'shape':>14} "
          f"{'XFP cos':>9} {'int4 cos':>9} {'Δ':>6} | "
          f"{'XFP/ch_med':>10} {'int4/ch_med':>11} | "
          f"{'XFP fwd':>9} {'int4 fwd':>9} {'Δfwd':>6} | "
          f"{'outlier%':>8}")
    for r in rows:
        e, proj, xc, ic, xm, im, xmi, imi, fx, fi, sh, of = r
        delta = (xc - ic) * 100
        delta_fwd = (fx - fi) * 100
        print(f"{e:>6} {proj:>10} {str(sh):>14} "
              f"{xc:>9.5f} {ic:>9.5f} {delta:>+5.2f} | "
              f"{xm:>10.5f} {im:>11.5f} | "
              f"{fx:>9.5f} {fi:>9.5f} {delta_fwd:>+5.2f} | "
              f"{of:>7.2f}%")

    import statistics
    xfp_avg = statistics.mean(r[2] for r in rows)
    int4_avg = statistics.mean(r[3] for r in rows)
    fwd_xfp_avg = statistics.mean(r[8] for r in rows)
    fwd_int4_avg = statistics.mean(r[9] for r in rows)
    print()
    print(f"=== AGGREGATE ({len(rows)} weight matrices) ===")
    print(f"  W cos:    XFP-4 = {xfp_avg:.5f} | int4-RTN = {int4_avg:.5f} | "
          f"Δ = {(xfp_avg - int4_avg)*100:+.3f}pp")
    print(f"  fwd cos:  XFP-4 = {fwd_xfp_avg:.5f} | int4-RTN = {fwd_int4_avg:.5f} | "
          f"Δ = {(fwd_xfp_avg - fwd_int4_avg)*100:+.3f}pp")
    print()
    print("Interpretation:")
    if (xfp_avg - int4_avg) > 0.005:
        print("  ✓ XFP-4 per-expert clearly beats int4-RTN — codebook learning works.")
    elif (xfp_avg - int4_avg) > -0.005:
        print("  ⚠ XFP-4 ≈ int4-RTN (within ±0.5pp) — codebook learning gives no measurable")
        print("    advantage; suspect bug or codebook init is too constrained.")
    else:
        print("  ✗ XFP-4 is WORSE than int4-RTN — definite bug somewhere.")


if __name__ == "__main__":
    main()
