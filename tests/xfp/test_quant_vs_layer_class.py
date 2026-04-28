"""Compare XFP-4 vs int4-{per-channel, per-group} on different layer classes.

Hypothesis layers:
  - linear_attn projections (in_proj_qkv, out_proj, in_proj_a/b, in_proj_z)
    → potentially heavy-tail with concentrated outliers
  - shared_expert (gate/up/down)
    → typical FFN distribution
  - routed_expert slices (per-expert from fused 3D)
    → MoE expert distribution (we already tested these)

For each weight matrix, compute reconstruction cos for:
  1. XFP-4 (per-channel learned codebook + 4σ outlier extraction)
  2. int4 per-channel symmetric RTN  (one scale per row)
  3. int4 per-group symmetric RTN, group_size=128  (16 scales per row of 2048)
  4. int4 per-group symmetric RTN, group_size=32   (64 scales per row of 2048)

If XFP matches/beats per-group int4 → codebook works.
If XFP loses to per-group int4 → per-group wins via more scale resolution.
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


def _find_tensor(model_dir: str, key: str) -> torch.Tensor | None:
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def rtn_int4_perchannel(W: torch.Tensor) -> torch.Tensor:
    W = W.float()
    amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    return (W / scale).round().clamp(-7, 7) * scale


def rtn_int4_pergroup(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group symmetric int4 RTN — like AutoRound iter=0."""
    W = W.float()
    N, K = W.shape
    pad = (group_size - K % group_size) % group_size
    if pad:
        Wp = torch.nn.functional.pad(W, (0, pad))
    else:
        Wp = W
    K2 = Wp.shape[1]
    Wg = Wp.reshape(N, K2 // group_size, group_size)
    amax = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    Wq = (Wg / scale).round().clamp(-7, 7) * scale
    Wq = Wq.reshape(N, K2)[:, :K]
    return Wq


def xfp4_recon(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    Wf = W.float()
    packed, codebook, o_idx, o_val, stats = xfp_pack(
        Wf, bits=4, also_score_widths=()
    )
    W_rec = dequant_xfp(packed, codebook, K=Wf.shape[1], bits=4)
    if o_idx is not None and o_idx.numel() > 0:
        flat = W_rec.reshape(-1).clone()
        flat[o_idx] = o_val.to(flat.dtype)
        W_rec = flat.reshape(Wf.shape)
    return W_rec, 100 * stats.outlier_fraction


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0), dim=1).item()


def _row_cos_pct(W: torch.Tensor, W_rec: torch.Tensor, q: float) -> float:
    """Per-row cos similarity, qth percentile."""
    cs = F.cosine_similarity(W.float(), W_rec.float(), dim=1)
    return float(cs.quantile(q).item())


def test_one(W: torch.Tensor, label: str) -> dict:
    Wf = W.float()
    print(f"\n{label}: shape={tuple(W.shape)} norm={Wf.norm().item():.3g} "
          f"std={Wf.std().item():.4g} abs_max={Wf.abs().max().item():.4g}")

    methods = {
        "XFP-4": lambda: xfp4_recon(W)[0],
        "int4 per-ch": lambda: rtn_int4_perchannel(Wf),
        "int4 g=128": lambda: rtn_int4_pergroup(Wf, 128),
        "int4 g=32": lambda: rtn_int4_pergroup(Wf, 32),
    }
    results = {}
    for name, fn in methods.items():
        Wr = fn()
        cos_g = _cos(Wf, Wr)
        cos_min = _row_cos_pct(Wf, Wr, 0.05)  # 5th-percentile worst row
        results[name] = (cos_g, cos_min)
        print(f"  {name:>12}: cos={cos_g:.5f} (worst-5%-row cos={cos_min:.5f})")
    return results


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"model: {MODEL_DIR}")

    # Attention/linear-attn projections (often heavy-tail)
    LAYER0 = "model.language_model.layers.0"
    cases = [
        (f"{LAYER0}.linear_attn.in_proj_qkv.weight",     "ATTN in_proj_qkv"),
        (f"{LAYER0}.linear_attn.out_proj.weight",        "ATTN out_proj"),
        (f"{LAYER0}.linear_attn.in_proj_a.weight",       "ATTN in_proj_a"),
        (f"{LAYER0}.linear_attn.in_proj_b.weight",       "ATTN in_proj_b"),
        (f"{LAYER0}.linear_attn.in_proj_z.weight",       "ATTN in_proj_z"),
        (f"{LAYER0}.mlp.shared_expert.gate_proj.weight", "SHARED gate_proj"),
        (f"{LAYER0}.mlp.shared_expert.down_proj.weight", "SHARED down_proj"),
    ]
    for key, label in cases:
        W = _find_tensor(MODEL_DIR, key)
        if W is not None:
            test_one(W, label)
        else:
            print(f"\n{label}: NOT FOUND ({key})")

    # 1 routed-expert slice (we already saw this is uniform-ish)
    fused = _find_tensor(MODEL_DIR, f"{LAYER0}.mlp.experts.gate_up_proj")
    if fused is not None:
        test_one(fused[0], "ROUTED gate_up_proj (expert 0)")


if __name__ == "__main__":
    main()
