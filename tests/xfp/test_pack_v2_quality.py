"""XFP-V2 quality test — Phase 1 acceptance gate.

Compares xfp_pack_v2 (per-group + shared library) against:
  - XFP-V1 per-channel learned codebook (current production)
  - int4-RTN per-channel symmetric (lower quality bound)
  - int4-RTN per-group g=32 (the target to beat)

Pass criteria:
  Phase 1 OK if:
    - V2 cos ≥ V1 cos on every weight class tested
    - V2 cos ≥ int4-g32 cos on average (the harder bar)
    - dequant_xfp_v2 matches what xfp_pack_v2 reconstructs (round-trip)
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.multiquant.xfp.xfp_pack import (
    xfp_pack, xfp_pack_v2, dequant_xfp, dequant_xfp_v2,
)


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
    W = W.float()
    N, K = W.shape
    pad = (group_size - K % group_size) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    K2 = W.shape[1]
    Wg = W.reshape(N, K2 // group_size, group_size)
    amax = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    Wq = (Wg / scale).round().clamp(-7, 7) * scale
    return Wq.reshape(N, K2)[:, :K - pad if pad else K]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0), dim=1).item()


def _v1_recon(W: torch.Tensor) -> torch.Tensor:
    Wf = W.float()
    packed, codebook, o_idx, o_val, _stats = xfp_pack(Wf, bits=4, also_score_widths=())
    W_rec = dequant_xfp(packed, codebook, K=Wf.shape[1], bits=4)
    if o_idx is not None and o_idx.numel() > 0:
        flat = W_rec.reshape(-1).clone()
        flat[o_idx] = o_val.to(flat.dtype)
        W_rec = flat.reshape(Wf.shape)
    return W_rec


def _v2_recon(W: torch.Tensor, group_size: int = 128, library_size: int = 32):
    Wf = W.float()
    packed, library, lib_id, scale, mid, stats = xfp_pack_v2(
        Wf, bits=4, group_size=group_size, library_size=library_size
    )
    # We need UNPACKED idx for the python dequant. Easiest: compute it
    # from the chosen library + per-group params (same as the pack does).
    N, K = Wf.shape
    G = K // group_size
    W_norm = (Wf.reshape(N, G, group_size) - mid.float().unsqueeze(-1)) / scale.float().unsqueeze(-1)
    chosen_lib = library.float()[lib_id.long()]  # [N, G, n_centroids]
    d = (W_norm.unsqueeze(-1) - chosen_lib.unsqueeze(2)).abs()
    idx_full = d.argmin(dim=-1)  # [N, G, group_size]
    idx_full = idx_full.reshape(N, K)
    W_rec = dequant_xfp_v2(library, lib_id, scale, mid, idx_full, group_size)
    return W_rec, stats


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"model: {MODEL_DIR}")

    LAYER0 = "model.language_model.layers.0"
    cases = [
        (f"{LAYER0}.linear_attn.in_proj_qkv.weight",     "ATTN qkv"),
        (f"{LAYER0}.linear_attn.out_proj.weight",        "ATTN out"),
        (f"{LAYER0}.linear_attn.in_proj_a.weight",       "ATTN a"),
        (f"{LAYER0}.linear_attn.in_proj_b.weight",       "ATTN b"),
        (f"{LAYER0}.linear_attn.in_proj_z.weight",       "ATTN z"),
        (f"{LAYER0}.mlp.shared_expert.gate_proj.weight", "SHARED gate"),
        (f"{LAYER0}.mlp.shared_expert.down_proj.weight", "SHARED down"),
    ]

    print(f"\n{'layer':>14} {'shape':>14} "
          f"{'V1 cos':>9} {'V2 cos':>9} {'g32 cos':>9} {'g128 cos':>9} | "
          f"{'V2-V1':>7} {'V2-g32':>7} | {'lib p5':>7}")
    print("-" * 110)

    sums = {"v1": 0, "v2": 0, "g32": 0, "g128": 0}
    n = 0
    for key, label in cases:
        W = _find_tensor(MODEL_DIR, key)
        if W is None:
            print(f"{label:>14}  (not found)")
            continue
        Wf = W.float()
        v1 = _v1_recon(W)
        v2, v2_stats = _v2_recon(W, group_size=128, library_size=32)
        g32 = rtn_int4_pergroup(Wf, 32)
        g128 = rtn_int4_pergroup(Wf, 128)
        c_v1 = _cos(Wf, v1)
        c_v2 = _cos(Wf, v2)
        c_g32 = _cos(Wf, g32)
        c_g128 = _cos(Wf, g128)
        sums["v1"] += c_v1
        sums["v2"] += c_v2
        sums["g32"] += c_g32
        sums["g128"] += c_g128
        n += 1
        print(f"{label:>14} {str(tuple(Wf.shape)):>14} "
              f"{c_v1:>9.5f} {c_v2:>9.5f} {c_g32:>9.5f} {c_g128:>9.5f} | "
              f"{(c_v2-c_v1)*100:>+6.2f}pp {(c_v2-c_g32)*100:>+6.2f}pp | "
              f"{v2_stats.library_p5_cos:>7.4f}")

    # 1 routed expert slice
    fused = _find_tensor(MODEL_DIR, f"{LAYER0}.mlp.experts.gate_up_proj")
    if fused is not None:
        Wf = fused[0].float()
        v1 = _v1_recon(Wf)
        v2, v2_stats = _v2_recon(Wf, group_size=128, library_size=32)
        g32 = rtn_int4_pergroup(Wf, 32)
        g128 = rtn_int4_pergroup(Wf, 128)
        c_v1 = _cos(Wf, v1)
        c_v2 = _cos(Wf, v2)
        c_g32 = _cos(Wf, g32)
        c_g128 = _cos(Wf, g128)
        sums["v1"] += c_v1; sums["v2"] += c_v2; sums["g32"] += c_g32; sums["g128"] += c_g128
        n += 1
        print(f"{'ROUTED e0':>14} {str(tuple(Wf.shape)):>14} "
              f"{c_v1:>9.5f} {c_v2:>9.5f} {c_g32:>9.5f} {c_g128:>9.5f} | "
              f"{(c_v2-c_v1)*100:>+6.2f}pp {(c_v2-c_g32)*100:>+6.2f}pp | "
              f"{v2_stats.library_p5_cos:>7.4f}")

    print("-" * 110)
    print(f"\n=== AGGREGATE ({n} weights) ===")
    print(f"  V1   avg cos: {sums['v1']/n:.5f}")
    print(f"  V2   avg cos: {sums['v2']/n:.5f}    (Δ V1: {(sums['v2']-sums['v1'])/n*100:+.3f}pp)")
    print(f"  g32  avg cos: {sums['g32']/n:.5f}    (Δ V1: {(sums['g32']-sums['v1'])/n*100:+.3f}pp)")
    print(f"  g128 avg cos: {sums['g128']/n:.5f}")
    print()
    if sums['v2'] >= sums['v1'] and sums['v2'] >= sums['g32']:
        print("  ✓ V2 ≥ V1 AND V2 ≥ int4-g32 — Phase 1 PASS")
    elif sums['v2'] >= sums['v1']:
        print("  ⚠ V2 ≥ V1 but < int4-g32 — investigate group_size / library_size")
    else:
        print("  ✗ V2 < V1 — bug somewhere, investigate")


if __name__ == "__main__":
    main()
