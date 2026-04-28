"""End-to-end XFP roundtrip test on real model weights.

Goal: pin-point where the −23pp GSM8K quality loss comes from.
Compare XFP-4 vs int4-RTN on REAL Qwen3.5-35B weights, layer by layer.

Stages tested (each layer):
  1. xfp_pack(W) → (packed, codebook, outliers)         <-- pack stage
  2. dequant_xfp(packed, codebook, ...) + apply outliers → W_rec    <-- decode stage
  3. y_ref  = x @ W.T   (BF16 ground truth)
  4. y_xfp  = x @ W_rec.T   (post-dequant)
  5. y_int4 = x @ rtn_int4(W).T   (RTN baseline at same bits)

Reports per layer:
  - W reconstruction: ||W - W_rec|| / ||W||,  cos_sim
  - y output:         ||y_ref - y_xfp|| / ||y_ref||,  cos_sim
  - vs int4 baseline: same metrics for sanity

If XFP's W_rec is close to W but y_xfp is far from y_ref → forward kernel bug.
If W_rec is far from W → pack/decode bug.
If XFP is significantly worse than int4-RTN at same bits → codebook learning has a flaw.

Usage (in container):
  python3 tests/xfp/test_real_weight_roundtrip.py
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
# Pick a few representative layer types: a stacked attention proj, an
# out_proj, a MoE expert weight (shared_expert is small so OK in CPU).
TARGET_KEYS = [
    "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
    "model.language_model.layers.0.linear_attn.in_proj_ba.weight",
    "model.language_model.layers.0.mlp.shared_expert.gate_up_proj.weight",
    "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
]


def _find_tensor(model_dir: str, key: str) -> torch.Tensor | None:
    """Scan safetensors shards for the given key."""
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def rtn_int4(W: torch.Tensor) -> torch.Tensor:
    """Simple per-channel symmetric int4 RTN (round-to-nearest).

    Baseline at the same 4-bit budget as XFP-4 (no codebook learning,
    no outlier extraction). Should be a *lower* quality bound — XFP
    with learned codebooks AND outliers must beat this, otherwise the
    XFP machinery is broken or buggy.
    """
    W = W.float()
    amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0  # int4 range [-7, 7]
    q = (W / scale).round().clamp(-7, 7)
    return q * scale


def _stats(name: str, ref: torch.Tensor, test: torch.Tensor) -> None:
    diff = (ref - test).float()
    ref_norm = ref.float().norm().item()
    rel_l2 = diff.norm().item() / max(ref_norm, 1e-12)
    cos = F.cosine_similarity(
        ref.float().reshape(-1).unsqueeze(0),
        test.float().reshape(-1).unsqueeze(0),
        dim=1,
    ).item()
    max_err = diff.abs().max().item()
    ref_max = ref.float().abs().max().item()
    print(f"    {name:>10}: rel_L2={rel_l2:.4g}  cos={cos:.6f}  "
          f"max_err={max_err:.4g}  (ref_max={ref_max:.4g})")


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"model_dir: {MODEL_DIR}")

    for key in TARGET_KEYS:
        print(f"\n=== {key} ===")
        W = _find_tensor(MODEL_DIR, key)
        if W is None:
            print("  (not found)")
            continue
        if W.dim() == 3:
            # MoE: pick first expert
            print(f"  3D tensor {tuple(W.shape)} → using expert 0")
            W = W[0]
        if W.dim() != 2:
            print(f"  skip non-2D shape {tuple(W.shape)}")
            continue
        print(f"  shape={tuple(W.shape)} dtype={W.dtype} "
              f"norm={W.float().norm().item():.4g} "
              f"abs_max={W.float().abs().max().item():.4g}")

        Wf = W.float()

        # XFP-4 pack/unpack
        packed, codebook, o_idx, o_val, stats = xfp_pack(
            Wf, bits=4, also_score_widths=()
        )
        # dequant_xfp returns the DENSE reconstruction of the bulk.
        W_rec_xfp_bulk = dequant_xfp(
            packed, codebook, K=Wf.shape[1], bits=4
        )
        # Apply outliers (replace bulk values at outlier positions).
        W_rec_xfp = W_rec_xfp_bulk.clone()
        if o_idx is not None and o_idx.numel() > 0:
            flat = W_rec_xfp.reshape(-1)
            flat[o_idx] = o_val.to(flat.dtype)
            W_rec_xfp = flat.reshape(Wf.shape)
        print(f"  pack stats: cos_sim={stats.cos_sim:.6f} "
              f"mse={stats.mse:.4g} outlier_frac={100*stats.outlier_fraction:.3f}%")

        # int4-RTN baseline (same 4-bit budget, no learning)
        W_rec_int4 = rtn_int4(Wf)

        # Stage 1: weight reconstruction
        print("  --- W reconstruction ---")
        _stats("XFP-4", Wf, W_rec_xfp.float())
        _stats("int4-RTN", Wf, W_rec_int4.float())

        # Stage 2: forward pass with random inputs
        torch.manual_seed(0)
        x = torch.randn(64, Wf.shape[1], dtype=torch.float32) * 0.5
        y_ref = x @ Wf.T
        y_xfp = x @ W_rec_xfp.float().T
        y_int4 = x @ W_rec_int4.float().T
        print("  --- forward output (y = x @ W.T) ---")
        _stats("XFP-4", y_ref, y_xfp)
        _stats("int4-RTN", y_ref, y_int4)


if __name__ == "__main__":
    main()
