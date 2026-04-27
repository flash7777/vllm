# SPDX-License-Identifier: Apache-2.0
"""Reproduce 122B attention layer cos=nan with the actual offending shapes.

Run with:
  python3 -m tests.xfp.test_attn_shape_repro
or directly:
  python3 tests/xfp/test_attn_shape_repro.py

Shapes reported as `mse=nan cos=nan` from a 122B XFP TP=2 PACK run:
  [64x3072]   — likely linear-attention dt/A/b projection
  [10240x3072]
  [8704x3072]
"""
from __future__ import annotations

import sys
import torch

from vllm.multiquant.xfp.xfp_pack import xfp_pack


def _stats(name: str, t: torch.Tensor) -> None:
    f = t.float()
    print(
        f"  {name:>14}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"norm={f.norm().item():.4g} "
        f"mean={f.mean().item():.4g} std={f.std().item():.4g} "
        f"abs_max={f.abs().max().item():.4g} "
        f"has_nan={bool(torch.isnan(f).any().item())} "
        f"has_inf={bool(torch.isinf(f).any().item())} "
        f"all_zero={bool((f == 0).all().item())} "
    )


def _try(label: str, W: torch.Tensor, bits: int = 4) -> None:
    print(f"\n=== {label} ===")
    _stats("W", W)
    try:
        packed, codebook, o_idx, o_val, stats = xfp_pack(
            W.float(), bits=bits, also_score_widths=()
        )
    except Exception as e:
        print(f"  xfp_pack RAISED {type(e).__name__}: {e}")
        return
    print(
        f"  pack stats: bits={stats.bits} mse={stats.mse:.3g} "
        f"cos={stats.cos_sim:.4f} outliers={100*stats.outlier_fraction:.3f}% "
        f"3sigma={100*stats.outlier_ratio_k3:.1f}%"
    )
    _stats("codebook", codebook)
    _stats("packed", packed)


def main() -> None:
    torch.manual_seed(0)
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")

    # 1) Healthy baseline — Gaussian random
    _try("baseline: gaussian (128, 256)",
         torch.randn(128, 256) * 0.1)

    # 2) Reproduce reported pathological shapes with random init
    _try("(64, 3072) gaussian", torch.randn(64, 3072) * 0.02)
    _try("(10240, 3072) gaussian", torch.randn(10240, 3072) * 0.02)
    _try("(8704, 3072) gaussian", torch.randn(8704, 3072) * 0.02)

    # 3) Pathological inputs — what GatedDeltaNet weights might look like
    _try("(64, 3072) all zeros", torch.zeros(64, 3072))
    _try("(64, 3072) all ones * 1e-8 (near-zero)",
         torch.full((64, 3072), 1e-8))
    _try("(64, 3072) one zero row", _patch_zero_row(torch.randn(64, 3072) * 0.02, row=7))
    _try("(64, 3072) NaN-init",
         torch.full((64, 3072), float("nan")))
    _try("(64, 3072) constant value",
         torch.full((64, 3072), 0.5))


def _patch_zero_row(W: torch.Tensor, row: int) -> torch.Tensor:
    W = W.clone()
    W[row] = 0.0
    return W


if __name__ == "__main__":
    main()
