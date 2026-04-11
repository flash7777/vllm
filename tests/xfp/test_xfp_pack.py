# SPDX-License-Identifier: Apache-2.0
"""Unit tests for xfp_pack encoder (CPU, no GPU needed)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm.multiquant.xfp.xfp_pack import xfp_pack, dequant_xfp


def _reconstruct(packed, codebook, K, bits):
    return dequant_xfp(packed, codebook, K=K, bits=bits).to(torch.float32)


@pytest.mark.parametrize(
    "bits, expected_cos_lower",
    [(2, 0.90), (3, 0.97), (4, 0.99)],
)
def test_roundtrip_gaussian(bits: int, expected_cos_lower: float) -> None:
    torch.manual_seed(0)
    W = torch.randn(128, 256, dtype=torch.float32) * 0.1

    packed, codebook, stats = xfp_pack(W, bits=bits)

    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    k_packed = (256 + vals_per_word - 1) // vals_per_word
    assert packed.shape == (k_packed, 128)
    assert codebook.shape == (128, 1 << bits)
    assert packed.dtype == torch.int32

    W_rec = _reconstruct(packed, codebook, K=256, bits=bits)
    cos = F.cosine_similarity(
        W.flatten().unsqueeze(0), W_rec.flatten().unsqueeze(0), dim=1
    ).item()
    assert cos >= expected_cos_lower, (
        f"xfp{bits} cos {cos:.4f} < expected {expected_cos_lower}"
    )

    assert stats.bits == bits
    assert stats.shape == (128, 256)
    assert stats.mse > 0
    assert stats.cos_sim == pytest.approx(cos, abs=1e-3)


def test_roundtrip_heavy_tail() -> None:
    torch.manual_seed(1)
    W = torch.randn(64, 128, dtype=torch.float32) * 0.05
    mask = torch.rand_like(W) < 0.03
    W = torch.where(mask, W * 20.0, W)

    packed, codebook, stats = xfp_pack(W, bits=4)
    assert torch.isfinite(codebook).all()

    W_rec = _reconstruct(packed, codebook, K=128, bits=4)
    assert torch.isfinite(W_rec).all()
    assert stats.outlier_ratio_k3 > 0.005


def test_index_bounds() -> None:
    torch.manual_seed(2)
    W = torch.randn(32, 64, dtype=torch.float32)

    for bits in (2, 3, 4):
        packed, _, _ = xfp_pack(W, bits=bits)
        vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
        mask = (1 << bits) - 1

        packed_nk = packed.t().to(torch.int64)
        for slot in range(vals_per_word):
            extracted = (packed_nk >> (slot * bits)) & mask
            assert int(extracted.min().item()) >= 0
            assert int(extracted.max().item()) < (1 << bits)


def test_candidate_scoring() -> None:
    torch.manual_seed(3)
    W = torch.randn(64, 128, dtype=torch.float32) * 0.2

    _, _, stats = xfp_pack(W, bits=4, also_score_widths=(2, 3, 4))
    assert set(stats.mse_per_bits.keys()) == {2, 3, 4}
    assert stats.mse_per_bits[2] > stats.mse_per_bits[3]
    assert stats.mse_per_bits[3] > stats.mse_per_bits[4]
    assert stats.recommended_bits == 4
    assert stats.recommended_gap == pytest.approx(1.0, abs=1e-6)


def test_invalid_bits() -> None:
    W = torch.randn(8, 16)
    with pytest.raises(ValueError, match="bits"):
        xfp_pack(W, bits=5)
    with pytest.raises(ValueError, match="bits"):
        xfp_pack(W, bits=1)


def test_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        xfp_pack(torch.randn(3, 4, 5), bits=4)
