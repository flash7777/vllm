# SPDX-License-Identifier: Apache-2.0
"""XFP pack utility — Lloyd codebook + word-aligned sub-byte packing.

One call per weight matrix at load time:
    packed, codebook, stats = xfp_pack(W, bits, also_score_widths=(2,3,4))

Per output channel (row of W), a 2^bits-entry codebook is fit via Lloyd
iteration on the 1-D weight distribution. Each weight is then replaced by the
index of its nearest codebook entry. Indices are bit-packed word-aligned into
uint32 words so the decode kernel can extract them with shift+mask and no
cross-word reads.

Packing per bits (all word-aligned on uint32):

    bits | values per uint32 | K_packed       | reserve bits per word
    -----+-------------------+----------------+----------------------
    2    | 16                | ceil(K / 16)   | 0
    3    | 10                | ceil(K / 10)   | 2
    4    |  8                | ceil(K / 8)    | 0

Output shape: packed [K_packed, N_out] uint32 (K-major, matches the layout
convention of mq_gemm_int2/int3 kernels).

Statistics:
    Per-layer XFPPackStats captures weight distribution moments, outlier
    ratios at k=3σ and k=4σ, reconstruction MSE/cos_sim at the chosen bits,
    and (when also_score_widths is set) MSE at alternative bit widths so
    callers can recommend a per-layer optimal bit width for future auto-sizing.

v1 scope: no outlier extraction (bulk-only codebook). Outlier-split path is v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class XFPPackStats:
    """Per-layer distribution + reconstruction stats from Lloyd packing."""

    bits: int
    shape: tuple  # (N_out, K)

    # Weight distribution moments (raw input)
    w_mean: float
    w_std: float
    w_abs_max: float

    # Outlier fraction estimates (|w - mean| > k * std)
    outlier_ratio_k3: float
    outlier_ratio_k4: float

    # Reconstruction quality at the chosen `bits`
    mse: float
    rmse_rel: float  # sqrt(mse) / w_std
    max_abs_err: float
    cos_sim: float

    # Candidate scoring — MSE at other bit widths (empty unless requested)
    mse_per_bits: dict[int, float] = field(default_factory=dict)

    # Auto-size recommendation derived from mse_per_bits
    # (falls back to `bits` when mse_per_bits is empty)
    recommended_bits: int = 0
    recommended_gap: float = 1.0  # mse[chosen] / mse[recommended], ≥1.0


# ─── Lloyd iteration ──────────────────────────────────────────────────


def _lloyd_per_channel(
    W: torch.Tensor,  # [N_out, K] fp32
    n_centroids: int,
    n_iters: int,
    row_chunk: int = 4096,
) -> torch.Tensor:
    """Return codebook [N_out, n_centroids] fp32.

    Lloyd's algorithm (1-D k-means) per row of W. Min-max linspace init,
    then alternating assignment/update. Large row_chunk (default 4096) so
    typical sub-64k-row matrices run in a single chunk — the outer Python
    loop then has one iteration per Lloyd pass, not dozens.
    """
    device = W.device
    N_out, K = W.shape

    # Min-max linspace initialization per row — O(N_out * K), avoids the
    # O(N_out * K * log K) cost of torch.quantile. On a smooth distribution
    # Lloyd converges from this within ~15 iterations; on heavy-tailed
    # distributions the first few iterations already pull centroids toward
    # the CDF-uniform optimum.
    w_min = W.min(dim=1, keepdim=True).values  # [N_out, 1]
    w_max = W.max(dim=1, keepdim=True).values  # [N_out, 1]
    t = torch.linspace(
        0.0, 1.0, n_centroids, device=device, dtype=torch.float32
    )  # [n_centroids]
    codebook = w_min + (w_max - w_min) * t.unsqueeze(0)  # [N_out, n_centroids]

    # Break ties so identical rows don't collapse all centroids
    jitter = torch.linspace(
        -1e-6, 1e-6, n_centroids, device=device, dtype=torch.float32,
    )
    codebook = codebook + jitter.unsqueeze(0)

    for _ in range(n_iters):
        new_codebook = torch.empty_like(codebook)
        for r0 in range(0, N_out, row_chunk):
            r1 = min(r0 + row_chunk, N_out)
            W_chunk = W[r0:r1]  # [C, K]
            cb_chunk = codebook[r0:r1]  # [C, n_centroids]

            # Nearest assignment — distance in [C, K, n_centroids]
            dist = (W_chunk.unsqueeze(-1) - cb_chunk.unsqueeze(1)).abs()
            idx = dist.argmin(-1)  # [C, K]

            # Mean-of-cluster update
            # one-hot scatter-add accumulation
            C = r1 - r0
            sums = torch.zeros(
                C, n_centroids, dtype=torch.float32, device=device
            )
            counts = torch.zeros(
                C, n_centroids, dtype=torch.float32, device=device
            )
            sums.scatter_add_(1, idx, W_chunk)
            counts.scatter_add_(1, idx, torch.ones_like(W_chunk))

            # Empty clusters → keep old centroid (count=1 with old centroid value)
            empty = counts == 0
            if empty.any():
                counts = counts + empty.float()
                sums = sums + torch.where(
                    empty, cb_chunk, torch.zeros_like(cb_chunk)
                )

            new_codebook[r0:r1] = sums / counts

        # Re-sort each row's codebook so indices are monotone
        # (helps logging and some decode optimizations)
        codebook = torch.sort(new_codebook, dim=1).values

    return codebook


def _assign_indices(
    W: torch.Tensor,  # [N_out, K] fp32
    codebook: torch.Tensor,  # [N_out, 2^bits] fp32
    row_chunk: int = 128,
) -> torch.Tensor:
    """Return argmin indices [N_out, K] int64."""
    N_out, K = W.shape
    idx = torch.empty(N_out, K, dtype=torch.int64, device=W.device)
    for r0 in range(0, N_out, row_chunk):
        r1 = min(r0 + row_chunk, N_out)
        dist = (W[r0:r1].unsqueeze(-1) - codebook[r0:r1].unsqueeze(1)).abs()
        idx[r0:r1] = dist.argmin(-1)
    return idx


# ─── Word-aligned sub-byte packing ────────────────────────────────────


def _pack_indices(
    idx: torch.Tensor,  # [N_out, K] int, values in [0, 2^bits)
    bits: int,
) -> torch.Tensor:
    """Pack into [K_packed, N_out] uint32.

    Word layout (bits → values/word):
        bits=2: 16 values, shift 2*i, no reserve
        bits=3: 10 values, shift 3*i, 2 reserve bits (bit 30-31 unused)
        bits=4:  8 values, shift 4*i, no reserve
    """
    N_out, K = idx.shape
    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    K_packed = (K + vals_per_word - 1) // vals_per_word

    # Pad K so it's a multiple of vals_per_word
    if K < K_packed * vals_per_word:
        pad = K_packed * vals_per_word - K
        idx = F.pad(idx, (0, pad), value=0)

    # idx is [N_out, K_packed * vals_per_word]; view as [N_out, K_packed, vals_per_word]
    idx = idx.view(N_out, K_packed, vals_per_word).to(torch.int64)

    # Build per-word packed value via shifted OR
    packed = torch.zeros(
        N_out, K_packed, dtype=torch.int64, device=idx.device
    )
    for slot in range(vals_per_word):
        packed |= (idx[:, :, slot] & ((1 << bits) - 1)) << (slot * bits)

    # Transpose to K-major layout [K_packed, N_out] and cast to int32
    # (int32 is the closest "uint32" torch offers; bit pattern is identical
    # as long as no arithmetic interprets it as signed)
    return packed.t().contiguous().to(torch.int32)


# ─── Reference dequant (used by tests and by the Python reconstruction
#     path that feeds stats.mse / stats.cos_sim) ──────────────────────


def dequant_xfp(
    packed: torch.Tensor,  # [K_packed, N_out] int32
    codebook: torch.Tensor,  # [N_out, 2^bits] any float dtype
    K: int,
    bits: int,
) -> torch.Tensor:
    """Reconstruct W [N_out, K] from packed + codebook.

    Reference implementation — torch ops only, no CUDA kernel. Used by
    xfp_pack() itself (to compute MSE) and by tests.
    """
    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    mask = (1 << bits) - 1
    K_packed = packed.shape[0]
    N_out = packed.shape[1]
    assert K_packed * vals_per_word >= K

    packed_nk = packed.t().to(torch.int64)  # [N_out, K_packed]

    # Unpack to [N_out, K_padded]
    unpacked = torch.zeros(
        N_out, K_packed * vals_per_word,
        dtype=torch.int64, device=packed.device,
    )
    for slot in range(vals_per_word):
        unpacked[:, slot::vals_per_word] = (packed_nk >> (slot * bits)) & mask

    idx = unpacked[:, :K]  # drop padding

    # Per-row gather: codebook is [N_out, 2^bits], index per (n, k)
    return torch.gather(codebook, 1, idx)


# ─── Stats ──────────────────────────────────────────────────────────


def _distribution_stats(W: torch.Tensor) -> tuple[float, float, float, float, float]:
    """Return (mean, std, abs_max, outlier_ratio_k3, outlier_ratio_k4)."""
    mean = W.mean().item()
    std = W.std().item()
    abs_max = W.abs().max().item()
    centered = (W - mean).abs()
    total = float(W.numel())
    outlier_k3 = float((centered > 3.0 * std).sum().item()) / total
    outlier_k4 = float((centered > 4.0 * std).sum().item()) / total
    return mean, std, abs_max, outlier_k3, outlier_k4


def _reconstruction_stats(
    W: torch.Tensor, W_rec: torch.Tensor
) -> tuple[float, float, float]:
    """Return (mse, max_abs_err, cos_sim)."""
    diff = W - W_rec
    mse = (diff * diff).mean().item()
    max_abs_err = diff.abs().max().item()
    cos = F.cosine_similarity(
        W.reshape(-1).unsqueeze(0), W_rec.reshape(-1).unsqueeze(0), dim=1
    ).item()
    return mse, max_abs_err, cos


def _score_mse_only(W: torch.Tensor, bits: int, lloyd_iters: int) -> float:
    """Fit a codebook at `bits` and return reconstruction MSE (no packing).

    Used to populate stats.mse_per_bits for the auto-size signal.
    """
    codebook = _lloyd_per_channel(W, 1 << bits, lloyd_iters)
    idx = _assign_indices(W, codebook)
    W_rec = torch.gather(codebook, 1, idx)
    return ((W - W_rec) * (W - W_rec)).mean().item()


# ─── Public entry point ────────────────────────────────────────────


def xfp_pack(
    W: torch.Tensor,
    bits: int,
    lloyd_iters: int = 20,
    also_score_widths: tuple[int, ...] = (),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, XFPPackStats]:
    """Pack a weight matrix via per-channel Lloyd codebook + bit-packed indices.

    Args:
        W: [N_out, K] weight tensor. Caller converts from bf16/fp16 to fp32.
        bits: target bit width; one of {2, 3, 4}. Determines codebook size
            (2^bits entries) and pack layout per §3.3 of XFP.PAPER.md.
        lloyd_iters: number of Lloyd refinement iterations (20–50 typical).
        also_score_widths: additional bit widths to fit a codebook for and
            score only for MSE, without packing. Used to populate
            stats.mse_per_bits as a per-layer auto-size signal.
        seed: random seed reserved for v2 outlier extraction. Ignored in v1.

    Returns:
        packed: [K_packed, N_out] int32 (bit-identical to uint32)
        codebook: [N_out, 2^bits] fp16
        stats: XFPPackStats
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"xfp_pack: unsupported bits={bits}, must be in {{2,3,4}}")
    if W.dim() != 2:
        raise ValueError(f"xfp_pack: W must be 2D [N_out, K], got shape {tuple(W.shape)}")

    del seed  # reserved for v2

    W = W.to(torch.float32)
    N_out, K = W.shape
    n_centroids = 1 << bits

    # 1. Lloyd codebook
    codebook_fp32 = _lloyd_per_channel(W, n_centroids, lloyd_iters)

    # 2. Assign indices
    idx = _assign_indices(W, codebook_fp32)

    # 3. Reconstruction (for MSE / cos sim)
    W_rec = torch.gather(codebook_fp32, 1, idx)

    # 4. Pack indices to word-aligned uint32
    packed = _pack_indices(idx, bits)

    # 5. Stats
    w_mean, w_std, w_abs_max, k3, k4 = _distribution_stats(W)
    mse, max_abs_err, cos_sim = _reconstruction_stats(W, W_rec)
    rmse_rel = (mse ** 0.5) / max(w_std, 1e-12)

    mse_per_bits: dict[int, float] = {bits: mse}
    for b in also_score_widths:
        if b == bits or b in mse_per_bits:
            continue
        if b not in (2, 3, 4):
            continue
        mse_per_bits[b] = _score_mse_only(W, b, lloyd_iters)

    # Recommendation: lowest MSE among scored widths, ties → prefer lower bits
    if mse_per_bits:
        recommended_bits = min(
            mse_per_bits.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        recommended_gap = mse_per_bits[bits] / max(
            mse_per_bits[recommended_bits], 1e-30
        )
    else:
        recommended_bits = bits
        recommended_gap = 1.0

    stats = XFPPackStats(
        bits=bits,
        shape=(N_out, K),
        w_mean=w_mean,
        w_std=w_std,
        w_abs_max=w_abs_max,
        outlier_ratio_k3=k3,
        outlier_ratio_k4=k4,
        mse=mse,
        rmse_rel=rmse_rel,
        max_abs_err=max_abs_err,
        cos_sim=cos_sim,
        mse_per_bits=mse_per_bits,
        recommended_bits=recommended_bits,
        recommended_gap=recommended_gap,
    )

    return packed, codebook_fp32.to(torch.float16), stats
