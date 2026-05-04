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

    # Outlier split (0 when outlier extraction is disabled)
    outlier_count: int = 0          # number of (row, col) pairs extracted
    outlier_sigma: float = 0.0      # threshold used (0 = none)
    outlier_fraction: float = 0.0   # count / numel

    # Candidate scoring — MSE at other bit widths (empty unless requested)
    mse_per_bits: dict[int, float] = field(default_factory=dict)

    # Auto-size recommendation derived from mse_per_bits
    # (falls back to `bits` when mse_per_bits is empty)
    recommended_bits: int = 0
    recommended_gap: float = 1.0  # mse[chosen] / mse[recommended], ≥1.0

    # Per-channel cosine-similarity histogram: 20 bins in [0.0, 1.0].
    # Captures the full quality landscape of the layer in 20 ints without
    # saving all per-channel cos values. See _compute_cos_hist().
    cos_hist: tuple = field(default_factory=tuple)  # length 20

    # Outlier magnitude histogram: bins for (|w - μ|/σ) ∈
    # [4σ, 5σ, 6σ, 7σ, 8σ+]. Always length 5; zeros when no outliers.
    outlier_hist: tuple = field(default_factory=tuple)  # length 5

    # Candidate-gate survival: list aligned to candidates (2, 3, 4) —
    # True if that bit width met the cos-gate. Picks first True;
    # `recommended_bits` above is the chosen one (same info cross-check).
    bits_survived_gate: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Flatten to JSON-safe dict for cache manifest storage.

        Floats rounded to 6 sig figs, dicts/tuples serialised natively.
        Stays small: ~50 keys × ~10 bytes = <1 KB per layer.
        """
        def r(x):
            return round(float(x), 6) if isinstance(x, (int, float)) else x
        d = {
            "bits": int(self.bits),
            "shape": list(self.shape),
            "w_mean": r(self.w_mean),
            "w_std": r(self.w_std),
            "w_abs_max": r(self.w_abs_max),
            "outlier_ratio_k3": r(self.outlier_ratio_k3),
            "outlier_ratio_k4": r(self.outlier_ratio_k4),
            "mse": r(self.mse),
            "rmse_rel": r(self.rmse_rel),
            "max_abs_err": r(self.max_abs_err),
            "cos_sim": r(self.cos_sim),
            "outlier_count": int(self.outlier_count),
            "outlier_sigma": r(self.outlier_sigma),
            "outlier_fraction": r(self.outlier_fraction),
            "mse_per_bits": {str(k): r(v)
                             for k, v in self.mse_per_bits.items()},
            "recommended_bits": int(self.recommended_bits),
            "recommended_gap": r(self.recommended_gap),
            "cos_hist": [int(x) for x in self.cos_hist],
            "outlier_hist": [int(x) for x in self.outlier_hist],
            "bits_survived_gate": list(self.bits_survived_gate),
        }
        return d


def _compute_cos_hist(W: torch.Tensor, rec: torch.Tensor,
                     n_bins: int = 20) -> tuple:
    """Per-channel cos-sim distribution over [0, 1] in n_bins (default 20).

    Returns tuple of length n_bins with channel counts per bin. W and rec
    are [N_out, K]. Robust to all-zero channels (cos=0 put in first bin).
    """
    import torch.nn.functional as F
    if W.dim() != 2 or rec.dim() != 2:
        return tuple([0] * n_bins)
    cos = F.cosine_similarity(W.float(), rec.float(), dim=1)  # [N_out]
    # Replace NaN (zero-norm rows) with 0.0
    cos = torch.nan_to_num(cos, nan=0.0)
    # Clamp into [0, 1] (cos can be negative but we care about quality)
    cos = cos.clamp(0.0, 1.0)
    # Bin edges: [0, 1/n, 2/n, ..., 1]; last bin inclusive.
    bin_idx = (cos * n_bins).floor().to(torch.int64)
    bin_idx = bin_idx.clamp(0, n_bins - 1)
    counts = torch.bincount(bin_idx, minlength=n_bins)[:n_bins]
    return tuple(int(x) for x in counts.cpu().tolist())


def _compute_outlier_hist(W: torch.Tensor, mu: float, sigma: float) -> tuple:
    """Outlier magnitude histogram: bins for |w - μ|/σ ∈ {4,5,6,7,8+}.

    Returns tuple of length 5 with absolute counts per sigma-band. mu/sigma
    from W's own distribution. Sigma=0 returns zeros (no outliers defined).
    """
    if sigma == 0:
        return (0, 0, 0, 0, 0)
    dev = (W.float() - mu).abs() / max(sigma, 1e-12)
    # Bands: [4,5), [5,6), [6,7), [7,8), [8,∞)
    b4 = int(((dev >= 4) & (dev < 5)).sum().item())
    b5 = int(((dev >= 5) & (dev < 6)).sum().item())
    b6 = int(((dev >= 6) & (dev < 7)).sum().item())
    b7 = int(((dev >= 7) & (dev < 8)).sum().item())
    b8 = int((dev >= 8).sum().item())
    return (b4, b5, b6, b7, b8)


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


# ─── Weight repack for coalesced warp reads ────────────────────────


def xfp_repack(packed: torch.Tensor, warp_size: int = 32) -> torch.Tensor:
    """Repack [K_packed, N] → [K_groups * N * warp_size] int32.

    v4opt kernel's access pattern: warp n, lane i reads
    B_packed[kw * N + n] where kw = lane, lane+32, lane+64...

    Without repack: consecutive lane reads (lane 0..31) at the same kw
    are at addresses kw*N+n — all in one cache line IF different warps
    happen to sync. In practice warps drift → poor L2 utilization.

    After repack: the K dimension is interleaved over warp_size so that
    one warp's consecutive lane reads at the same kw_group form a
    contiguous 128-byte block:

      repacked[kw_group * N * WS + n * WS + lane]

    All 32 lane reads are consecutive → 1 cache line → 100% utilization.

    The tensor is returned as a flat [K_groups * N * warp_size] int32
    to avoid 3D stride complications in the kernel. The kernel computes:
      idx = kw_group * (N * WS) + n * WS + lane
    where kw_group = (kw_original / WS), and lane = kw_original % WS.
    """
    K_packed, N = packed.shape
    K_groups = (K_packed + warp_size - 1) // warp_size

    # Pad K_packed to a multiple of warp_size
    if K_packed % warp_size != 0:
        pad = warp_size - K_packed % warp_size
        packed = F.pad(packed, (0, 0, 0, pad), value=0)

    # [K_groups, warp_size, N] → [K_groups, N, warp_size] → flatten
    repacked = (
        packed.reshape(K_groups, warp_size, N)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(-1)
    )
    return repacked


# ─── Auto bit-width selection ──────────────────────────────────────


def xfp_auto_select(
    W: torch.Tensor,
    candidates: tuple[int, ...] = (2, 3, 4),
    min_cos: float = 0.98,
    lloyd_iters: int = 20,
    outlier_sigma: Optional[float] = 4.0,
    outlier_max_fraction: float = 0.02,
) -> int:
    """Pick the lowest bit width where reconstruction meets the cos gate.

    Runs Lloyd at each candidate width, computes per-channel cosine
    similarity, and returns the first (lowest) bits where the median
    per-channel cos >= min_cos.

    The cos gate is the sole discriminator. MSE ratio was tested and
    rejected: on real MoE models, XFP2 has ~12× higher MSE than XFP4
    but identical math accuracy (the error is spread uniformly and
    doesn't concentrate in model-critical channels). Cos similarity
    captures this: it measures directional preservation per channel,
    which is what matters for downstream quality.

    Falls back to max(candidates) if no lower width qualifies.

    Returns:
        Chosen bits (one of the candidates).
    """
    if W.dim() != 2:
        raise ValueError(f"xfp_auto_select: W must be 2D, got {W.dim()}D")

    W = W.to(torch.float32)
    candidates = tuple(sorted(candidates))
    best_bits = candidates[-1]  # fallback

    # Outlier split (shared across all candidates — same mask)
    if outlier_sigma is not None and outlier_sigma > 0:
        mu = W.mean()
        sigma = W.std()
        threshold = float(outlier_sigma) * sigma
        mask = (W - mu).abs() > threshold
        total = W.numel()
        max_allowed = int(outlier_max_fraction * total)
        nnz = int(mask.sum().item())
        if nnz > max_allowed and max_allowed > 0:
            flat_abs = (W - mu).abs().reshape(-1)
            _, top_idx = torch.topk(flat_abs, max_allowed, largest=True, sorted=False)
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[top_idx] = True
            mask = mask.reshape_as(W)
        if nnz > 0:
            W_bulk = W.clone()
            W_bulk[mask] = mu
        else:
            W_bulk = W
            mask = None
    else:
        W_bulk = W
        mask = None

    # Test each candidate from lowest to highest.
    # Intermediate tensors (idx, rec) are huge for large MoE stacks
    # (e.g. 256-expert Qwen 122B → idx ≈ 12 GB int64, rec ≈ 6 GB float).
    # Explicit del + empty_cache between candidates prevents UMA blowup.
    result_bits = best_bits
    for bits in candidates:
        if bits == best_bits:
            # Highest candidate always qualifies as fallback
            result_bits = bits
            break

        n_centroids = 1 << bits
        cb = _lloyd_per_channel(W_bulk, n_centroids, lloyd_iters)
        idx = _assign_indices(W_bulk, cb)
        rec = torch.gather(cb, 1, idx)
        if mask is not None:
            # Overwrite outliers in-place on rec — avoids a full clone
            # that would double peak allocation on huge MoE stacks.
            mask_flat = mask.reshape(-1)
            rec_flat = rec.reshape(-1)
            rec_flat[mask_flat] = W.reshape(-1)[mask_flat]

        # Per-channel cos similarity — sole quality gate
        cos_per_ch = F.cosine_similarity(W, rec, dim=1)  # [N_out]
        median_cos = float(cos_per_ch.median().item())

        del cb, idx, rec
        if W.is_cuda:
            torch.cuda.empty_cache()

        if median_cos >= min_cos:
            result_bits = bits
            break

    # Release bulky intermediates before returning (UMA pressure matters)
    del W_bulk
    if mask is not None:
        del mask
    if W.is_cuda:
        torch.cuda.empty_cache()
    return result_bits


# ─── Public entry point ────────────────────────────────────────────


def xfp_pack(
    W: torch.Tensor,
    bits: int,
    lloyd_iters: int = 20,
    also_score_widths: tuple[int, ...] = (),
    outlier_sigma: Optional[float] = None,
    outlier_max_fraction: float = 0.02,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    XFPPackStats,
]:
    """Pack a weight matrix via per-channel Lloyd codebook + bit-packed indices.

    Args:
        W: [N_out, K] weight tensor. Caller converts from bf16/fp16 to fp32.
        bits: target bit width; one of {2, 3, 4}. Determines codebook size
            (2^bits entries) and pack layout per §3.3 of XFP.PAPER.md.
        lloyd_iters: number of Lloyd refinement iterations (20–50 typical).
        also_score_widths: additional bit widths to fit a codebook for and
            score only for MSE, without packing. Used to populate
            stats.mse_per_bits as a per-layer auto-size signal.
        outlier_sigma: if set, extract weights with |w - mean| > sigma * std
            as sparse residuals BEFORE fitting the codebook. Paper §4 Step 2.
            Typical values 3.0–4.0. None disables outlier extraction.
            Based on GLM-4.7-Flash weight inspection (tests/xfp/inspect_
            distribution.py), 4.0 is the sweet spot for MoE models: it
            catches the 40σ attention outliers (kv_b_proj, q_b_proj) while
            marking only ~0.01–0.8 % of weights, comfortably below the
            15 % sparse-path threshold of Paper §4 Step 1.
        outlier_max_fraction: safety cap. If the k-sigma threshold would
            mark more than this fraction of weights as outliers (e.g. for
            a broad-spectrum distribution where the split is counter-
            productive), keep only the top-by-magnitude
            `outlier_max_fraction * numel` weights and reclassify the rest
            as bulk. Default 0.02 (= 2 %). Paper §4 says ratios above ~30 %
            should drop sparse extraction entirely — we use a tighter
            cap because v1 has no inline sparse kernel yet and larger
            outlier sets hurt the apply-path throughput.
        seed: random seed (reserved for future stochastic variants).

    Returns:
        packed: [K_packed, N_out] int32 (bit-identical to uint32)
        codebook: [N_out, 2^bits] fp16
        outlier_indices: [n_outliers] int64 flat index (row*K + col), or None
        outlier_values: [n_outliers] fp16 original weight values, or None
        stats: XFPPackStats

    When `outlier_sigma` is None the two outlier tensors are None and the
    returned pipeline matches the v1 bulk-only encoder exactly.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"xfp_pack: unsupported bits={bits}, must be in {{2,3,4}}")
    if W.dim() != 2:
        raise ValueError(f"xfp_pack: W must be 2D [N_out, K], got shape {tuple(W.shape)}")

    del seed  # reserved

    W = W.to(torch.float32)
    N_out, K = W.shape
    n_centroids = 1 << bits

    # Outlier split (Paper §4 Step 2) — optional. Runs BEFORE Lloyd so the
    # codebook fits the cleaned bulk distribution.
    outlier_indices: Optional[torch.Tensor] = None
    outlier_values: Optional[torch.Tensor] = None
    outlier_count = 0
    outlier_fraction = 0.0
    if outlier_sigma is not None and outlier_sigma > 0:
        mu = W.mean()
        sigma = W.std()
        total_numel = W.numel()
        centered_abs = (W - mu).abs()
        threshold = float(outlier_sigma) * sigma
        mask = centered_abs > threshold  # [N_out, K] bool
        nnz = int(mask.sum().item())
        max_allowed = int(outlier_max_fraction * total_numel)

        if nnz > max_allowed and max_allowed > 0:
            # Safety cap: keep only the top-by-magnitude weights. Anything
            # beyond outlier_max_fraction is clamped back into the bulk.
            # Uses a top-k on the flattened abs-centered tensor.
            flat_abs = centered_abs.reshape(-1)
            _, top_flat_idx = torch.topk(
                flat_abs, max_allowed, largest=True, sorted=False
            )
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[top_flat_idx] = True
            mask = mask.reshape_as(W)
            nnz = max_allowed

        if nnz > 0:
            # Flat indices so the apply path can split (row, col) cheaply.
            flat_mask = mask.reshape(-1)
            outlier_indices = flat_mask.nonzero(as_tuple=False).squeeze(1).to(torch.int64)
            outlier_values = W.reshape(-1)[outlier_indices].to(torch.float16)
            # Replace outlier positions with the layer mean so Lloyd fits
            # the bulk without being pulled by extreme values. Using mean
            # (not zero) avoids creating a new cluster at zero that would
            # consume a codebook entry.
            W_bulk = W.clone()
            W_bulk[mask] = mu
            outlier_count = nnz
            outlier_fraction = float(nnz) / float(total_numel)
        else:
            W_bulk = W
    else:
        W_bulk = W

    # 1. Lloyd codebook on the (possibly cleaned) bulk
    codebook_fp32 = _lloyd_per_channel(W_bulk, n_centroids, lloyd_iters)

    # 2. Assign indices (still on the bulk — outlier positions will be
    #    reconstructed via the scatter-add path at apply time, so their
    #    codebook index is irrelevant)
    idx = _assign_indices(W_bulk, codebook_fp32)

    # 3. Reconstruction (for MSE / cos sim) — include outlier correction
    #    so stats reflect the full XFP reconstruction, not just the bulk.
    W_rec_bulk = torch.gather(codebook_fp32, 1, idx)
    if outlier_indices is not None and outlier_values is not None:
        W_rec = W_rec_bulk.clone()
        flat_rec = W_rec.reshape(-1)
        flat_rec[outlier_indices] = outlier_values.to(torch.float32)
        W_rec = flat_rec.reshape(N_out, K)
    else:
        W_rec = W_rec_bulk

    # 4. Pack indices to word-aligned uint32
    packed = _pack_indices(idx, bits)

    # 5. Stats — distribution stats use the ORIGINAL W (incl. outliers)
    w_mean, w_std, w_abs_max, k3, k4 = _distribution_stats(W)
    mse, max_abs_err, cos_sim = _reconstruction_stats(W, W_rec)
    rmse_rel = (mse ** 0.5) / max(w_std, 1e-12)

    mse_per_bits: dict[int, float] = {bits: mse}
    for b in also_score_widths:
        if b == bits or b in mse_per_bits:
            continue
        if b not in (2, 3, 4):
            continue
        mse_per_bits[b] = _score_mse_only(W_bulk, b, lloyd_iters)

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

    # Paper-analysis histograms (cheap: O(N_out*K) on already-materialised
    # tensors). <1 KB per layer after to_dict() → negligible in manifest.
    cos_hist = _compute_cos_hist(W, W_rec, n_bins=20)
    outlier_hist = _compute_outlier_hist(W, float(w_mean), float(w_std))

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
        outlier_count=outlier_count,
        outlier_sigma=float(outlier_sigma or 0.0),
        outlier_fraction=outlier_fraction,
        mse_per_bits=mse_per_bits,
        recommended_bits=recommended_bits,
        recommended_gap=recommended_gap,
        cos_hist=cos_hist,
        outlier_hist=outlier_hist,
        # Gate-survival is only populated in xfp_auto_select path;
        # xfp_pack itself runs at a pre-chosen bits, so empty here.
        bits_survived_gate=tuple(),
    )

    return (
        packed,
        codebook_fp32.to(torch.float16),
        outlier_indices,
        outlier_values,
        stats,
    )


# ─── XFP-V2: per-group quant + shared codebook library ─────────────────
#
# V1 design: each output row gets its own learned 16-centroid codebook
# fitted on the row's K weights.
#
# V2 design (this section): each output row is split into G groups of
# `group_size` weights. Each group references one of `library_size`
# shared prototype codebooks (16 centroids, normalized to [-1, +1]).
# Per group we also store one fp16 scale + midpoint so the same
# normalized library entry can serve groups with different magnitudes.
#
# Reuses (unchanged):
#   - _lloyd_per_channel (operates on [N*G, group_size] reshape)
#   - _pack_indices (existing 4-bit packing)
#   - xfp_repack (warp-interleaved layout for kernel)
#   - outlier extraction (orthogonal — wired in v2 same path as v1)
#
# New (this section):
#   - _build_codebook_library (k-means over normalized group codebooks)
#   - xfp_pack_v2 (top-level orchestration)
#   - dequant_xfp_v2 (Python reference for the kernel)


@dataclass
class XFPPackV2Stats:
    """Statistics for XFP-V2 pack: per-group + shared library."""

    bits: int
    group_size: int
    library_size: int  # number of prototype codebooks in library
    shape: tuple[int, int]
    mse: float
    cos_sim: float
    # Library-coverage diagnostics: how good is the per-group nearest-lib match?
    library_p5_cos: float = 0.0
    library_min_cos: float = 0.0
    # Group params overhead: bits/param added by per-group scale+mid
    overhead_bits_per_param: float = 0.0


def _build_codebook_library(
    cb_norm: torch.Tensor,
    library_size: int,
    iters: int = 30,
    seed: int = 0,
) -> torch.Tensor:
    """K-means over normalized group codebooks → [library_size, n_centroids].

    Each codebook is treated as a 16-D point. We use k-means++-style init
    (sample first centroid uniformly, subsequent ones biased toward the
    farthest unassigned points) followed by Lloyd refinement.
    """
    M, D = cb_norm.shape
    if library_size >= M:
        # Library can hold every codebook — return them as-is (deduped not necessary).
        return cb_norm.clone()

    g = torch.Generator(device=cb_norm.device).manual_seed(seed)

    # k-means++ init
    idx0 = int(torch.randint(
        0, M, (1,), generator=g, device=cb_norm.device).item())
    cents = cb_norm[idx0:idx0 + 1].clone()

    # Distance computation is the memory hog: ((cb_norm[:, None] - cents[None]) ** 2).sum(-1)
    # materializes [M, k, D] which is ~13 GB for 122B (M=6.3M, k=32, D=16). Use
    # torch.cdist (fused, no broadcast intermediate, peak ≈ output size only).
    def _pairwise_d2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # cdist returns euclidean distance; we want squared, so square output.
        return torch.cdist(a, b, p=2.0).pow(2)

    while cents.shape[0] < library_size:
        d2 = _pairwise_d2(cb_norm, cents).amin(dim=1)
        prob = d2 / d2.sum().clamp(min=1e-12)
        nxt = int(torch.multinomial(prob, 1, generator=g).item())
        cents = torch.cat([cents, cb_norm[nxt:nxt + 1]], dim=0)

    # Lloyd refinement
    for _ in range(iters):
        d2 = _pairwise_d2(cb_norm, cents)  # [M, k] — no [M,k,D] intermediate
        assign = d2.argmin(dim=1)
        new_cents = cents.clone()
        for c in range(library_size):
            mask = assign == c
            if mask.any():
                new_cents[c] = cb_norm[mask].mean(dim=0)
        cents = new_cents
    return cents


def xfp_pack_v2(
    W: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    library_size: int = 32,
    lloyd_iters: int = 20,
    library_iters: int = 30,
) -> tuple[
    torch.Tensor,  # packed indices [K_packed, N] int32 (existing _pack_indices format)
    torch.Tensor,  # library [library_size, n_centroids] fp16
    torch.Tensor,  # group_lib_id [N, G] int32 (uint8 viable when library_size ≤ 256)
    torch.Tensor,  # group_scale  [N, G] fp16
    torch.Tensor,  # group_mid    [N, G] fp16
    XFPPackV2Stats,
]:
    """V2 pack: per-group Lloyd + shared codebook library.

    Reuses _lloyd_per_channel / _pack_indices unchanged. Adds library
    construction and per-group reference assignment.

    Currently outlier extraction is left out — it can be wired in later
    on the residual after library reconstruction (orthogonal to library
    learning). Keeping V1 + outliers as the existing baseline.
    """
    if W.dim() != 2:
        raise ValueError(f"xfp_pack_v2: W must be 2D, got {W.dim()}D")
    if bits not in (2, 3, 4):
        raise ValueError(f"xfp_pack_v2: bits must be in (2,3,4), got {bits}")
    Wf = W.float()
    N, K = Wf.shape
    if K % group_size != 0:
        raise ValueError(
            f"xfp_pack_v2: K={K} not divisible by group_size={group_size}"
        )
    G = K // group_size
    n_centroids = 1 << bits

    # Step 1 — reshape rows into groups: each group is one "virtual row"
    # for Lloyd. [N, K] → [N*G, group_size]
    W_groups = Wf.reshape(N, G, group_size).reshape(N * G, group_size)

    # Step 2 — fit per-group codebooks (REUSE existing _lloyd_per_channel)
    cb_per_group = _lloyd_per_channel(W_groups, n_centroids, lloyd_iters)
    # cb_per_group: [N*G, n_centroids] fp32

    # Step 3 — normalize each group codebook to [-1, +1]; record scale+mid
    cb_min = cb_per_group.amin(dim=1, keepdim=True)
    cb_max = cb_per_group.amax(dim=1, keepdim=True)
    midpoint = (cb_min + cb_max) / 2  # [N*G, 1]
    scale = ((cb_max - cb_min) / 2).clamp(min=1e-12)  # [N*G, 1]
    cb_norm = (cb_per_group - midpoint) / scale  # [N*G, n_centroids] in [-1, +1]

    # Step 4 — build shared library via k-means over the normalized codebooks
    library = _build_codebook_library(cb_norm, library_size, iters=library_iters)
    # [library_size, n_centroids] fp32, in [-1, +1]

    # Step 5 — assign each group's normalized codebook to the nearest library entry
    d2 = ((cb_norm.unsqueeze(1) - library.unsqueeze(0)) ** 2).sum(-1)  # [N*G, library_size]
    group_lib_id = d2.argmin(dim=1)  # [N*G]

    # Library-coverage diagnostics
    nearest = library[group_lib_id]
    cb_cos = F.cosine_similarity(cb_norm, nearest, dim=1)
    cb_cos_sorted = cb_cos.sort().values
    p5 = float(cb_cos_sorted[max(0, int(0.05 * cb_cos.numel()) - 1)].item())
    cb_min_cos = float(cb_cos_sorted[0].item())

    # Step 6 — re-assign weight indices given the chosen library codebook
    # For each weight w in group g: idx = argmin_k |w_norm - library[lib_id, k]|
    W_norm_groups = (W_groups - midpoint) / scale  # [N*G, group_size]
    chosen_lib = library[group_lib_id]  # [N*G, n_centroids]
    d_w = (W_norm_groups.unsqueeze(-1) - chosen_lib.unsqueeze(1)).abs()
    idx_per_group = d_w.argmin(dim=-1).to(torch.int32)  # [N*G, group_size]
    # Reassemble back to [N, K]
    idx_full = idx_per_group.reshape(N, G, group_size).reshape(N, K)

    # Step 7 — pack indices via existing _pack_indices (UNCHANGED layout)
    packed = _pack_indices(idx_full, bits)

    # Reconstruction for stats
    rec_norm = torch.gather(chosen_lib, 1, idx_per_group.long())  # [N*G, group_size]
    W_rec_groups = rec_norm * scale + midpoint  # [N*G, group_size]
    W_rec = W_rec_groups.reshape(N, K)
    diff = Wf - W_rec
    mse = float((diff * diff).mean().item())
    cos_sim = float(F.cosine_similarity(
        Wf.reshape(-1).unsqueeze(0),
        W_rec.reshape(-1).unsqueeze(0),
        dim=1,
    ).item())

    # Reshape group params to [N, G]
    group_lib_id_2d = group_lib_id.reshape(N, G).to(torch.int32)
    group_scale_2d = scale.reshape(N, G).to(torch.float16)
    group_mid_2d = midpoint.reshape(N, G).to(torch.float16)

    # Bits/param overhead from group params (scale+mid in fp16, lib_id 4-8 bit).
    # Per group: 2*16 = 32 bits for scale+mid + 8 bits lib_id (uint8 fits ≤256)
    lib_id_bits = 4 if library_size <= 16 else 8 if library_size <= 256 else 32
    overhead_bits = (2 * 16 + lib_id_bits) / group_size

    stats = XFPPackV2Stats(
        bits=bits,
        group_size=group_size,
        library_size=library_size,
        shape=(N, K),
        mse=mse,
        cos_sim=cos_sim,
        library_p5_cos=p5,
        library_min_cos=cb_min_cos,
        overhead_bits_per_param=overhead_bits,
    )

    return packed, library.to(torch.float16), group_lib_id_2d, group_scale_2d, group_mid_2d, stats


def _unpack_indices(packed: torch.Tensor, K: int, bits: int) -> torch.Tensor:
    """Reverse of `_pack_indices`. Returns [N_out, K] int64 indices.

    Extracted from `dequant_xfp` so V2 reference paths can reuse it.
    """
    vals_per_word = {2: 16, 3: 10, 4: 8}[bits]
    mask = (1 << bits) - 1
    K_packed = packed.shape[0]
    N_out = packed.shape[1]
    assert K_packed * vals_per_word >= K
    packed_nk = packed.t().to(torch.int64)
    unpacked = torch.zeros(
        N_out, K_packed * vals_per_word,
        dtype=torch.int64, device=packed.device,
    )
    for slot in range(vals_per_word):
        unpacked[:, slot::vals_per_word] = (packed_nk >> (slot * bits)) & mask
    return unpacked[:, :K]


def dequant_xfp_v2_packed(
    packed: torch.Tensor,          # [K_packed, N] int32 (existing _pack_indices format)
    library: torch.Tensor,         # [library_size, n_centroids] fp16/fp32
    group_lib_id: torch.Tensor,    # [N, G] int (uint8 / int32)
    group_scale: torch.Tensor,     # [N, G] fp16/fp32
    group_mid: torch.Tensor,       # [N, G] fp16/fp32
    K: int,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """V2 reference dequant from packed indices.

    Wrapper that unpacks via `_unpack_indices` and applies the V2 lookup
    formula. Mirrors what the v17_lib kernel will compute, but in pure
    PyTorch (slow). Used by `online_linear.apply()` V2 branch until the
    kernel ships.
    """
    idx = _unpack_indices(packed, K, bits)
    return dequant_xfp_v2(library, group_lib_id, group_scale, group_mid,
                          idx, group_size)


def xfp_moe_pack_v2(
    W_stack: torch.Tensor,         # [E, N, K] bf16/fp32 — one stack (w13 or w2)
    bits: int = 4,
    group_size: int = 128,
    library_size: int = 32,
    lloyd_iters: int = 5,          # MoE default (per-expert is fast)
    library_iters: int = 30,
) -> tuple[
    torch.Tensor,  # packed [E, K_packed, N] int32
    torch.Tensor,  # library [L, n_centroids] fp16    (shared across experts)
    torch.Tensor,  # group_lib_id [E, N, G] int32
    torch.Tensor,  # group_scale  [E, N, G] fp16
    torch.Tensor,  # group_mid    [E, N, G] fp16
    XFPPackV2Stats,
]:
    """V2 MoE pack with shared library across experts in one stack.

    Per-expert per-group Lloyd codebooks are first fitted independently,
    then ALL such codebooks (across all E experts) are clustered into a
    single shared library. Each (expert, row, group) triple references
    that shared library via a small lib_id.

    Memory rationale: with E=128 experts, N=1024, K=2048, group_size=128,
    we have 128·1024·16 = 2M per-group codebooks before sharing. The
    library compresses these to L≈32 prototypes (same insight as the
    Linear case — gaussian distributions cluster heavily).
    """
    if W_stack.dim() != 3:
        raise ValueError(f"xfp_moe_pack_v2: W must be 3D [E,N,K], got {W_stack.shape}")
    Wf = W_stack.float()
    E, N, K = Wf.shape
    if K % group_size != 0:
        raise ValueError(
            f"xfp_moe_pack_v2: K={K} not divisible by group_size={group_size}"
        )
    G = K // group_size
    n_centroids = 1 << bits

    # Step 1 — fit per-expert per-group codebooks (REUSE _lloyd_per_channel)
    all_cb_norm: list[torch.Tensor] = []
    all_scale: list[torch.Tensor] = []
    all_mid: list[torch.Tensor] = []
    for e in range(E):
        W_e = Wf[e].reshape(N, G, group_size).reshape(N * G, group_size)
        cb_e = _lloyd_per_channel(W_e, n_centroids, lloyd_iters)
        cb_min = cb_e.amin(dim=1, keepdim=True)
        cb_max = cb_e.amax(dim=1, keepdim=True)
        mid_e = (cb_min + cb_max) / 2
        scale_e = ((cb_max - cb_min) / 2).clamp(min=1e-12)
        all_cb_norm.append((cb_e - mid_e) / scale_e)        # [N*G, n_cents]
        all_scale.append(scale_e.squeeze(-1))                # [N*G]
        all_mid.append(mid_e.squeeze(-1))                    # [N*G]

    # Step 2 — build SHARED library across all experts in this stack
    cb_norm_all = torch.cat(all_cb_norm, dim=0)  # [E*N*G, n_cents]
    library = _build_codebook_library(cb_norm_all, library_size, iters=library_iters)
    # [library_size, n_centroids] fp32 in [-1, +1]

    # Step 3 — per-expert: assign + re-quantize + pack (REUSE _pack_indices)
    packed_list: list[torch.Tensor] = []
    lib_id_list: list[torch.Tensor] = []
    scale_list: list[torch.Tensor] = []
    mid_list: list[torch.Tensor] = []
    library_p5_min = 1.0
    library_min_min = 1.0
    total_mse = 0.0
    total_cos_num = 0.0
    total_cos_den_a = 0.0
    total_cos_den_b = 0.0
    for e in range(E):
        cb_norm_e = all_cb_norm[e]  # [N*G, n_cents]
        scale_e = all_scale[e]      # [N*G]
        mid_e = all_mid[e]          # [N*G]
        # Library assignment
        d2 = ((cb_norm_e.unsqueeze(1) - library.unsqueeze(0)) ** 2).sum(-1)
        lib_id_e = d2.argmin(dim=1)  # [N*G]
        # Diagnostics
        nearest = library[lib_id_e]
        cb_cos = F.cosine_similarity(cb_norm_e, nearest, dim=1)
        cb_cos_sorted = cb_cos.sort().values
        p5 = float(cb_cos_sorted[max(0, int(0.05 * cb_cos.numel()) - 1)].item())
        library_p5_min = min(library_p5_min, p5)
        library_min_min = min(library_min_min, float(cb_cos_sorted[0].item()))
        # Re-quantize weights with chosen library codebook
        W_groups_e = Wf[e].reshape(N, G, group_size).reshape(N * G, group_size)
        W_norm_e = (W_groups_e - mid_e.unsqueeze(-1)) / scale_e.unsqueeze(-1)
        chosen_lib = library[lib_id_e]  # [N*G, n_cents]
        d_w = (W_norm_e.unsqueeze(-1) - chosen_lib.unsqueeze(1)).abs()
        idx_e = d_w.argmin(dim=-1).to(torch.int32)  # [N*G, group_size]
        idx_full_e = idx_e.reshape(N, G, group_size).reshape(N, K)
        # Pack indices
        packed_e = _pack_indices(idx_full_e, bits)
        packed_list.append(packed_e)
        lib_id_list.append(lib_id_e.reshape(N, G).to(torch.int32))
        scale_list.append(scale_e.reshape(N, G).to(torch.float16))
        mid_list.append(mid_e.reshape(N, G).to(torch.float16))
        # MSE/cos for stats (running aggregates)
        rec_norm = torch.gather(chosen_lib, 1, idx_e.long())
        W_rec_e = (rec_norm * scale_e.unsqueeze(-1) + mid_e.unsqueeze(-1)).reshape(N, K)
        diff = Wf[e] - W_rec_e
        total_mse += float((diff * diff).sum().item())
        total_cos_num += float((Wf[e] * W_rec_e).sum().item())
        total_cos_den_a += float((Wf[e] * Wf[e]).sum().item())
        total_cos_den_b += float((W_rec_e * W_rec_e).sum().item())

    n_total = float(E * N * K)
    mse = total_mse / n_total
    cos_sim = total_cos_num / max(
        (total_cos_den_a ** 0.5) * (total_cos_den_b ** 0.5), 1e-12
    )
    lib_id_bits = 4 if library_size <= 16 else 8 if library_size <= 256 else 32
    overhead_bits = (2 * 16 + lib_id_bits) / group_size

    packed = torch.stack(packed_list, dim=0)         # [E, K_packed, N]
    group_lib_id = torch.stack(lib_id_list, dim=0)   # [E, N, G]
    group_scale = torch.stack(scale_list, dim=0)     # [E, N, G]
    group_mid = torch.stack(mid_list, dim=0)         # [E, N, G]

    stats = XFPPackV2Stats(
        bits=bits, group_size=group_size, library_size=library_size,
        shape=(E * N, K), mse=mse, cos_sim=cos_sim,
        library_p5_cos=library_p5_min,
        library_min_cos=library_min_min,
        overhead_bits_per_param=overhead_bits,
    )
    return packed, library.to(torch.float16), group_lib_id, group_scale, group_mid, stats


def dequant_xfp_v2(
    library: torch.Tensor,         # [library_size, n_centroids] fp16/fp32
    group_lib_id: torch.Tensor,    # [N, G] int (uint8 / int32)
    group_scale: torch.Tensor,     # [N, G] fp16/fp32
    group_mid: torch.Tensor,       # [N, G] fp16/fp32
    idx: torch.Tensor,             # [N, K] int — UNPACKED indices in [0, n_centroids)
    group_size: int,
) -> torch.Tensor:
    """Python reference reconstruction. Returns W_rec [N, K] fp32.

    Mirrors what the v17_lib kernel must compute:
      W[n, k] = group_scale[n, g] * library[group_lib_id[n, g], idx[n, k]]
               + group_mid[n, g]
    where g = k // group_size.

    Note: this takes UNPACKED idx for clarity. The kernel will work
    directly on packed uint32 indices via the same unpack pattern as v12.
    """
    N, K = idx.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}")
    G = K // group_size
    idx_g = idx.reshape(N, G, group_size).long()
    # library [L, n_cents], group_lib_id [N, G] → gather → [N, G, n_cents]
    lib_per_group = library.float()[group_lib_id.long()]
    # gather centroids per weight: [N, G, group_size]
    cb_norm = torch.gather(lib_per_group, 2, idx_g)
    W_rec = cb_norm * group_scale.float().unsqueeze(-1) + group_mid.float().unsqueeze(-1)
    return W_rec.reshape(N, K)
