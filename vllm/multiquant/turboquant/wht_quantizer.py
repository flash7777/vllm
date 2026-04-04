# SPDX-License-Identifier: Apache-2.0
"""TurboQuant v2 quantizer — WHT block compression.

Uses Walsh-Hadamard Transform (WHT) on configurable block sizes
instead of full D×D random rotation. No per-layer state needed.

Reference: github.com/animehacker/llama-turboquant
"""

from __future__ import annotations

import torch
from torch import Tensor

from vllm.multiquant.base import KVQuantizer, KVQuantizerConfig
from vllm.multiquant.shared.centroids import get_wht_centroids, get_wht_thresholds
from vllm.multiquant.shared.wht import wht_forward, wht_inverse
from vllm.multiquant.turboquant.wht_config import TurboQuantWHTConfig


class TurboQuantWHTQuantizer(KVQuantizer):
    """WHT-based TurboQuant quantizer.

    No per-layer state (Pi, S matrices) — WHT is deterministic.
    Only stores universal centroids (8 values for 3-bit).
    """

    def init_buffers(self, head_dim: int, seed: int) -> dict[str, Tensor]:
        """No per-layer buffers needed for WHT mode."""
        return {}

    def pack(
        self,
        vector: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compress float vectors to packed uint8 using WHT + block quantization.

        Args:
            vector: [N, D] float tensor
            buffers: unused (WHT has no per-layer state)
            config: TurboQuantWHTConfig

        Returns:
            [N, packed_size] uint8 tensor
        """
        assert isinstance(config, TurboQuantWHTConfig)
        return pack_wht(vector.float(), config)

    def unpack(
        self,
        packed: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Decompress packed uint8 back to float vector."""
        assert isinstance(config, TurboQuantWHTConfig)
        return unpack_wht(packed, config)

    def attention_score(
        self,
        query: Tensor,
        packed_key: Tensor,
        buffers: dict[str, Tensor],
        config: KVQuantizerConfig,
    ) -> Tensor:
        """Compute Q·K score from compressed key.

        For WHT mode: transform Q into WHT space, then dot with
        reconstructed K in WHT space (no inverse WHT needed for score).
        """
        assert isinstance(config, TurboQuantWHTConfig)
        # Decompress K and dot with Q
        k_recon = unpack_wht(packed_key, config)
        return (query.float() * k_recon).sum(dim=-1)


def pack_wht(
    x: Tensor, config: TurboQuantWHTConfig
) -> Tensor:
    """Pack float vectors using WHT block compression.

    Args:
        x: [N, D] float32 tensor
        config: WHT config with block_size and total_bits

    Returns:
        [N, packed_size] uint8 tensor
    """
    N, D = x.shape
    bs = config.block_size
    bits = config.mse_bits
    n_blocks = D // bs

    # Step 1: WHT transform (per block)
    rotated = wht_forward(x, block_size=bs)  # [N, D]
    blocks = rotated.reshape(N, n_blocks, bs)  # [N, n_blocks, bs]

    # Step 2: Per-block amax normalization
    # Cache centroids/thresholds on GPU (host→device transfer is NOT graph-safe)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [N, n_blocks, 1]
    _cache_key = (bits, x.device)
    if not hasattr(pack_wht, '_gpu_cache'):
        pack_wht._gpu_cache = {}
    if _cache_key not in pack_wht._gpu_cache:
        c = get_wht_centroids(bits).to(x.device)
        t = get_wht_thresholds(bits).to(x.device)
        pack_wht._gpu_cache[_cache_key] = (c, t)
    centroids, thresholds = pack_wht._gpu_cache[_cache_key]
    outermost = centroids[-1]  # tensor, no .item()
    gamma = (amax / outermost).squeeze(-1)  # [N, n_blocks]
    normalized = blocks / (gamma.unsqueeze(-1) + 1e-10)
    # idx[i] = number of thresholds that normalized[i] exceeds
    idx = (normalized.unsqueeze(-1) > thresholds).sum(dim=-1).to(torch.uint8)
    # [N, n_blocks, bs] uint8 indices

    # Step 4: Bitpack per block — VECTORIZED (no Python loops, graph-safe)
    bpb = config.bytes_per_block
    idx_int = idx.to(torch.int32)  # [N, n_blocks, bs]

    # Cache constant shift tensors on GPU (graph-safe: no host→device during capture)
    if not hasattr(pack_wht, '_shift_cache'):
        pack_wht._shift_cache = {}
    _sk = x.device
    if _sk not in pack_wht._shift_cache:
        pack_wht._shift_cache[_sk] = {
            'qs': torch.tensor([0, 2, 4, 6], device=_sk, dtype=torch.int32),
            'qr': torch.arange(8, device=_sk, dtype=torch.int32),
            '4b': torch.tensor([0, 4], device=_sk, dtype=torch.int32),
        }
    _shifts = pack_wht._shift_cache[_sk]

    if bits == 3:
        qs_bytes = bs * 2 // 8
        qr_bytes = bs // 8
        low2 = (idx_int & 0x3).reshape(N, n_blocks, qs_bytes, 4)
        qs = (low2 << _shifts['qs']).sum(dim=-1).to(torch.uint8)
        hi1 = ((idx_int >> 2) & 1).reshape(N, n_blocks, qr_bytes, 8)
        qr = (hi1 << _shifts['qr']).sum(dim=-1).to(torch.uint8)
        # Gamma as fp16
        gamma_bytes = gamma.to(torch.float16).view(torch.uint8).reshape(N, n_blocks, 2)
        # Concat: [qs | qr | gamma]
        packed = torch.cat([qs, qr, gamma_bytes], dim=-1)  # [N, n_blocks, bpb]
    elif bits == 4:
        n_bytes = bs // 2
        idx_pairs = idx_int.reshape(N, n_blocks, n_bytes, 2)
        packed_bytes = (idx_pairs[..., 0] & 0xF) | ((idx_pairs[..., 1] & 0xF) << 4)
        packed_bytes = packed_bytes.to(torch.uint8)
        gamma_bytes = gamma.to(torch.float16).view(torch.uint8).reshape(N, n_blocks, 2)
        packed = torch.cat([packed_bytes, gamma_bytes], dim=-1)
    elif bits == 2:
        n_bytes = bs // 4
        idx_quads = idx_int.reshape(N, n_blocks, n_bytes, 4)
        packed_bytes = ((idx_quads & 0x3) << _shifts['qs']).sum(dim=-1).to(torch.uint8)
        gamma_bytes = gamma.to(torch.float16).view(torch.uint8).reshape(N, n_blocks, 2)
        packed = torch.cat([packed_bytes, gamma_bytes], dim=-1)
    else:
        raise ValueError(f"WHT pack: unsupported bits={bits}")

    return packed.reshape(N, -1)  # [N, packed_size]


def unpack_wht(
    packed: Tensor, config: TurboQuantWHTConfig
) -> Tensor:
    """Unpack compressed uint8 back to float vectors.

    Args:
        packed: [N, packed_size] uint8 tensor
        config: WHT config

    Returns:
        [N, D] float32 tensor
    """
    N = packed.shape[0]
    D = config.head_dim
    bs = config.block_size
    bits = config.mse_bits
    n_blocks = D // bs
    bpb = config.bytes_per_block

    blocks_packed = packed.reshape(N, n_blocks, bpb)

    # Cache centroids on GPU (host→device NOT graph-safe)
    _ukey = (bits, packed.device)
    if not hasattr(unpack_wht, '_gpu_cache'):
        unpack_wht._gpu_cache = {}
    if _ukey not in unpack_wht._gpu_cache:
        unpack_wht._gpu_cache[_ukey] = get_wht_centroids(bits).to(packed.device)
    centroids = unpack_wht._gpu_cache[_ukey]

    # Unpack indices — VECTORIZED (graph-safe: cached shift tensors)
    if not hasattr(unpack_wht, '_shift_cache'):
        unpack_wht._shift_cache = {}
    _sk = packed.device
    if _sk not in unpack_wht._shift_cache:
        unpack_wht._shift_cache[_sk] = {
            'qs': torch.tensor([0, 2, 4, 6], device=_sk, dtype=torch.int32),
            'qr': torch.arange(8, device=_sk, dtype=torch.int32),
            '4b': torch.tensor([0, 4], device=_sk, dtype=torch.int32),
        }
    _sh = unpack_wht._shift_cache[_sk]

    if bits == 3:
        qs_bytes = bs * 2 // 8
        qr_bytes = bs // 8
        qs_raw = blocks_packed[:, :, :qs_bytes].to(torch.int32)
        low2 = ((qs_raw.unsqueeze(-1) >> _sh['qs']) & 0x3).reshape(N, n_blocks, bs)
        qr_raw = blocks_packed[:, :, qs_bytes:qs_bytes+qr_bytes].to(torch.int32)
        hi1 = ((qr_raw.unsqueeze(-1) >> _sh['qr']) & 1).reshape(N, n_blocks, bs)
        idx = (low2 | (hi1 << 2)).to(torch.long)
        gamma_off = qs_bytes + qr_bytes
    elif bits == 4:
        n_bytes = bs // 2
        raw = blocks_packed[:, :, :n_bytes].to(torch.int32)
        idx = ((raw.unsqueeze(-1) >> _sh['4b']) & 0xF).reshape(N, n_blocks, bs).to(torch.long)
        gamma_off = n_bytes
    elif bits == 2:
        n_bytes = bs // 4
        raw = blocks_packed[:, :, :n_bytes].to(torch.int32)
        idx = ((raw.unsqueeze(-1) >> _sh['qs']) & 0x3).reshape(N, n_blocks, bs).to(torch.long)
        gamma_off = n_bytes
    else:
        raise ValueError(f"WHT unpack: unsupported bits={bits}")

    # Unpack gamma (fp16) — vectorized
    gamma_raw = blocks_packed[:, :, gamma_off:gamma_off+2].contiguous()
    gamma_fp16 = gamma_raw.view(torch.uint8).reshape(N * n_blocks, 2).view(
        torch.float16).reshape(N, n_blocks).float()

    # Reconstruct: centroid lookup + scale + inverse WHT
    values_wht = centroids[idx] * gamma_fp16.unsqueeze(-1)  # [N, n_blocks, bs]
    values_flat = values_wht.reshape(N, D)  # [N, D]

    # Inverse WHT
    return wht_inverse(values_flat, block_size=bs)
