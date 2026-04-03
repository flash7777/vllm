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
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [N, n_blocks, 1]
    centroids = get_wht_centroids(bits).to(x.device)
    outermost = centroids[-1].item()  # e.g., 2.1573 for 3-bit
    gamma = (amax / outermost).squeeze(-1)  # [N, n_blocks] — scale factor
    normalized = blocks / (gamma.unsqueeze(-1) + 1e-10)  # [N, n_blocks, bs]

    # Step 3: Quantize to nearest centroid via thresholds (faster than argmin)
    thresholds = get_wht_thresholds(bits).to(x.device)
    # idx[i] = number of thresholds that normalized[i] exceeds
    idx = (normalized.unsqueeze(-1) > thresholds).sum(dim=-1).to(torch.uint8)
    # [N, n_blocks, bs] uint8 indices

    # Step 4: Bitpack per block
    bpb = config.bytes_per_block
    packed = torch.zeros(N, n_blocks, bpb, dtype=torch.uint8, device=x.device)

    if bits == 2:
        # 2-bit: 4 indices per byte
        for b in range(bs // 4):
            packed[:, :, b] = (
                (idx[:, :, b*4] & 0x3) |
                ((idx[:, :, b*4+1] & 0x3) << 2) |
                ((idx[:, :, b*4+2] & 0x3) << 4) |
                ((idx[:, :, b*4+3] & 0x3) << 6)
            )
        gamma_off = bs // 4
    elif bits == 3:
        # 3-bit split: lower 2 bits in qs, upper 1 bit in qr
        qs_bytes = bs * 2 // 8  # e.g., 8 for bs=32
        qr_bytes = bs // 8      # e.g., 4 for bs=32
        # Pack lower 2 bits (4 per byte)
        for b in range(qs_bytes):
            packed[:, :, b] = (
                (idx[:, :, b*4] & 0x3) |
                ((idx[:, :, b*4+1] & 0x3) << 2) |
                ((idx[:, :, b*4+2] & 0x3) << 4) |
                ((idx[:, :, b*4+3] & 0x3) << 6)
            )
        # Pack upper 1 bit (8 per byte)
        for b in range(qr_bytes):
            val = torch.zeros_like(packed[:, :, 0])
            for k in range(8):
                j = b * 8 + k
                if j < bs:
                    val = val | (((idx[:, :, j] >> 2) & 1) << k)
            packed[:, :, qs_bytes + b] = val
        gamma_off = qs_bytes + qr_bytes
    elif bits == 4:
        # 4-bit: 2 indices per byte
        for b in range(bs // 2):
            packed[:, :, b] = (
                (idx[:, :, b*2] & 0xF) |
                ((idx[:, :, b*2+1] & 0xF) << 4)
            )
        gamma_off = bs // 2
    else:
        raise ValueError(f"WHT pack: unsupported bits={bits}")

    # Store gamma as fp16 (2 bytes)
    gamma_fp16 = gamma.to(torch.float16)  # [N, n_blocks]
    gamma_bytes = gamma_fp16.view(torch.uint8).reshape(N, n_blocks, 2)
    packed[:, :, gamma_off:gamma_off+2] = gamma_bytes

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

    centroids = get_wht_centroids(bits).to(packed.device)

    # Unpack indices
    idx = torch.zeros(N, n_blocks, bs, dtype=torch.long, device=packed.device)

    if bits == 2:
        for b in range(bs // 4):
            byte_val = blocks_packed[:, :, b].to(torch.int32)
            idx[:, :, b*4]   = byte_val & 0x3
            idx[:, :, b*4+1] = (byte_val >> 2) & 0x3
            idx[:, :, b*4+2] = (byte_val >> 4) & 0x3
            idx[:, :, b*4+3] = (byte_val >> 6) & 0x3
        gamma_off = bs // 4
    elif bits == 3:
        qs_bytes = bs * 2 // 8
        qr_bytes = bs // 8
        # Lower 2 bits
        for b in range(qs_bytes):
            byte_val = blocks_packed[:, :, b].to(torch.int32)
            idx[:, :, b*4]   = byte_val & 0x3
            idx[:, :, b*4+1] = (byte_val >> 2) & 0x3
            idx[:, :, b*4+2] = (byte_val >> 4) & 0x3
            idx[:, :, b*4+3] = (byte_val >> 6) & 0x3
        # Upper 1 bit
        for b in range(qr_bytes):
            byte_val = blocks_packed[:, :, qs_bytes + b].to(torch.int32)
            for k in range(8):
                j = b * 8 + k
                if j < bs:
                    idx[:, :, j] = idx[:, :, j] | (((byte_val >> k) & 1) << 2)
        gamma_off = qs_bytes + qr_bytes
    elif bits == 4:
        # 4-bit: 2 indices per byte
        for b in range(bs // 2):
            byte_val = blocks_packed[:, :, b].to(torch.int32)
            idx[:, :, b*2]   = byte_val & 0xF
            idx[:, :, b*2+1] = (byte_val >> 4) & 0xF
        gamma_off = bs // 2
    else:
        raise ValueError(f"WHT unpack: unsupported bits={bits}")

    # Unpack gamma (fp16)
    gamma_bytes = blocks_packed[:, :, gamma_off:gamma_off+2].contiguous()
    gamma = gamma_bytes.view(torch.uint8).reshape(N, n_blocks, 2)
    gamma = gamma.view(torch.uint8).reshape(N * n_blocks, 2)
    gamma_fp16 = gamma.view(torch.float16).reshape(N, n_blocks).float()

    # Reconstruct: centroid lookup + scale + inverse WHT
    values_wht = centroids[idx] * gamma_fp16.unsqueeze(-1)  # [N, n_blocks, bs]
    values_flat = values_wht.reshape(N, D)  # [N, D]

    # Inverse WHT
    return wht_inverse(values_flat, block_size=bs)
