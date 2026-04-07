# SPDX-License-Identifier: Apache-2.0
"""GPTQ INT2 Linear — direct dequant without Marlin repack.

AutoRound INT2 models use standard GPTQ format (qweight/scales/qzeros)
with pack_factor=16 (16 x 2-bit values per int32). Marlin doesn't
support INT2 natively, so we dequant on-the-fly and use F.linear.

This is analogous to Archer but reads GPTQ format directly.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


def dequant_gptq_int2(
    qweight: torch.Tensor,    # [K_packed, N] int32, 16 values per int32
    scales: torch.Tensor,      # [n_groups, N] float16
    qzeros: torch.Tensor,      # [n_groups, N_zp_packed] int32
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize GPTQ INT2 packed weights to float16.

    Returns: [K, N] float16 weight matrix.
    """
    K_packed, N = qweight.shape
    K = K_packed * 16  # 16 x 2-bit per int32
    n_groups = K // group_size

    # Unpack 2-bit values: [K_packed, N] int32 → [K, N] uint8
    # Each int32 has 16 x 2-bit values, packed LSB-first
    shifts = torch.arange(0, 32, 2, device=qweight.device, dtype=torch.int32)
    # Reshape for broadcasting: [K_packed, 1, N] and [16] → [K_packed, 16, N]
    expanded = qweight.unsqueeze(1).expand(-1, 16, -1)
    shifted = (expanded >> shifts.view(1, 16, 1)) & 0x3
    unpacked = shifted.reshape(K, N)  # [K, N] values 0-3

    # Unpack zero points: same 2-bit packing
    zp_shifts = torch.arange(0, 32, 2, device=qzeros.device, dtype=torch.int32)
    zp_expanded = qzeros.unsqueeze(1).expand(-1, 16, -1)
    zp_shifted = (zp_expanded >> zp_shifts.view(1, 16, 1)) & 0x3
    zp_unpacked = zp_shifted.reshape(n_groups, -1)  # [n_groups, N_zp_unpacked]
    # Slice to N columns (may have padding)
    zp = zp_unpacked[:, :N]  # [n_groups, N]

    # Dequant: weight = scale * (qval - zero_point)
    # Group assignment: group_idx = k // group_size
    group_idx = torch.arange(K, device=qweight.device) // group_size
    w = scales[group_idx] * (unpacked.float() - zp[group_idx].float())

    return w.to(torch.float16)


class GPTQInt2LinearMethod:
    """Linear method for GPTQ INT2 without Marlin.

    Dequantizes on-the-fly per forward pass.
    Slower than Marlin but correct for INT2.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def apply(self, x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
        """x @ dequant(W).T"""
        W = dequant_gptq_int2(
            layer.qweight, layer.scales, layer.qzeros,
            self.group_size,
        )
        return F.linear(x, W)
