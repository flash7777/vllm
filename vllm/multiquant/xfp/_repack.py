# SPDX-License-Identifier: Apache-2.0
"""On-device repack/un-repack for XFP packed tensors.

The cache stores packed tensors in pre-repack 2D form ``[K_packed, N]``
(per-expert for MoE: ``[E, K_packed, N]``). The forward kernel expects a
warp-interleaved 1D layout ``[K_groups * N * warp_size]`` for coalesced
memory access. This split makes:

- **Cache storage** TP-slicable with simple ``narrow()`` calls along K
  (dim 0 / dim 1 for MoE) or N (dim 1 / dim 2 for MoE).
- **Forward kernel** still benefits from warp-interleaved coalescing.

The conversion happens once at load time, on-device, after TP-slicing has
already reduced the tensor to the per-rank shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

WARP_SIZE = 32


def repack_2d_to_warp_flat(
    packed_2d: torch.Tensor, warp_size: int = WARP_SIZE,
) -> torch.Tensor:
    """[K_packed, N] int32 → [K_groups * N * warp_size] int32 flat.

    Pads K_packed up to a multiple of warp_size with zeros if needed.
    """
    K_packed, N = packed_2d.shape
    K_groups = (K_packed + warp_size - 1) // warp_size
    if K_packed % warp_size != 0:
        pad = warp_size - K_packed % warp_size
        packed_2d = F.pad(packed_2d, (0, 0, 0, pad), value=0)
    return (
        packed_2d.reshape(K_groups, warp_size, N)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(-1)
    )


def repack_3d_moe_to_flat(
    packed_3d: torch.Tensor, warp_size: int = WARP_SIZE,
) -> torch.Tensor:
    """[E, K_packed, N] int32 → [E * K_groups * N * warp_size] int32 flat.

    Per-expert repack, concatenated. Used by the MoE forward kernel which
    reads ``offsets[e] = e * K_groups * N * warp_size``.
    """
    E, K_packed, N = packed_3d.shape
    K_groups = (K_packed + warp_size - 1) // warp_size
    if K_packed % warp_size != 0:
        pad = warp_size - K_packed % warp_size
        packed_3d = F.pad(packed_3d, (0, 0, 0, pad), value=0)
    # [E, K_groups, warp_size, N] → [E, K_groups, N, warp_size] → flatten
    return (
        packed_3d.reshape(E, K_groups, warp_size, N)
        .permute(0, 1, 3, 2)
        .contiguous()
        .reshape(-1)
    )
