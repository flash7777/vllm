# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN online linear method — BF16 → INT4 packed at load time.

Simple: load BF16 normally, quantize in process_weights_after_loading,
store packed int32, decompress on-the-fly in apply().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.multiquant.autoround.config import AutoRoundRTNConfig

logger = init_logger(__name__)


class AutoRoundRTNLinearMethod(QuantizeMethodBase):
    """BF16 → INT4 symmetric RTN at load time."""

    uses_meta_device: bool = False

    def __init__(self, quant_config: AutoRoundRTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.parameter import ModelWeightParameter
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """BF16 → packed INT4 int32."""
        if getattr(layer, "_rtn_packed", False):
            return

        W = layer.weight.data.float()
        out_features, in_features = W.shape
        device = W.device
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        n_levels = 2 ** bits

        if group_size <= 0 or group_size > in_features:
            group_size = in_features
        n_groups = (in_features + group_size - 1) // group_size

        # Per-group symmetric RTN
        scales = torch.zeros(n_groups, out_features, dtype=torch.float32,
                             device=device)
        W_int = torch.zeros_like(W, dtype=torch.int32)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            group = W[:, start:end]
            max_val = group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_val / (n_levels // 2 - 1)
            scales[g] = scale.squeeze(-1)
            q = torch.clamp(
                torch.round(group / scale),
                -(n_levels // 2), n_levels // 2 - 1,
            ).to(torch.int32)
            W_int[:, start:end] = q

        # Pack into int32
        pack_factor = 32 // bits
        W_unsigned = (W_int + n_levels // 2).to(torch.int32)
        packed_cols = (in_features + pack_factor - 1) // pack_factor
        packed_w = torch.zeros(out_features, packed_cols,
                               dtype=torch.int32, device=device)
        for k in range(pack_factor):
            cols = torch.arange(k, in_features, pack_factor, device=device)
            packed_w[:, :len(cols)] |= (
                (W_unsigned[:, cols] & ((1 << bits) - 1)) << (k * bits)
            )

        # Replace weight
        layer.weight = nn.Parameter(packed_w, requires_grad=False)
        layer.register_buffer("_rtn_scales",
                              scales.T.contiguous().half(), persistent=False)
        layer._rtn_packed = True
        layer._rtn_in_features = in_features
        layer._rtn_bits = bits
        layer._rtn_group_size = group_size

        logger.info(
            "RTN: %s (%dx%d) → INT%d packed, %.1f%% of BF16",
            getattr(layer, "layer_name", "?"),
            out_features, in_features, bits,
            100.0 * packed_w.numel() * 4 / (out_features * in_features * 2),
        )

    @torch.compiler.disable
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_rtn_packed", False):
            W = self._decompress(layer).to(x.dtype)
            return F.linear(x, W, bias)
        return F.linear(x, layer.weight, bias)

    @torch.compiler.disable
    def _decompress(self, layer: nn.Module) -> torch.Tensor:
        packed = layer.weight.data
        out_features = packed.shape[0]
        in_features = layer._rtn_in_features
        bits = layer._rtn_bits
        group_size = layer._rtn_group_size
        scales = layer._rtn_scales.float()  # (out, n_groups)
        device = packed.device

        n_levels = 2 ** bits
        pack_factor = 32 // bits

        W_float = torch.zeros(out_features, in_features,
                              dtype=torch.float32, device=device)
        for k in range(pack_factor):
            cols = torch.arange(k, in_features, pack_factor, device=device)
            W_float[:, cols] = (
                ((packed[:, :len(cols)] >> (k * bits))
                 & ((1 << bits) - 1)).float() - n_levels // 2
            )

        # Apply per-group scales
        n_groups = scales.shape[1]
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            W_float[:, start:end] *= scales[:, g].unsqueeze(-1)

        return W_float
