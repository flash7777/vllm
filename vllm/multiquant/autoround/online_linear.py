# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN online linear method — BF16 → GPTQ format at load time.

Loads BF16 normally, packs to GPTQ int32 in process_weights_after_loading,
then delegates inference to MQSub4LinearMethod (INT2/INT3) or Marlin (INT4).

Same storage format as pre-quantized AutoRound/GPTQ models → same kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.multiquant.autoround.config import AutoRoundRTNConfig

logger = init_logger(__name__)


class AutoRoundRTNLinearMethod(QuantizeMethodBase):
    """BF16 → GPTQ-format INT2/INT3/INT4 at load time.

    After packing, delegates apply() to MQSub4LinearMethod (INT2/3)
    or uses the packed format directly for INT4 (Marlin-compatible).
    """

    uses_meta_device: bool = False

    def __init__(self, quant_config: "AutoRoundRTNConfig",
                 bits: int = 4, group_size: int = 128):
        self.quant_config = quant_config
        self.bits = bits
        self.group_size = group_size

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
        """BF16 → GPTQ int32 packed format."""
        if getattr(layer, "_rtn_packed", False):
            return

        from vllm.multiquant.autoround.rtn_pack import rtn_pack_gptq

        W = layer.weight.data  # [N, K] (out_features, in_features)
        N, K = W.shape
        device = W.device

        qweight, scales, qzeros = rtn_pack_gptq(
            W.float(), self.bits, self.group_size)

        # Replace BF16 weight with GPTQ tensors
        del layer.weight
        layer.qweight = nn.Parameter(qweight.to(device), requires_grad=False)
        layer.scales = nn.Parameter(scales.to(device), requires_grad=False)
        layer.qzeros = nn.Parameter(qzeros.to(device), requires_grad=False)
        layer.g_idx = nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=device),
            requires_grad=False)

        layer._rtn_packed = True
        layer._rtn_bits = self.bits

        logger.info(
            "RTN: %s (%dx%d) → INT%d GPTQ, %.1f%% of BF16",
            getattr(layer, "layer_name", "?"), N, K, self.bits,
            100.0 * qweight.numel() * 4 / (N * K * 2),
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to MQSub4LinearMethod for INT2/INT3 fused GEMM."""
        if not getattr(layer, "_rtn_packed", False):
            return torch.nn.functional.linear(x, layer.weight, bias)

        bits = layer._rtn_bits
        if bits in (2, 3):
            # Use MQSub4LinearMethod's apply (fused GEMM kernels)
            from vllm.multiquant.weight_quant.mq_sub4_linear import (
                _load_mq_gemm,
            )
            kernel = _load_mq_gemm(bits)
            if kernel is None:
                raise RuntimeError(
                    f"mq_gemm_int{bits} kernel not available")

            out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
            reshaped_x = x.reshape(-1, x.shape[-1]).to(torch.float16)
            M = reshaped_x.shape[0]
            N = layer.qweight.shape[1]

            C = torch.zeros(M, N, dtype=torch.float16, device=x.device)
            if bits == 2:
                kernel.mq_gemm_int2(
                    reshaped_x, layer.qweight, layer.scales,
                    layer.qzeros, C, self.group_size)
            else:
                K = reshaped_x.shape[1]
                kernel.mq_gemm_int3(
                    reshaped_x, layer.qweight, layer.scales,
                    layer.qzeros, C, K, self.group_size)

            if bias is not None:
                C.add_(bias)
            return C.reshape(out_shape)

        # INT4: decompress + F.linear (TODO: Marlin integration)
        W = self._decompress_int4(layer).to(x.dtype)
        return torch.nn.functional.linear(x, W, bias)

    @torch.compiler.disable
    def _decompress_int4(self, layer: nn.Module) -> torch.Tensor:
        """Fallback INT4 decompress for non-Marlin path."""
        qw = layer.qweight  # [K/8, N]
        scales = layer.scales  # [n_groups, N]
        K_packed, N = qw.shape
        K = K_packed * 8
        device = qw.device

        W = torch.zeros(K, N, dtype=torch.float32, device=device)
        for i in range(8):
            vals = ((qw >> (i * 4)) & 0xF).float()
            W[torch.arange(K_packed, device=device) * 8 + i] = vals

        n_groups = scales.shape[0]
        gs = self.group_size
        gi = torch.arange(K, device=device) // gs
        gi = gi.clamp(max=n_groups - 1)
        W = scales[gi].float() * (W - 8.0)  # zp=8 for INT4

        return W.T  # [N, K] for F.linear
