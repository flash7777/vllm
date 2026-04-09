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
        """BF16 → GPTQ int32 packed format.

        For INT4: additionally repack to Marlin layout for fused kernel.
        For INT2/INT3: raw GPTQ format for mq_gemm kernels.
        """
        if getattr(layer, "_rtn_packed", False):
            return

        from vllm.multiquant.autoround.rtn_pack import rtn_pack_gptq

        W = layer.weight.data  # [N, K] (out_features, in_features)
        N, K = W.shape
        device = W.device

        qweight, scales, qzeros = rtn_pack_gptq(
            W.float(), self.bits, self.group_size)

        # Replace BF16 weight with packed tensors
        del layer.weight

        if self.bits == 4:
            # Marlin repack for fused INT4 dequant+GEMM
            self._setup_marlin(layer, qweight, scales, N, K, device)
        else:
            # INT2/INT3: raw GPTQ format for mq_gemm kernels
            layer.qweight = nn.Parameter(
                qweight.to(device), requires_grad=False)
            layer.scales = nn.Parameter(
                scales.to(device), requires_grad=False)
            layer.qzeros = nn.Parameter(
                qzeros.to(device), requires_grad=False)
            layer.g_idx = nn.Parameter(
                torch.empty(0, dtype=torch.int32, device=device),
                requires_grad=False)

        layer._rtn_packed = True
        layer._rtn_bits = self.bits
        layer._rtn_K = K
        layer._rtn_N = N

        logger.info(
            "RTN: %s (%dx%d) → INT%d %s, %.1f%% of BF16",
            getattr(layer, "layer_name", "?"), N, K, self.bits,
            "Marlin" if self.bits == 4 else "GPTQ",
            100.0 * qweight.numel() * 4 / (N * K * 2),
        )

    def _setup_marlin(self, layer: nn.Module,
                      qweight: torch.Tensor, scales: torch.Tensor,
                      N: int, K: int, device: torch.device) -> None:
        """Repack GPTQ weights to Marlin layout."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )

        qweight = qweight.to(device)
        scales = scales.to(device)

        # Empty g_idx (no activation order)
        g_idx = torch.empty(0, dtype=torch.int32, device=device)
        g_idx_sort = torch.empty(0, dtype=torch.int32, device=device)

        # Repack weights for Marlin kernel
        marlin_qweight = ops.gptq_marlin_repack(
            qweight, g_idx_sort, K, N, num_bits=4)

        # Permute scales for Marlin's MMA layout
        marlin_scales = marlin_permute_scales(
            scales, K, N, self.group_size)

        # Workspace for Marlin kernel
        workspace = marlin_make_workspace_new(device)

        layer.marlin_qweight = nn.Parameter(
            marlin_qweight, requires_grad=False)
        layer.marlin_scales = nn.Parameter(
            marlin_scales, requires_grad=False)
        layer.marlin_workspace = workspace  # buffer, not parameter
        layer.marlin_g_idx = g_idx
        layer.marlin_g_idx_sort = g_idx_sort

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

        # INT4: Marlin fused dequant+GEMM kernel
        return self._apply_marlin(layer, x, bias)

    def _apply_marlin(self, layer: nn.Module,
                      x: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
        """INT4 inference via Marlin kernel."""
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
        )

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.marlin_qweight,
            weight_scale=layer.marlin_scales,
            weight_zp=torch.empty(0, dtype=torch.int32,
                                  device=x.device),
            g_idx=layer.marlin_g_idx,
            g_idx_sort_indices=layer.marlin_g_idx_sort,
            workspace=layer.marlin_workspace,
            wtype=scalar_types.uint4b8,
            output_size_per_partition=layer._rtn_N,
            input_size_per_partition=layer._rtn_K,
            is_k_full=True,
            bias=bias,
        )
