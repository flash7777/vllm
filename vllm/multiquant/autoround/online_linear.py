# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN online linear method.

Loads BF16/FP8 weights, quantizes at load time using AutoRound iters=0,
stores as GPTQ-format INT4, uses Marlin kernel for inference.

Requires: pip install auto_round
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from vllm.multiquant.autoround.config import AutoRoundRTNConfig

logger = init_logger(__name__)


def _copy_missing_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    for attr in dir(src):
        if attr.startswith("_") or hasattr(dst, attr):
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except (AttributeError, RuntimeError):
            pass


class CopyNumelCounter(torch.overrides.TorchFunctionMode):
    def __init__(self):
        self.copied_numel = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func is torch.Tensor.copy_:
            self.copied_numel += args[0].numel()
        return result


class AutoRoundRTNLinearMethod(QuantizeMethodBase):
    """Online INT4 quantization via AutoRound opt_rtn (iters=0).

    Phase 1: Quantize at load time, decompress to BF16 for F.linear.
    Phase 2: Store as GPTQ format, use Marlin kernel.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: AutoRoundRTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                weight = ModelWeightParameter(
                    data=torch.empty_like(
                        layer.weight, device=layer._load_device
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            param = layer.weight
            copy_counter = CopyNumelCounter()
            with copy_counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            layer._loaded_numel += copy_counter.copied_numel

            if layer._loaded_numel >= layer.weight.numel():
                self.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True
            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """BF16/FP8 → INT4 via AutoRound RTN (iters=0).

        Phase 1: Simple per-group RTN quantize → dequantize to BF16.
        (AutoRound package call deferred to Phase 2 for Marlin format.)
        """
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(
                    layer.weight, device=layer._load_device
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)

        W = layer.weight.data.float()
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        n_levels = 2 ** bits

        # Simple per-group symmetric RTN quantization
        out_features, in_features = W.shape
        if group_size <= 0 or group_size > in_features:
            group_size = in_features

        n_groups = (in_features + group_size - 1) // group_size
        W_q = torch.zeros_like(W)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            group = W[:, start:end]

            # Symmetric quantization: scale = max(|w|) / (n_levels/2 - 1)
            max_val = group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_val / (n_levels // 2 - 1)

            # Quantize (round-to-nearest)
            q = torch.clamp(
                torch.round(group / scale),
                -(n_levels // 2), n_levels // 2 - 1
            )

            # Dequantize
            W_q[:, start:end] = q * scale

        from vllm.model_executor.model_loader.utils import replace_parameter
        replace_parameter(layer, "weight", W_q.to(layer.orig_dtype).data)

        layer._already_called_process_weights_after_loading = True
        logger.info(
            "AutoRound RTN: quantized %s (%dx%d) to %d-bit, group=%d",
            getattr(layer, "layer_name", "?"),
            out_features, in_features, bits, group_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Phase 1: Standard F.linear with dequantized weights."""
        return torch.nn.functional.linear(x, layer.weight, bias)
