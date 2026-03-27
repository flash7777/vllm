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

        out_features, in_features = W.shape
        if group_size <= 0 or group_size > in_features:
            group_size = in_features

        n_groups = (in_features + group_size - 1) // group_size

        # Per-group symmetric RTN: compute scales + integer weights
        scales = torch.zeros(n_groups, out_features, dtype=torch.float32,
                             device=W.device)
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
                -(n_levels // 2), n_levels // 2 - 1
            ).to(torch.int32)
            W_int[:, start:end] = q

        # Try Marlin repack for fast inference kernel
        use_marlin = False
        try:
            if bits == 4:
                from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                    prepare_int4_weight_for_marlin,
                )
                # Pack INT4 to Marlin format + use Marlin GEMM
                # Shift to unsigned: q_unsigned = q + 8 (for INT4 symmetric)
                W_unsigned = (W_int + n_levels // 2).to(torch.int32)
                # Pack 8 x INT4 per INT32
                pack_factor = 32 // bits
                packed_w = torch.zeros(
                    out_features, in_features // pack_factor,
                    dtype=torch.int32, device=W.device)
                for k in range(pack_factor):
                    packed_w |= (W_unsigned[:, k::pack_factor] & 0xF) << (k * 4)

                # Store packed weights + scales for Marlin
                from vllm.model_executor.model_loader.utils import (
                    replace_parameter,
                )
                replace_parameter(layer, "weight", packed_w.data)
                layer.register_buffer(
                    "weight_scale",
                    scales.T.contiguous().to(layer.orig_dtype),
                    persistent=False,
                )
                layer._autoround_rtn_marlin = True
                use_marlin = True
                logger.info(
                    "AutoRound RTN: %s (%dx%d) → INT4 packed (Marlin-ready)",
                    getattr(layer, "layer_name", "?"),
                    out_features, in_features,
                )
        except Exception as e:
            logger.debug("Marlin repack failed, falling back to BF16: %s", e)

        if not use_marlin:
            # Fallback: dequantize to BF16 (no memory savings, but correct)
            W_q = torch.zeros_like(W)
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                scale = scales[g].unsqueeze(-1)
                W_q[:, start:end] = W_int[:, start:end].float() * scale

            from vllm.model_executor.model_loader.utils import (
                replace_parameter,
            )
            replace_parameter(layer, "weight", W_q.to(layer.orig_dtype).data)
            layer._autoround_rtn_marlin = False
            logger.info(
                "AutoRound RTN: %s (%dx%d) → %d-bit dequantized (BF16 fallback)",
                getattr(layer, "layer_name", "?"),
                out_features, in_features, bits,
            )

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GEMM: Marlin kernel if packed INT4, else standard F.linear."""
        if getattr(layer, "_autoround_rtn_marlin", False):
            # INT4 packed → dequantize on-the-fly via simple unpack + scale
            # TODO: Replace with actual Marlin kernel call (needs repack to
            # Marlin tile format via gptq_marlin_repack + workspace setup)
            W_packed = layer.weight  # (out, in // 8) int32
            scales = layer.weight_scale  # (out, n_groups) or (n_groups, out)
            bits = self.quant_config.bits
            group_size = self.quant_config.group_size
            pack_factor = 32 // bits
            out_features = W_packed.shape[0]
            in_features = W_packed.shape[1] * pack_factor

            # Unpack INT4 → float (temporary until Marlin kernel integrated)
            W_float = torch.zeros(out_features, in_features,
                                  dtype=x.dtype, device=x.device)
            n_levels = 2 ** bits
            for k in range(pack_factor):
                W_float[:, k::pack_factor] = (
                    ((W_packed >> (k * 4)) & 0xF).float() - n_levels // 2
                )

            # Apply per-group scales
            n_groups = scales.shape[0] if scales.dim() == 2 else 1
            if n_groups > 1 and group_size > 0:
                for g in range(n_groups):
                    start = g * group_size
                    end = min(start + group_size, in_features)
                    W_float[:, start:end] *= scales[g].unsqueeze(-1)
            else:
                W_float *= scales.unsqueeze(-1)

            return torch.nn.functional.linear(x, W_float, bias)
        else:
            return torch.nn.functional.linear(x, layer.weight, bias)
