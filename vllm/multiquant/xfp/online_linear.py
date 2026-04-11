# SPDX-License-Identifier: Apache-2.0
"""XFP online linear method — BF16 → learned codebook + packed indices.

Quant-on-load pipeline for Linear layers:
    create_weights                — allocate BF16 ModelWeightParameter
    process_weights_after_loading — xfp_pack(W, bits), register packed +
                                    codebook, record stats, free BF16
    apply                         — fused depack + LUT + GEMM via
                                    _load_xfp_gemm(bits).xfp_gemm

Self-contained — no Archer imports. Stats are recorded in the central
MultiQuantPolicyRegistry for end-of-load summary and future auto-size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


# ─── Custom op registration for torch.compile compatibility ────────────
#
# Dynamo cannot trace pybind11 C/C++ extensions (including torch's cpp_extension
# JIT-loaded modules). We wrap the kernel call in a torch custom op with a
# fake implementation so torch.compile's graph capture sees a well-defined
# operator with known output shape, while eager execution calls the real
# CUDA kernel. Pattern mirrored from ArcherOnlineLinearMethod's custom op
# registration in weight_quant/online_linear.py:143-155.

def _xfp_apply_impl(
    x: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    K: int,
    N_out: int,
) -> torch.Tensor:
    """Real impl: eager fused xfp_gemm. Output dtype follows input."""
    from vllm.multiquant.xfp import xfp_kernel as _xk
    kernel = _xk._xfp_gemm_kernel
    if kernel is None:
        raise RuntimeError(
            "xfp custom op called before xfp_gemm kernel was loaded. "
            "process_weights_after_loading should have triggered JIT."
        )
    out_dtype = x.dtype  # typically bf16; we preserve it on the output
    x_fp16 = x.to(torch.float16).contiguous()
    C = torch.zeros(x_fp16.shape[0], N_out,
                    dtype=torch.float16, device=x.device)
    kernel.xfp_gemm(x_fp16, packed, codebook, C, int(bits), int(K))
    return C.to(out_dtype)


def _xfp_apply_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    K: int,
    N_out: int,
) -> torch.Tensor:
    """Fake impl for torch.compile graph tracing. Matches real dtype."""
    return torch.empty(
        x.shape[0], N_out, dtype=x.dtype, device=x.device
    )


try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="xfp_apply",
        op_func=_xfp_apply_impl,
        fake_impl=_xfp_apply_fake,
    )
    _xfp_op = torch.ops.vllm.xfp_apply
    logger.info("XFP custom op registered (torch.compile safe)")
except Exception as e:
    logger.warning(
        "XFP custom op registration failed: %s — "
        "torch.compile with --weight-dtype-*=xfp* may not work", e
    )
    _xfp_op = _xfp_apply_impl


class XFPLinearMethod(QuantizeMethodBase):
    """Learned-codebook quantization (XFP2/XFP3/XFP4) at model load time."""

    uses_meta_device: bool = False

    def __init__(
        self,
        quant_config: "QuantizationConfig",
        dtype: str = "xfp4",
    ):
        if dtype not in ("xfp2", "xfp3", "xfp4"):
            raise ValueError(
                f"XFPLinearMethod: unsupported dtype '{dtype}', "
                f"supported: xfp2, xfp3, xfp4 (v1)"
            )
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS[dtype]  # 2, 3, or 4

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
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
        layer._xfp_input_size = input_size
        layer._xfp_output_size = output_size
        layer._xfp_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """BF16 weight → xfp_pack → (packed uint32, codebook fp16).

        Records XFPPackStats on layer and into the central registry.
        Drops the BF16 weight after packing. Also eagerly compiles the
        xfp_gemm kernel so the forward-pass `apply()` is torch.compile
        friendly (no lazy JIT inside a graphed region).
        """
        if getattr(layer, "_xfp_packed_done", False):
            return

        from vllm.multiquant.xfp.xfp_pack import xfp_pack
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm

        # Eagerly JIT-compile the kernel once, outside any torch.compile
        # graph. Subsequent .apply() calls will find the cached module.
        _load_xfp_gemm(self.bits)

        W = layer.weight.data  # [N_out, K]
        device = W.device

        # Pack only the chosen bit width. Candidate-scoring via
        # also_score_widths is a v3 feature (auto-size selection) — for
        # v1 it's disabled to keep load time proportional to one Lloyd
        # pass per layer instead of three.
        packed, codebook, stats = xfp_pack(
            W.float(),
            bits=self.bits,
            also_score_widths=(),
        )

        layer.xfp_packed = nn.Parameter(
            packed.to(device), requires_grad=False
        )
        layer.xfp_codebook = nn.Parameter(
            codebook.to(device), requires_grad=False
        )
        layer._xfp_bits = self.bits
        layer._xfp_K = int(W.shape[1])
        layer._xfp_N = int(W.shape[0])
        layer._xfp_stats = stats
        layer._xfp_packed_done = True

        # Accumulate into the central registry for end-of-load summary
        from vllm.multiquant.policy import (
            MultiQuantPolicyRegistry,
            classify_layer,
        )
        reg = MultiQuantPolicyRegistry.get_active()
        if reg is not None:
            layer_prefix = getattr(layer, "layer_name", "") or \
                getattr(layer, "prefix", "") or ""
            layer_type = classify_layer(layer_prefix) or "other"
            reg.record_stats(layer_type, stats)

        logger.info(
            "XFP %s [%dx%d] -> %s | mse=%.3g cos=%.3f | "
            "3sigma=%.1f%% | recommend=xfp%d (gap=%.2fx)",
            getattr(layer, "layer_name", "?"),
            stats.shape[0], stats.shape[1],
            self.dtype, stats.mse, stats.cos_sim,
            100.0 * stats.outlier_ratio_k3,
            stats.recommended_bits, stats.recommended_gap,
        )

        # Free the BF16 weight
        try:
            del layer.weight
        except AttributeError:
            pass

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer._xfp_N,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        # Dispatch through the registered custom op so torch.compile
        # sees a traceable operator instead of a pybind11 extension call.
        C = _xfp_op(
            reshaped_x,
            layer.xfp_packed,
            layer.xfp_codebook,
            int(self.bits),
            int(layer._xfp_K),
            int(layer._xfp_N),
        )
        if bias is not None:
            C = C + bias.to(C.dtype)
        return C.reshape(out_shape)
