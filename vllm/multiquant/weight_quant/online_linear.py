# SPDX-License-Identifier: Apache-2.0
"""Archer online linear method — BF16/FP8 → MultiQuant compressed weights.

Phase 1: Python decompress + standard GEMM (correct, slow).
Phase 2: Archer fused kernel (decompress+GEMM in one kernel).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from vllm.multiquant.weight_quant.config import ArcherConfig

logger = init_logger(__name__)


def _copy_missing_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy custom attributes from src to dst if not present."""
    for attr in dir(src):
        if attr.startswith("_") or hasattr(dst, attr):
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except (AttributeError, RuntimeError):
            pass


class CopyNumelCounter(torch.overrides.TorchFunctionMode):
    """Track total elements copied via torch.Tensor.copy_."""

    def __init__(self):
        self.copied_numel = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func is torch.Tensor.copy_:
            self.copied_numel += args[0].numel()
        return result


class ArcherOnlineLinearMethod(QuantizeMethodBase):
    """Online weight quantization using MultiQuant (TurboQuant/RotorQuant).

    Loads BF16/FP8 weights, quantizes per-row at load time to 2-4 bits.
    Decompresses on-the-fly during inference.

    Phase 1: Python decompress + F.linear (correct reference).
    Phase 2: Archer fused CUDA kernel.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: ArcherConfig):
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self.method = quant_config.method  # "rq" or "tq"

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
        layer._archer_config = self.quant_config

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                # Materialize weight from meta → actual device
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
        """BF16/FP8 weights → MultiQuant compressed uint8."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Handle dummy weights
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

        W = layer.weight.data.float()  # (out_features, in_features)
        out_features, in_features = W.shape

        # Per-row quantization: each row is an in_features-dim vector
        # 1. Compute row norms
        row_norms = W.norm(dim=-1)  # (out_features,)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)

        # 2. Get quantizer config + centroids
        cache_dtype = f"{self.method}{self.bits}"
        from vllm.multiquant.registry import get_kv_quantizer_config
        mq_config = get_kv_quantizer_config(cache_dtype, in_features)
        mse_bits = mq_config.mse_bits

        from vllm.multiquant.shared.centroids import get_centroids
        centroids = get_centroids(in_features, mse_bits).to(W.device)

        # 3. Generate rotation (per-layer seed)
        seed = self.quant_config.seed
        if self.method == "rq":
            from vllm.multiquant.rotorquant.quantizer import generate_rotors
            rotation = generate_rotors(in_features, seed=seed).to(W.device)
        else:
            from vllm.multiquant.turboquant.quantizer import (
                generate_rotation_matrix,
            )
            rotation = generate_rotation_matrix(
                in_features, seed=seed
            ).to(W.device)

        # 4. Quantize each row
        packed_size = mq_config.key_packed_size
        packed_W = torch.zeros(
            out_features, packed_size, dtype=torch.uint8, device=W.device
        )

        # Generate QJL matrix (shared between TQ and RQ)
        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        S = generate_qjl_matrix(in_features, seed=seed + 1).to(W.device)

        # Vectorized quantization for all rows at once
        if self.method == "rq":
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse,
                rotor_sandwich,
            )
            # Rotor sandwich on all rows simultaneously
            mv = embed_vectors_as_multivectors(W_unit)
            mv_rot = rotor_sandwich(rotation, mv)
            # Quantize non-zero grades
            for grade_idx in [1, 2, 3, 7]:
                comp = mv_rot[..., grade_idx]
                diffs = comp.unsqueeze(-1) - centroids
                indices = diffs.abs().argmin(dim=-1)
                mv_rot[..., grade_idx] = centroids[indices]
            # Inverse rotation
            rotor_rev = reverse(rotation)
            mv_recon = rotor_sandwich(rotor_rev, mv_rot)
            W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
        else:
            # TurboQuant: dense matrix rotation
            rotated = W_unit @ rotation.T
            diffs = rotated.unsqueeze(-1) - centroids
            indices = diffs.abs().argmin(dim=-1)
            quantized = centroids[indices]
            W_mse = quantized @ rotation

        # QJL residual correction
        residual = W_unit - W_mse
        res_norms = residual.norm(dim=-1)  # (out_features,)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        correction_scale = math.sqrt(math.pi / 2) / in_features
        W_qjl = correction_scale * res_norms.unsqueeze(-1) * (signs @ S)
        W_recon = row_norms.unsqueeze(-1) * (W_mse + W_qjl)

        # 5. Pack into uint8 (same format as KV-cache pack)
        # For now, store decompressed BF16 — full packing in Phase 2
        # This gives correct results but no memory savings yet
        W_compressed = W_recon.to(layer.orig_dtype)

        from vllm.model_executor.model_loader.utils import replace_parameter
        replace_parameter(layer, "weight", W_compressed.data)

        # Store metadata for potential future Archer kernel
        layer.register_buffer(
            "_archer_row_norms", row_norms.half(), persistent=False
        )

        layer._already_called_process_weights_after_loading = True
        logger.info(
            "Archer: quantized %s (%dx%d) with %s%d",
            getattr(layer, "layer_name", "?"),
            out_features, in_features,
            self.method, self.bits,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: standard F.linear with decompressed weights.

        Phase 1: Weights already decompressed in process_weights_after_loading.
        Phase 2: Archer kernel will decompress on-the-fly.
        """
        return F.linear(x, layer.weight, bias)
