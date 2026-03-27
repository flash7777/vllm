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

        # 5. Compute MSE indices for packing (re-use rotation result)
        if self.method == "rq":
            # For RQ, re-compute indices from rotated multivector
            mv2 = embed_vectors_as_multivectors(W_unit)
            mv2_rot = rotor_sandwich(rotation, mv2)
            # Collect quantized coords for all non-zero grades
            all_coords = []
            for gi in [1, 2, 3, 7]:
                all_coords.append(mv2_rot[..., gi])
            flat_coords = torch.cat(
                [c.reshape(out_features, -1) for c in all_coords], dim=-1
            )
            diffs = flat_coords.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)
        else:
            rotated = W_unit @ rotation.T
            diffs = rotated.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)

        # QJL residual
        residual = W_unit - W_mse
        res_norms = residual.norm(dim=-1)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        # 6. Pack into uint8 — REAL compression, stays packed in VRAM
        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        packed_W = pack_vectors_batched(
            mse_indices, signs, row_norms, res_norms,
            in_features, mse_bits,
        )

        from vllm.model_executor.model_loader.utils import replace_parameter
        # Store packed uint8 weights (REAL memory savings!)
        replace_parameter(layer, "weight",
                          torch.nn.Parameter(packed_W, requires_grad=False))

        # Store decompression metadata
        layer.register_buffer(
            "_archer_rotation", rotation, persistent=False)
        layer.register_buffer(
            "_archer_S", S, persistent=False)
        layer.register_buffer(
            "_archer_centroids", centroids, persistent=False)
        layer._archer_in_features = in_features
        layer._archer_mse_bits = mse_bits
        layer._archer_method = self.method
        layer._archer_packed = True

        layer._already_called_process_weights_after_loading = True
        logger.info(
            "Archer: packed %s (%dx%d) → (%dx%d) uint8, %.1f%% of BF16",
            getattr(layer, "layer_name", "?"),
            out_features, in_features,
            out_features, packed_size,
            100.0 * packed_size / (in_features * 2),
        )

    def _decompress_weights(self, layer: torch.nn.Module) -> torch.Tensor:
        """Decompress packed uint8 → BF16 weight matrix on-the-fly."""
        packed = layer.weight.data  # (out, packed_size) uint8
        out_features = packed.shape[0]
        in_features = layer._archer_in_features
        mse_bits = layer._archer_mse_bits
        device = packed.device

        rotation = layer._archer_rotation
        S = layer._archer_S
        centroids = layer._archer_centroids

        mse_bytes = math.ceil(in_features * mse_bits / 8)
        qjl_bytes = math.ceil(in_features / 8)
        mask = (1 << mse_bits) - 1
        coords_per_byte = 8 // mse_bits

        # Unpack MSE indices
        idx_all = torch.zeros(out_features, in_features,
                              dtype=torch.long, device=device)
        for b in range(mse_bytes):
            bv = packed[:, b].long()
            for k in range(coords_per_byte):
                j = b * coords_per_byte + k
                if j >= in_features:
                    break
                idx_all[:, j] = (bv >> (k * mse_bits)) & mask

        # Unpack signs
        signs_all = torch.zeros(out_features, in_features,
                                dtype=torch.float32, device=device)
        for b in range(qjl_bytes):
            bv = packed[:, mse_bytes + b].long()
            for k in range(8):
                j = b * 8 + k
                if j >= in_features:
                    break
                signs_all[:, j] = torch.where(
                    ((bv >> k) & 1).bool(),
                    torch.ones(out_features, device=device),
                    -torch.ones(out_features, device=device),
                )

        # Unpack norms
        no = mse_bytes + qjl_bytes
        row_norms = packed[:, no:no + 2].contiguous().view(
            torch.float16).float().squeeze(-1)
        res_norms = packed[:, no + 2:no + 4].contiguous().view(
            torch.float16).float().squeeze(-1)

        # Reconstruct
        c_vals = centroids[idx_all]

        if layer._archer_method == "rq":
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            mv_q = embed_vectors_as_multivectors(c_vals)
            rotor_rev = reverse(rotation)
            mv_recon = rotor_sandwich(rotor_rev, mv_q)
            W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
        else:
            W_mse = c_vals @ rotation

        correction = math.sqrt(math.pi / 2) / in_features
        W_qjl = correction * res_norms.unsqueeze(-1) * (signs_all @ S)
        W_recon = row_norms.unsqueeze(-1) * (W_mse + W_qjl)

        return W_recon

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decompress packed weights on-the-fly → GEMM.

        Weights stored as packed uint8 (real compression).
        Decompressed to BF16/FP16 per forward pass.
        TODO: Archer CUDA kernel for fused decompress+GEMM.
        """
        if getattr(layer, '_archer_packed', False):
            W = self._decompress_weights(layer).to(x.dtype)
            return F.linear(x, W, bias)
        return F.linear(x, layer.weight, bias)
