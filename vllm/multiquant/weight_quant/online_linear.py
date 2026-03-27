# SPDX-License-Identifier: Apache-2.0
"""Archer online linear method — BF16/FP8 → MultiQuant packed weights.

Per-layer quantization: each layer is quantized as soon as all its shards
arrive, then BF16 is replaced by packed uint8. Peak memory = 1 layer BF16.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from vllm.multiquant.weight_quant.config import ArcherConfig

logger = init_logger(__name__)



class ArcherOnlineLinearMethod(QuantizeMethodBase):
    """BF16/FP8 → MultiQuant packed, quantized per-layer on arrival.

    uses_meta_device=True: weight starts on meta (0 bytes).
    patched_weight_loader: materializes on first shard, quantizes when done.
    Peak memory = 1 BF16 layer + accumulated packed layers.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: ArcherConfig):
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self.method = quant_config.method

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
        out_size = sum(output_partition_sizes)
        orig_loader = extra_weight_attrs.get("weight_loader")
        archer_method = self  # closure reference

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # First shard: materialize from meta → real device
            if not hasattr(layer, "_archer_loaded"):
                layer._archer_loaded = 0
                real = ModelWeightParameter(
                    data=torch.empty(
                        out_size, input_size_per_partition,
                        dtype=params_dtype,
                        device=layer._archer_device,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                layer.register_parameter("weight", real)
                del layer._archer_device

            # Load this shard via original loader
            param = layer.weight
            if orig_loader is not None:
                orig_loader(param, loaded_weight, *args, **kwargs)
            else:
                param.data.copy_(loaded_weight)
            # Track by loaded_weight size (not copy_ ops — those vary)
            layer._archer_loaded += loaded_weight.numel()

            # All shards for this layer arrived? → quantize + free BF16
            if layer._archer_loaded >= param.data.numel():
                archer_method._quantize_layer(layer)
                layer._already_called_process_weights_after_loading = True

        # Create on meta device — zero memory
        weight = ModelWeightParameter(
            data=torch.empty(
                out_size, input_size_per_partition,
                device="meta", dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._archer_device = torch.get_default_device()
        layer.register_parameter("weight", weight)

    def _quantize_layer(self, layer: nn.Module) -> None:
        """Compress one layer: BF16 → packed uint8."""
        W = layer.weight.data.float()
        out_features, in_features = W.shape
        device = W.device

        cache_dtype = f"{self.method}{self.bits}"
        from vllm.multiquant.registry import get_kv_quantizer_config
        mq_config = get_kv_quantizer_config(cache_dtype, in_features)
        mse_bits = mq_config.mse_bits
        packed_size = mq_config.key_packed_size

        from vllm.multiquant.shared.centroids import get_centroids
        centroids = get_centroids(in_features, mse_bits).to(device)

        seed = self.quant_config.seed
        if self.method == "rq":
            from vllm.multiquant.rotorquant.quantizer import generate_rotors
            rotation = generate_rotors(in_features, seed=seed).to(device)
        else:
            from vllm.multiquant.turboquant.quantizer import (
                generate_rotation_matrix,
            )
            rotation = generate_rotation_matrix(
                in_features, seed=seed
            ).to(device)

        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        S = generate_qjl_matrix(in_features, seed=seed + 1).to(device)

        # Per-row quantization
        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)

        if self.method == "rq":
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            mv = embed_vectors_as_multivectors(W_unit)
            mv_rot = rotor_sandwich(rotation, mv)
            for gi in [1, 2, 3, 7]:
                comp = mv_rot[..., gi]
                diffs = comp.unsqueeze(-1) - centroids
                idx = diffs.abs().argmin(dim=-1)
                mv_rot[..., gi] = centroids[idx]
            rotor_rev = reverse(rotation)
            mv_recon = rotor_sandwich(rotor_rev, mv_rot)
            W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
            # Re-compute indices for packing
            mv2 = embed_vectors_as_multivectors(W_unit)
            mv2_rot = rotor_sandwich(rotation, mv2)
            coords = [mv2_rot[..., gi] for gi in [1, 2, 3, 7]]
            flat_coords = torch.cat(
                [c.reshape(out_features, -1) for c in coords], dim=-1)
            diffs = flat_coords.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)
        else:
            rotated = W_unit @ rotation.T
            diffs = rotated.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)
            quantized = centroids[mse_indices]
            W_mse = quantized @ rotation

        residual = W_unit - W_mse
        res_norms = residual.norm(dim=-1)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        packed_W = pack_vectors_batched(
            mse_indices, signs, row_norms, res_norms,
            in_features, mse_bits,
        )

        # Zero the BF16 data first — releases actual GPU memory even if
        # params_dict in model.load_weights() still holds a reference.
        # Then resize to 0 to minimize the dangling allocation.
        old_weight = layer.weight
        old_weight.data = torch.empty(0, device=device, dtype=old_weight.dtype)

        # Set new packed weight
        layer.weight = nn.Parameter(packed_W, requires_grad=False)

        # Force GC to release the old allocation
        del old_weight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Decompression metadata
        layer.register_buffer("_archer_rotation", rotation, persistent=False)
        layer.register_buffer("_archer_S", S, persistent=False)
        layer.register_buffer("_archer_centroids", centroids, persistent=False)
        layer._archer_in_features = in_features
        layer._archer_mse_bits = mse_bits
        layer._archer_method = self.method
        layer._archer_packed = True

        logger.info(
            "Archer: (%dx%d) → (%dx%d) uint8, %.1f%% of BF16",
            out_features, in_features,
            out_features, packed_size,
            100.0 * packed_size / (in_features * 2),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Fallback — normally patched_weight_loader already quantized."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if getattr(layer, "_archer_packed", False):
            return
        if layer.weight.device != torch.device("meta"):
            self._quantize_layer(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_archer_packed", False):
            W = self._decompress(layer).to(x.dtype)
            return F.linear(x, W, bias)
        return F.linear(x, layer.weight, bias)

    def _decompress(self, layer: nn.Module) -> torch.Tensor:
        packed = layer.weight.data
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
        cpb = 8 // mse_bits

        idx = torch.zeros(out_features, in_features,
                          dtype=torch.long, device=device)
        for b in range(mse_bytes):
            bv = packed[:, b].long()
            for k in range(cpb):
                j = b * cpb + k
                if j >= in_features:
                    break
                idx[:, j] = (bv >> (k * mse_bits)) & mask

        signs = torch.zeros(out_features, in_features,
                            dtype=torch.float32, device=device)
        for b in range(qjl_bytes):
            bv = packed[:, mse_bytes + b].long()
            for k in range(8):
                j = b * 8 + k
                if j >= in_features:
                    break
                signs[:, j] = torch.where(
                    ((bv >> k) & 1).bool(),
                    torch.ones(out_features, device=device),
                    -torch.ones(out_features, device=device),
                )

        no = mse_bytes + qjl_bytes
        row_norms = packed[:, no:no + 2].contiguous().view(
            torch.float16).float().squeeze(-1)
        res_norms = packed[:, no + 2:no + 4].contiguous().view(
            torch.float16).float().squeeze(-1)

        c_vals = centroids[idx]
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
        W_qjl = correction * res_norms.unsqueeze(-1) * (signs @ S)
        return row_norms.unsqueeze(-1) * (W_mse + W_qjl)
