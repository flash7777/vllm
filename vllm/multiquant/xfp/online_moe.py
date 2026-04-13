# SPDX-License-Identifier: Apache-2.0
"""XFP online MoE method — BF16 → per-expert learned codebook at load time.

v3: fused CUDA MoE kernel. Single kernel launch per GEMM handles all experts
via sorted_token_ids / expert_ids (Marlin pattern). No Python expert loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


class XFPMoEMethod(FusedMoEMethodBase):
    """Learned-codebook quant-on-load for FusedMoE layers.

    Per-expert Lloyd codebook + word-aligned packed indices.
    Apply uses fused CUDA kernel — one launch per GEMM, all experts.
    """

    def __init__(
        self,
        quant_config: "QuantizationConfig",
        dtype: str = "xfp4",
        moe_config: "FusedMoEConfig | None" = None,
    ):
        if dtype not in ("xfp2", "xfp3", "xfp4"):
            raise ValueError(
                f"XFPMoEMethod: unsupported dtype '{dtype}', "
                f"supported: xfp2, xfp3, xfp4"
            )
        if moe_config is not None:
            super().__init__(moe_config)
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS[dtype]

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        self._unquant = UnquantizedFusedMoEMethod(self.moe)
        self._unquant.create_weights(
            layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype,
            **extra_weight_attrs,
        )
        layer._xfp_moe_hidden = hidden_size
        layer._xfp_moe_intermediate = intermediate_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_xfp_moe_packed", False):
            return

        from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
        from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm

        bits = self.bits
        device = layer.w13_weight.device
        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]
        E = int(w13.shape[0])

        _load_xfp_gemm(bits)
        _load_xfp_moe_gemm()

        # Save shape metadata before freeing weights
        K13 = int(w13.shape[2])
        N13 = int(w13.shape[1])
        K2 = int(w2.shape[2])
        N2 = int(w2.shape[1])

        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        reg = MultiQuantPolicyRegistry.get_active()

        # MoE experts have homogeneous distributions — fewer Lloyd iters
        # suffice. Default 5 (vs 20 for attention). Override via XFP_MOE_LLOYD_ITERS.
        import os
        moe_lloyd_iters = int(os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))

        def _batched_pack_and_repack(W_stack: torch.Tensor):
            """W_stack: [E, N, K] -> flat packed [E*flat], flat codebook [E*N*lut].

            Memory-conscious: deletes intermediates immediately.
            """
            E_ = int(W_stack.shape[0])
            N_ = int(W_stack.shape[1])
            K_ = int(W_stack.shape[2])
            W_flat = W_stack.reshape(E_ * N_, K_).float()
            del W_stack
            packed_flat, codebook_flat, _, _, stats = xfp_pack(
                W_flat, bits=bits, outlier_sigma=None,
                lloyd_iters=moe_lloyd_iters,
            )
            del W_flat  # free float32 copy
            k_packed = packed_flat.shape[0]
            packed = packed_flat.view(k_packed, E_, N_).permute(1, 0, 2).contiguous()
            del packed_flat
            lut_size = 1 << bits
            codebook = codebook_flat.view(E_, N_, lut_size)

            # Repack each expert and concatenate flat
            repacked_list = []
            for e in range(E_):
                repacked_list.append(xfp_repack(packed[e]))
            del packed
            flat_per_expert = repacked_list[0].numel()
            all_repacked = torch.cat(repacked_list, dim=0)
            del repacked_list
            all_codebook = codebook.reshape(-1)
            del codebook

            return all_repacked, all_codebook, flat_per_expert, stats

        # Pack w13, then free BF16 w13 before packing w2
        p13, cb13, fpe13, stats13 = _batched_pack_and_repack(w13.clone())
        del w13
        layer.w13_weight.data = torch.empty(0)  # free BF16

        p2, cb2, fpe2, stats2 = _batched_pack_and_repack(w2.clone())
        del w2
        layer.w2_weight.data = torch.empty(0)  # free BF16

        if reg is not None:
            reg.record_stats("routed_expert", stats13)
            reg.record_stats("routed_expert", stats2)

        # Store as flat tensors for fused kernel
        layer.w13_xfp_packed = nn.Parameter(p13.to(device), requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(cb13.to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(p2.to(device), requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(cb2.to(device), requires_grad=False)

        layer._xfp_moe_bits = bits
        layer._xfp_moe_dtype = self.dtype
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13
        layer._xfp_moe_K2 = K2
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = E
        layer._xfp_moe_fpe13 = fpe13
        layer._xfp_moe_fpe2 = fpe2
        layer._xfp_moe_packed = True

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

        logger.info(
            "XFP MoE: %d experts w13[%dx%d] + w2[%dx%d] -> %s "
            "(fused, fpe=%d/%d, lloyd=%d)",
            E, layer._xfp_moe_N13, layer._xfp_moe_K13,
            layer._xfp_moe_N2, layer._xfp_moe_K2, self.dtype,
            fpe13, fpe2, moe_lloyd_iters,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not getattr(layer, "_xfp_moe_packed", False):
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                x, layer.w13_weight, layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids,
            )

        from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm

        moe_kernel = _load_xfp_moe_gemm()
        if moe_kernel is None:
            raise RuntimeError("XFP MoE kernel not available")

        bits = layer._xfp_moe_bits
        B, K_in = x.shape
        topk = topk_ids.shape[1]
        E = layer._xfp_moe_E
        N13 = layer._xfp_moe_N13
        K13 = layer._xfp_moe_K13
        N2 = layer._xfp_moe_N2
        K2 = layer._xfp_moe_K2
        half_n = N13 // 2

        # Token sorting (Marlin pattern)
        sorted_token_ids, expert_ids, num_tokens_post = moe_align_block_size(
            topk_ids, block_size=1, num_experts=E)
        num_valid = int(num_tokens_post.item())

        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        no_weights = torch.empty(0, dtype=torch.float32, device=x.device)

        # Debug: log first call details
        if not hasattr(layer, '_xfp_debug_logged'):
            logger.info(
                "XFP MoE apply: B=%d topk=%d E=%d N13=%d K13=%d N2=%d K2=%d "
                "num_valid=%d fpe13=%d fpe2=%d",
                B, topk, E, N13, K13, N2, K2, num_valid,
                layer._xfp_moe_fpe13, layer._xfp_moe_fpe2,
            )
            logger.info(
                "  sorted_ids[:8]=%s expert_ids[:8]=%s topk_ids[:2]=%s",
                sorted_token_ids[:min(8, num_valid)].tolist(),
                expert_ids[:min(8, num_valid)].tolist(),
                topk_ids[:min(2, B)].tolist(),
            )
            layer._xfp_debug_logged = True

        BT = B * topk

        # Gate+Up GEMM: kernel writes C[token_id] in ORIGINAL topk order
        gate_up = torch.zeros(
            BT, N13, dtype=torch.bfloat16, device=x.device)
        moe_kernel.xfp_moe_gemm(
            x_bf16, layer.w13_xfp_packed, layer.w13_xfp_codebook,
            gate_up, sorted_token_ids[:num_valid], expert_ids,
            no_weights,
            int(bits), int(K13), int(N13), int(topk),
            int(layer._xfp_moe_fpe13), num_valid)

        # SiLU activation (gate_up is in original topk order)
        gate = F.silu(gate_up[:, :half_n])
        up = gate_up[:, half_n:]
        activated = gate * up  # [BT, half_n]

        # Down GEMM: activated[i] is for topk-position i.
        # Use identity sorted_ids + expert mapping from topk_ids directly.
        down = torch.zeros(
            BT, N2, dtype=torch.bfloat16, device=x.device)
        # Build expert_ids for down: topk_ids flattened
        down_expert_ids = topk_ids.reshape(-1).to(torch.int32)
        down_sorted_ids = torch.arange(BT, dtype=torch.int32, device=x.device)
        moe_kernel.xfp_moe_gemm(
            activated, layer.w2_xfp_packed, layer.w2_xfp_codebook,
            down, down_sorted_ids, down_expert_ids,
            no_weights,
            int(bits), int(K2), int(N2), 1,  # top_k=1
            int(layer._xfp_moe_fpe2), BT)

        # Scatter-reduce with topk_weights back to [B, K_in]
        # C is written at C[token_id] by the kernel, so C is in ORIGINAL
        # topk order (not sorted order). Index directly.
        BT = B * topk
        orig_tokens = torch.arange(BT, device=x.device, dtype=torch.int64) // topk
        topk_w_flat = topk_weights.reshape(-1)
        weighted_down = down[:BT].float() * topk_w_flat.unsqueeze(1)

        output = torch.zeros(B, N2, dtype=x.dtype, device=x.device)
        output.scatter_add_(
            0,
            orig_tokens.unsqueeze(1).expand_as(weighted_down),
            weighted_down.to(output.dtype),
        )

        return output
