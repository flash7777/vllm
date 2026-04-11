# SPDX-License-Identifier: Apache-2.0
"""XFP online MoE method — BF16 → per-expert learned codebook at load time.

Pipeline:
    create_weights                — delegate to UnquantizedFusedMoEMethod
                                    (BF16 alloc for w13_weight / w2_weight)
    process_weights_after_loading — per-expert xfp_pack, stack [E, ...] tensors,
                                    drop BF16 weights, record stats per expert
    apply                         — naive per-token × per-topk × per-expert
                                    loop calling xfp_gemm twice (gate+up, down)
                                    with SiLU between

v1 is a reference implementation — correctness first. Batched grouped MoE
kernel is v3 scope.
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
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


class XFPMoEMethod(FusedMoEMethodBase):
    """Learned-codebook quant-on-load for FusedMoE layers.

    Per-expert Lloyd codebook + word-aligned packed indices. Apply path
    is the per-token naive loop — reference implementation, not perf.
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
                f"supported: xfp2, xfp3, xfp4 (v1)"
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

        from vllm.multiquant.xfp.xfp_pack import xfp_pack

        bits = self.bits
        device = layer.w13_weight.device
        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]
        E = int(w13.shape[0])

        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        reg = MultiQuantPolicyRegistry.get_active()

        # Batched pack: flatten [E, N_out, K] to [E*N_out, K], run Lloyd
        # once over all rows (each row is independent), then reshape the
        # results back. This turns 2*E xfp_pack calls into exactly 2 and
        # makes MoE pack time proportional to the total row count, not
        # to E. Scales from O(E × Lloyd_chunk) to O(1 × Lloyd_chunk).
        def _batched_pack(W_stack: torch.Tensor):
            """W_stack: [E, N, K] -> packed [E, K_packed, N], codebook [E, N, 2^b]."""
            E_ = int(W_stack.shape[0])
            N_ = int(W_stack.shape[1])
            K_ = int(W_stack.shape[2])
            W_flat = W_stack.reshape(E_ * N_, K_).float()
            packed_flat, codebook_flat, stats = xfp_pack(W_flat, bits=bits)
            # packed_flat is [K_packed, E*N]; reshape N dim
            k_packed = packed_flat.shape[0]
            packed = packed_flat.view(k_packed, E_, N_).permute(1, 0, 2).contiguous()
            codebook = codebook_flat.view(E_, N_, 1 << bits)
            return packed, codebook, stats

        p13, cb13, stats13 = _batched_pack(w13)
        p2, cb2, stats2 = _batched_pack(w2)

        if reg is not None:
            # One aggregate stats record per tensor — cheaper than one-per-expert.
            reg.record_stats("routed_expert", stats13)
            reg.record_stats("routed_expert", stats2)

        layer.w13_xfp_packed = nn.Parameter(p13.to(device), requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(cb13.to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(p2.to(device), requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(cb2.to(device), requires_grad=False)

        # Save shape metadata so apply() can call the kernel.
        layer._xfp_moe_bits = bits
        layer._xfp_moe_dtype = self.dtype
        layer._xfp_moe_K13 = int(w13.shape[2])     # hidden in
        layer._xfp_moe_N13 = int(w13.shape[1])     # 2 * intermediate
        layer._xfp_moe_K2 = int(w2.shape[2])       # intermediate in
        layer._xfp_moe_N2 = int(w2.shape[1])       # hidden out
        layer._xfp_moe_E = E
        layer._xfp_moe_packed = True

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

        logger.info(
            "XFP MoE: %d experts w13[%dx%d] + w2[%dx%d] -> %s",
            E, layer._xfp_moe_N13, layer._xfp_moe_K13,
            layer._xfp_moe_N2, layer._xfp_moe_K2, self.dtype,
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

        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm

        bits = layer._xfp_moe_bits
        kernel = _load_xfp_gemm(bits)
        if kernel is None:
            raise RuntimeError(
                f"XFPMoEMethod.apply: xfp_gemm kernel not available for bits={bits}"
            )

        B, K = x.shape
        topk = topk_ids.shape[1]
        E = layer._xfp_moe_E
        N13 = layer._xfp_moe_N13
        K13 = layer._xfp_moe_K13
        N2 = layer._xfp_moe_N2
        K2 = layer._xfp_moe_K2

        output = torch.zeros(B, K, dtype=x.dtype, device=x.device)

        for b in range(B):
            for t in range(topk):
                expert_id = int(topk_ids[b, t].item())
                weight = float(topk_weights[b, t].item())
                if expert_id < 0 or expert_id >= E:
                    continue

                x_row = x[b:b + 1].to(torch.float16)  # [1, K]

                # Gate+Up: x_row [1, K13] @ dequant(w13[e]) -> [1, N13]
                w13_p = layer.w13_xfp_packed[expert_id]
                w13_c = layer.w13_xfp_codebook[expert_id]
                gate_up = torch.zeros(
                    1, N13, dtype=torch.float16, device=x.device
                )
                kernel.xfp_gemm(
                    x_row, w13_p, w13_c, gate_up, int(bits), int(K13)
                )

                # SiLU(gate) * up — first half is gate, second half is up
                half_n = N13 // 2
                gate = gate_up[0, :half_n]
                up = gate_up[0, half_n:]
                activated = (F.silu(gate) * up).unsqueeze(0)  # [1, half_n]

                # Down: activated [1, K2] @ dequant(w2[e]) -> [1, N2]
                w2_p = layer.w2_xfp_packed[expert_id]
                w2_c = layer.w2_xfp_codebook[expert_id]
                down = torch.zeros(
                    1, N2, dtype=torch.float16, device=x.device
                )
                kernel.xfp_gemm(
                    activated, w2_p, w2_c, down, int(bits), int(K2)
                )

                output[b] = output[b] + weight * down[0].to(output.dtype)

        if shared_experts_input is not None:
            return output, shared_experts_input
        return output
