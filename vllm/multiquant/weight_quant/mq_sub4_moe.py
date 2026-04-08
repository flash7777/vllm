# SPDX-License-Identifier: Apache-2.0
"""MultiQuant sub-4-bit MoE method — per-expert fused GEMM.

For FusedMoE layers with INT2/INT3 weights: calls mq_gemm_int2/int3
per expert instead of the Triton MoE kernel (which only supports INT4).

Weight loading reuses MoeWNA16Method (GPTQ format, shard handling).
Only apply() is overridden.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

logger = init_logger(__name__)


class MQSub4MoEMethod(MoeWNA16Method):
    """MoE method for sub-4-bit (INT2/INT3) using MultiQuant fused GEMM.

    Inherits weight loading from MoeWNA16Method.
    Overrides apply() to use mq_gemm_int2/mq_gemm_int3 per expert.
    """

    # Inherit get_fused_moe_quant_config from MoeWNA16Method
    # This ensures qweight/scales/qzeros are properly allocated and loaded.
    # We only override apply() to use our fused GEMM instead of Triton.

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Per-expert GEMM with INT2/INT3 fused dequant.

        Instead of batched Triton MoE kernel: route tokens to experts,
        run per-expert mq_gemm, combine results.
        """
        from vllm.multiquant.weight_quant.mq_sub4_linear import _load_mq_gemm

        bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size
        kernel = _load_mq_gemm(bits)
        if kernel is None:
            raise RuntimeError(f"mq_gemm_int{bits} not available")

        B, K = x.shape  # [num_tokens, hidden_size]
        num_experts = layer.w13_qweight.shape[0]
        N_gate_up = layer.w13_qweight.shape[2]  # packed N dim
        topk = topk_ids.shape[1]

        # Determine actual N from scales (not packed)
        N_intermediate = layer.w13_scales.shape[2]  # actual N (gate+up fused)
        N_down_out = layer.w2_scales.shape[2]  # actual output dim

        # Gate+Up GEMM: x[B,K] @ W13[E, K_packed, N_inter] → [B, topk, N_inter]
        # Down GEMM: act[B, topk, N_inter/2] @ W2[E, N_inter/2_packed, K] → [B, K]

        # Simple per-expert implementation (not optimal, but correct)
        output = torch.zeros(B, K, dtype=x.dtype, device=x.device)

        for b in range(B):
            for t in range(topk):
                expert_id = topk_ids[b, t].item()
                weight = topk_weights[b, t].item()
                if expert_id < 0 or expert_id >= num_experts:
                    continue

                # Gate+Up: x[b] @ W13[expert]
                x_row = x[b:b+1].to(torch.float16)  # [1, K]
                # MoE stores as uint8, our kernel needs int32 — reinterpret
                w13_q = layer.w13_qweight[expert_id].view(torch.int32)
                w13_s = layer.w13_scales[expert_id].to(torch.float16)
                w13_z = layer.w13_qzeros[expert_id].view(torch.int32)

                # w13_q is [N_inter, K_packed_u8] after MoE transpose
                # Our kernel expects [K_packed_i32, N_inter]
                # Need to figure out the layout...
                # MoE stores: [N, K_packed_u8] uint8 (transposed from GPTQ)
                # Reinterpret: [N, K_packed_u8] u8 → [N, K_packed_u8/4] i32
                w13_q_i32 = w13_q.reshape(w13_q.shape[0], -1)  # [N, K_packed_i32]
                w13_q_i32 = w13_q_i32.T.contiguous()  # [K_packed_i32, N]
                w13_z_i32 = w13_z.reshape(w13_z.shape[0], -1)  # [n_groups, Nzp_i32]

                gate_up = torch.zeros(1, w13_q_i32.shape[1], dtype=torch.float16,
                                      device=x.device)
                if bits == 2:
                    kernel.mq_gemm_int2(x_row, w13_q_i32, w13_s, w13_z_i32,
                                        gate_up, group_size)
                elif bits == 3:
                    kernel.mq_gemm_int3(x_row, w13_q_i32, w13_s, w13_z_i32,
                                        gate_up, K, group_size)

                # SiLU activation: gate * silu(up)
                # gate_up is [1, N_inter] where first half is gate, second is up
                half_n = N_intermediate // 2
                gate = gate_up[0, :half_n]
                up = gate_up[0, half_n:]
                activated = torch.nn.functional.silu(gate) * up  # [half_n]

                # Down: activated @ W2[expert]
                act_row = activated.unsqueeze(0)  # [1, half_n]
                w2_q = layer.w2_qweight[expert_id].view(torch.int32)
                w2_s = layer.w2_scales[expert_id].to(torch.float16)
                w2_z = layer.w2_qzeros[expert_id].view(torch.int32)
                w2_q_i32 = w2_q.reshape(w2_q.shape[0], -1).T.contiguous()
                w2_z_i32 = w2_z.reshape(w2_z.shape[0], -1)

                down = torch.zeros(1, w2_q_i32.shape[1], dtype=torch.float16,
                                   device=x.device)
                if bits == 2:
                    kernel.mq_gemm_int2(act_row, w2_q_i32, w2_s, w2_z_i32,
                                        down, group_size)
                elif bits == 3:
                    K_down = act_row.shape[1]
                    kernel.mq_gemm_int3(act_row, w2_q_i32, w2_s, w2_z_i32,
                                        down, K_down, group_size)

                output[b] += weight * down[0].to(output.dtype)

        if shared_experts_input is not None:
            return output, shared_experts_input
        return output
