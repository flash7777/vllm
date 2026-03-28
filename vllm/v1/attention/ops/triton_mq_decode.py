# SPDX-License-Identifier: Apache-2.0
"""MultiQuant fused MLA decode attention — Triton kernel.

Reads directly from packed uint8 KV-cache. No decompress step.
Inline: unpack MSE indices → centroid lookup → inverse rotation → Q·K score.

Based on triton_decode_attention.py (vLLM/SGLang).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mq_mla_decode_stage1(
    Q,                      # [B, num_heads, qk_dim]
    KV_Cache,               # [num_pages, page_size, packed_size] uint8
    Centroids,              # [n_centroids] float32
    Pi,                     # [D, D] float32 (rotation matrix)
    S,                      # [D, D] float32 (QJL projection)
    Block_table,            # [B, max_blocks] int32
    Seq_lens,               # [B] int32
    Out,                    # [B, num_heads, kv_lora_rank]
    Lse,                    # [B, num_heads]
    sm_scale,
    stride_qb, stride_qh,
    stride_cache_page, stride_cache_slot,
    stride_pi_row,
    stride_s_row,
    stride_bt_b,
    stride_ob, stride_oh,
    D: tl.constexpr,           # original head_size (latent dim)
    MSE_BITS: tl.constexpr,    # 2 or 3
    PACKED_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,     # tokens per iteration
):
    """Stage 1: compute attention scores from packed cache.

    Each program handles one (batch, head) pair.
    Iterates over cached tokens in blocks of BLOCK_N.
    For each cached token: unpack → centroid → rotate → dot with q.
    Online softmax accumulation.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    seq_len = tl.load(Seq_lens + cur_batch)
    if seq_len <= 0:
        return

    # Load query for this head
    offs_d = tl.arange(0, D)
    mask_d = offs_d < D
    q = tl.load(Q + cur_batch * stride_qb + cur_head * stride_qh + offs_d,
                mask=mask_d, other=0.0)

    # Constants for unpacking
    MSE_BYTES: tl.constexpr = (D * MSE_BITS + 7) // 8
    QJL_BYTES: tl.constexpr = (D + 7) // 8
    COORDS_PER_BYTE: tl.constexpr = 8 // MSE_BITS
    MASK: tl.constexpr = (1 << MSE_BITS) - 1
    CORR_SCALE = 1.2533141373155003 / D  # sqrt(π/2) / D

    # Online softmax state
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([KV_LORA_RANK], dtype=tl.float32)

    # Iterate over cached tokens
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = offs_n < seq_len

        # For each token in this block, compute score
        for ni in range(BLOCK_N):
            token_idx = start_n + ni
            if token_idx >= seq_len:
                break

            # Page lookup
            page_idx = tl.load(
                Block_table + cur_batch * stride_bt_b + token_idx // PAGE_SIZE
            )
            slot_in_page = token_idx % PAGE_SIZE
            cache_offset = page_idx * stride_cache_page + slot_in_page * stride_cache_slot

            # Load packed bytes for this token
            # Unpack MSE indices and compute centroid values
            k_reconstructed = tl.zeros([D], dtype=tl.float32)

            # Simple per-coordinate unpack + centroid lookup
            for j in range(D):
                byte_idx = j // COORDS_PER_BYTE
                bit_pos = (j % COORDS_PER_BYTE) * MSE_BITS
                packed_byte = tl.load(KV_Cache + cache_offset + byte_idx)
                idx = (packed_byte >> bit_pos) & MASK
                # Centroid lookup
                k_reconstructed = tl.where(
                    offs_d == j,
                    tl.load(Centroids + idx).to(tl.float32),
                    k_reconstructed,
                )

            # Inverse rotation: Pi^T @ k_reconstructed
            # This is the expensive part — D×D matmul per token
            k_rotated = tl.zeros([D], dtype=tl.float32)
            for row in range(D):
                # k_rotated[row] = sum_j Pi[j, row] * k_reconstructed[j]
                # = column `row` of Pi dotted with k_reconstructed
                pi_col = tl.load(Pi + tl.arange(0, D) * stride_pi_row + row,
                                 mask=mask_d, other=0.0)
                k_rotated = tl.where(
                    offs_d == row,
                    tl.sum(pi_col * k_reconstructed),
                    k_rotated,
                )

            # Unpack norms
            norm_offset = cache_offset + MSE_BYTES + QJL_BYTES
            # TODO: load float16 norms properly in Triton
            # For now: approximate with unit norms
            vec_norm = 1.0
            res_norm = 0.0

            # Reconstruct: vec_norm * k_rotated (skip QJL for now — perf first)
            k_final = vec_norm * k_rotated

            # Score: q · k
            score = tl.sum(q * k_final) * sm_scale

            # Online softmax update
            m_new = tl.maximum(m_prev, score)
            exp_prev = tl.exp(m_prev - m_new)
            exp_new = tl.exp(score - m_new)
            l_new = l_prev * exp_prev + exp_new

            # For MLA: output is kv_lora_rank dims of the latent
            # Use k_final[:KV_LORA_RANK] as the "value" component
            v_component = tl.zeros([KV_LORA_RANK], dtype=tl.float32)
            offs_v = tl.arange(0, KV_LORA_RANK)
            mask_v = offs_v < KV_LORA_RANK
            for vj in range(KV_LORA_RANK):
                byte_idx = vj // COORDS_PER_BYTE
                bit_pos = (vj % COORDS_PER_BYTE) * MSE_BITS
                packed_byte = tl.load(KV_Cache + cache_offset + byte_idx)
                idx = (packed_byte >> bit_pos) & MASK
                v_component = tl.where(
                    offs_v == vj,
                    tl.load(Centroids + idx).to(tl.float32),
                    v_component,
                )

            acc = acc * (l_prev * exp_prev / l_new) + v_component * (exp_new / l_new)
            m_prev = m_new
            l_prev = l_new

    # Write output
    offs_out = cur_batch * stride_ob + cur_head * stride_oh + tl.arange(0, KV_LORA_RANK)
    tl.store(Out + offs_out, acc.to(Out.dtype.element_ty), mask=tl.arange(0, KV_LORA_RANK) < KV_LORA_RANK)
    tl.store(Lse + cur_batch * stride_oh + cur_head, m_prev + tl.log(l_prev))


def mq_mla_decode_attention(
    q: torch.Tensor,            # [B, num_heads, qk_dim]
    kv_cache: torch.Tensor,     # [num_pages, page_size, packed_size] uint8
    centroids: torch.Tensor,    # [n_centroids] float32
    Pi: torch.Tensor,           # [D, D] float32
    S: torch.Tensor,            # [D, D] float32
    block_table: torch.Tensor,  # [B, max_blocks]
    seq_lens: torch.Tensor,     # [B]
    scale: float,
    page_size: int,
    head_size: int,
    kv_lora_rank: int,
    mse_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MQ-MLA decode: read packed cache → attention → output.

    NOTE: This is a PROTOTYPE Triton kernel. Per-token D×D rotation
    is very slow in Triton (no shared memory tiling). Production version
    needs batched approach or pre-rotation of queries.

    Returns: (output [B, num_heads, kv_lora_rank], lse [B, num_heads])
    """
    B, num_heads, qk_dim = q.shape
    packed_size = kv_cache.shape[2]

    o = torch.zeros(B, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, num_heads, dtype=torch.float32, device=q.device)

    BLOCK_N = 1  # Process 1 token at a time (simple, correct)

    grid = (B, num_heads)
    _mq_mla_decode_stage1[grid](
        q, kv_cache, centroids, Pi, S,
        block_table, seq_lens, o, lse,
        scale,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        Pi.stride(0), S.stride(0),
        block_table.stride(0),
        o.stride(0), o.stride(1),
        D=head_size,
        MSE_BITS=mse_bits,
        PACKED_SIZE=packed_size,
        KV_LORA_RANK=kv_lora_rank,
        PAGE_SIZE=page_size,
        BLOCK_N=BLOCK_N,
    )

    return o, lse
