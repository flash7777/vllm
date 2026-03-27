// SPDX-License-Identifier: Apache-2.0
// TurboQuant Fused Compressed Decode Attention
//
// Single kernel: compressed K-cache → scores → softmax → compressed V → output
// No decompression buffers needed. Reads 28 bytes/key + 28 bytes/value (TQ3/D=64).
//
// For decode: 1 query token attending to all cached tokens.
// Each thread block handles one (query_head, cache_block) pair.
// Reduction across blocks done in a second pass (or atomics).
//
// Simplified approach for standalone testing:
// - One block per query_head processes ALL tokens sequentially
// - Online softmax (numerically stable, single pass)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace turboquant {

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    return val;
}

// Fused decode attention from compressed KV cache.
// One block per query head. Processes all tokens sequentially.
// Uses online softmax for numerical stability.
//
// Grid: (num_q_tokens * num_q_heads)
// Block: (HEAD_DIM) — one thread per dimension for output accumulation
template <int HEAD_DIM, int MSE_BITS, int N_CENTROIDS>
__global__ void tq_fused_decode_attention_kernel(
    // Precomputed query projections
    const float* __restrict__ q_rot,       // [num_q, num_q_heads, D]
    const float* __restrict__ q_proj,      // [num_q, num_q_heads, D]
    // Compressed KV cache (K and V packed separately)
    const uint8_t* __restrict__ k_cache,   // [num_blocks, block_size, num_kv_heads, packed_size]
    const uint8_t* __restrict__ v_cache,   // [num_blocks, block_size, num_kv_heads, packed_size]
    // Reconstruction matrices (for V decompression)
    const float* __restrict__ Pi,          // [D, D]
    const float* __restrict__ S,           // [D, D]
    const float* __restrict__ centroids,   // [N_CENTROIDS]
    // Block table
    const int* __restrict__ block_table,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    // Output
    float* __restrict__ output,            // [num_q, num_q_heads, D]
    // Dims
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int packed_size,
    int max_blocks_per_seq,
    float attn_scale,
    float correction_scale
) {
    const int qh_idx = blockIdx.x;
    const int q_token = qh_idx / num_q_heads;
    const int q_head = qh_idx % num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;
    constexpr int D = HEAD_DIM;
    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    int seq_len = seq_lens[q_token];

    // Shared memory for query projections and centroids
    __shared__ float s_q_rot[D];
    __shared__ float s_q_proj[D];
    __shared__ float s_centroids[N_CENTROIDS];

    int q_base = (q_token * num_q_heads + q_head) * D;
    for (int i = tid; i < D; i += blockDim.x) {
        s_q_rot[i] = q_rot[q_base + i];
        s_q_proj[i] = q_proj[q_base + i];
    }
    if (tid < N_CENTROIDS) s_centroids[tid] = centroids[tid];
    __syncthreads();

    // Online softmax state (per thread, accumulating output[tid])
    float m_prev = -INFINITY;  // running max
    float d_prev = 0.0f;       // running sum of exp
    float acc = 0.0f;          // running weighted sum for output[tid]

    // Process all cached tokens
    for (int pos = 0; pos < seq_len; pos++) {
        int bi = pos / block_size;
        int bo = pos % block_size;
        int phys_block = block_table[q_token * max_blocks_per_seq + bi];

        int k_entry = (phys_block * block_size + bo) * num_kv_heads * packed_size
                    + kv_head * packed_size;
        const uint8_t* k_packed = k_cache + k_entry;

        // === Compute score from compressed K ===
        // Only thread 0 computes the score (all dims needed)
        __shared__ float s_score;
        if (tid == 0) {
            float term1 = 0.0f;
            if constexpr (MSE_BITS == 2) {
                for (int b = 0; b < MSE_BYTES; b++) {
                    uint8_t byte_val = k_packed[b];
                    for (int k = 0; k < 4 && (b*4+k) < D; k++) {
                        int j = b*4+k;
                        int idx = (byte_val >> (k*2)) & MASK;
                        term1 += s_q_rot[j] * s_centroids[idx];
                    }
                }
            } else if constexpr (MSE_BITS == 3) {
                for (int j = 0; j < D; j++) {
                    int bit_off = j*3; int byte_idx = bit_off/8; int bit_idx = bit_off%8;
                    int val = (k_packed[byte_idx] >> bit_idx);
                    if (bit_idx > 5 && byte_idx+1 < MSE_BYTES)
                        val |= (k_packed[byte_idx+1] << (8-bit_idx));
                    term1 += s_q_rot[j] * s_centroids[val & MASK];
                }
            }

            float term2 = 0.0f;
            const uint8_t* k_signs = k_packed + MSE_BYTES;
            for (int b = 0; b < QJL_BYTES; b++) {
                uint8_t bv = k_signs[b];
                for (int k = 0; k < 8 && (b*8+k) < D; k++) {
                    int j = b*8+k;
                    float sv = ((bv >> k) & 1) ? 1.0f : -1.0f;
                    term2 += s_q_proj[j] * sv;
                }
            }

            int norm_off = MSE_BYTES + QJL_BYTES;
            uint16_t vn_u16 = k_packed[norm_off] | (k_packed[norm_off+1] << 8);
            uint16_t rn_u16 = k_packed[norm_off+2] | (k_packed[norm_off+3] << 8);
            float vn = __half2float(*reinterpret_cast<const __half*>(&vn_u16));
            float rn = __half2float(*reinterpret_cast<const __half*>(&rn_u16));

            s_score = vn * (term1 + correction_scale * rn * term2) * attn_scale;
        }
        __syncthreads();
        float score = s_score;

        // === Decompress V for this token (each thread handles one dim) ===
        float v_val = 0.0f;
        if (tid < D) {
            int v_entry = (phys_block * block_size + bo) * num_kv_heads * packed_size
                        + kv_head * packed_size;
            const uint8_t* v_packed = v_cache + v_entry;

            // Unpack V indices
            int v_idx;
            if constexpr (MSE_BITS == 2) {
                int b_idx = tid / 4; int k = tid % 4;
                v_idx = (v_packed[b_idx] >> (k*2)) & MASK;
            } else {
                int bit_off = tid * MSE_BITS;
                int byte_idx = bit_off / 8; int bit_idx = bit_off % 8;
                v_idx = (v_packed[byte_idx] >> bit_idx);
                if (bit_idx > 5 && byte_idx+1 < MSE_BYTES)
                    v_idx |= (v_packed[byte_idx+1] << (8-bit_idx));
                v_idx &= MASK;
            }

            // Unpack V sign
            int s_byte = MSE_BYTES + tid / 8;
            int s_bit = tid % 8;
            float v_sign = ((v_packed[s_byte] >> s_bit) & 1) ? 1.0f : -1.0f;

            // Unpack V norms
            int vn_off = MSE_BYTES + QJL_BYTES;
            uint16_t vvn_u16 = v_packed[vn_off] | (v_packed[vn_off+1] << 8);
            uint16_t vrn_u16 = v_packed[vn_off+2] | (v_packed[vn_off+3] << 8);
            float v_vecnorm = __half2float(*reinterpret_cast<const __half*>(&vvn_u16));
            float v_resnorm = __half2float(*reinterpret_cast<const __half*>(&vrn_u16));

            // Reconstruct V[tid]: vec_norm * (Pi^T @ centroids[idx] + corr * res_norm * S^T @ sign)[tid]
            // V_mse[tid] = sum_j Pi[j, tid] * centroids[idx[j]]  — but we only have idx for THIS dim
            // Actually for per-element decompression we need the FULL idx vector.
            // Simplified: use centroid[v_idx] directly (MSE only, no rotation unrotate)
            // This is an approximation — correct version needs the full GEMV.
            //
            // For now: v_val = vec_norm * centroids[v_idx]
            // TODO: Full decompression with Pi rotation
            v_val = v_vecnorm * s_centroids[v_idx];
        }

        // === Online softmax + weighted V accumulation ===
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(score - m_new);
        float d_new = d_prev * exp_prev + exp_cur;

        if (tid < D) {
            acc = acc * exp_prev + exp_cur * v_val;
        }
        m_prev = m_new;
        d_prev = d_new;
    }

    // Final normalization
    if (tid < D && d_prev > 0.0f) {
        output[q_base + tid] = acc / d_prev;
    }
}


void tq_fused_decode_attention(
    torch::Tensor q_rot,
    torch::Tensor q_proj,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor Pi,
    torch::Tensor S,
    torch::Tensor centroids,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor output,
    int head_dim,
    int mse_bits,
    int n_centroids,
    float attn_scale
) {
    int num_q = q_rot.size(0);
    int num_q_heads = q_rot.size(1);
    int num_kv_heads = k_cache.size(2);
    int block_size = k_cache.size(1);
    int packed_size = k_cache.size(3);
    int max_blocks = block_table.size(1);
    float correction = sqrtf(M_PI_2) / static_cast<float>(head_dim);

    dim3 grid(num_q * num_q_heads);
    // Need at least D threads for V decompression + output accumulation
    int threads = ((head_dim + 31) / 32) * 32;
    dim3 block(threads);

    if (head_dim == 64 && mse_bits == 2 && n_centroids == 4) {
        tq_fused_decode_attention_kernel<64, 2, 4><<<grid, block>>>(
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(),
            k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
            Pi.data_ptr<float>(), S.data_ptr<float>(),
            centroids.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            output.data_ptr<float>(),
            num_q_heads, num_kv_heads, block_size, packed_size,
            max_blocks, attn_scale, correction);
    } else if (head_dim == 64 && mse_bits == 3 && n_centroids == 8) {
        tq_fused_decode_attention_kernel<64, 3, 8><<<grid, block>>>(
            q_rot.data_ptr<float>(), q_proj.data_ptr<float>(),
            k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
            Pi.data_ptr<float>(), S.data_ptr<float>(),
            centroids.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            output.data_ptr<float>(),
            num_q_heads, num_kv_heads, block_size, packed_size,
            max_blocks, attn_scale, correction);
    } else {
        TORCH_CHECK(false, "TQ attention: unsupported config");
    }
}

}  // namespace turboquant
