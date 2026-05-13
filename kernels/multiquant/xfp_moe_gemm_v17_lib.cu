// SPDX-License-Identifier: Apache-2.0
// XFP-V2 MoE GEMM v17_lib — per-group + shared codebook library, MoE variant.
//
// Companion to xfp_gemm_v17_lib.cu (Linear V2). Reuses the same
// xfp_gemm_v2_templated_kernel core, but with MoEPolicyV2 — meaning
// B_packed is indexed by expert_id, per-group metadata is sliced by
// expert × N × G, and output is written to sorted_token_ids[tb].
//
// This replaces the temporary "dequant ALL experts to BF16" reference
// path in online_moe.py — V2 MoE now operates directly on packed indices,
// just like V1 does. No bf16 expert materialization.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core_v2.cuh"

namespace multiquant {

template <int BITS>
static void launch_moe_v2(
    torch::Tensor A, torch::Tensor B_packed,
    torch::Tensor library, torch::Tensor group_lib_id,
    torch::Tensor group_scale, torch::Tensor group_mid,
    torch::Tensor C,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor topk_weights,
    int M, int N, int K, int K_packed,
    int G, int group_size, int library_size,
    int top_k, int flat_per_expert, int num_valid_tokens)
{
    int num_token_blocks = static_cast<int>(sorted_token_ids.size(0));

    dim3 block(XFP_BLOCK_SIZE);

    const float* tw_ptr = (topk_weights.defined() && topk_weights.numel() > 0)
        ? topk_weights.data_ptr<float>() : nullptr;

    // CUDA grid Y is capped at 65535. For prefill chunks with topk=8 and
    // max_num_batched_tokens=8192, num_token_blocks reaches 65536. Chunk
    // the launch into ≤65535-sized batches and advance the sorted_token_ids
    // / expert_ids pointers per chunk.
    constexpr int MAX_GRID_Y = 65535;

    // Dynamic SMEM: s_A (K * bf16) + s_library (library_size * 16 * fp16).
    size_t smem_bytes = static_cast<size_t>(K) * sizeof(__nv_bfloat16)
                      + static_cast<size_t>(library_size) * 16 * sizeof(__half);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int blocks_done = 0;
    int32_t* sorted_ids_base = sorted_token_ids.data_ptr<int32_t>();
    int32_t* expert_ids_base = expert_ids.data_ptr<int32_t>();

    while (blocks_done < num_token_blocks) {
        int chunk = num_token_blocks - blocks_done;
        if (chunk > MAX_GRID_Y) chunk = MAX_GRID_Y;

        MoEPolicyV2::Params params{
            sorted_ids_base + blocks_done,
            expert_ids_base + blocks_done,
            tw_ptr,
            M, top_k, flat_per_expert, num_valid_tokens, G
        };

        dim3 grid(
            (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
            chunk
        );

        xfp_gemm_v2_templated_kernel<BITS, MoEPolicyV2>
            <<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
                reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
                reinterpret_cast<const half*>(library.data_ptr()),
                group_lib_id.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(group_scale.data_ptr()),
                reinterpret_cast<const half*>(group_mid.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, K_packed, G, group_size, library_size, params);

        blocks_done += chunk;
    }
}

void xfp_moe_gemm_v17_lib(
    torch::Tensor A,
    torch::Tensor B_packed,           // [E * flat_per_expert] int32 (warp-interleaved)
    torch::Tensor library,            // [L, LUT_SIZE] fp16 (shared across experts)
    torch::Tensor group_lib_id,       // [E, N, G] int32
    torch::Tensor group_scale,        // [E, N, G] fp16
    torch::Tensor group_mid,          // [E, N, G] fp16
    torch::Tensor C,                  // [num_tokens_padded, N] bf16
    torch::Tensor sorted_token_ids,   // [num_token_blocks] int32
    torch::Tensor expert_ids,         // [num_token_blocks] int32
    torch::Tensor topk_weights,       // [M * top_k] float (or empty)
    int64_t bits,
    int64_t K,
    int64_t N,
    int64_t group_size,
    int64_t top_k,
    int64_t flat_per_expert,
    int64_t num_valid_tokens)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && library.is_cuda() &&
                group_lib_id.is_cuda() && group_scale.is_cuda() &&
                group_mid.is_cuda() && C.is_cuda() &&
                sorted_token_ids.is_cuda() && expert_ids.is_cuda(),
                "xfp_moe_gemm_v17_lib: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(library.dtype() == torch::kFloat16, "library must be fp16");
    TORCH_CHECK(group_lib_id.dtype() == torch::kInt32,
                "group_lib_id must be int32");
    TORCH_CHECK(group_scale.dtype() == torch::kFloat16,
                "group_scale must be fp16");
    TORCH_CHECK(group_mid.dtype() == torch::kFloat16,
                "group_mid must be fp16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(K % 2 == 0, "K must be even (bf162 vector load)");
    TORCH_CHECK(K <= MoEPolicyV2::K_SMEM_MAX,
                "xfp_moe_gemm_v17_lib: K=", K, " exceeds K_SMEM_MAX=",
                MoEPolicyV2::K_SMEM_MAX);
    TORCH_CHECK(bits == 2 || bits == 4,
                "MoE v17_lib: only BITS=2 or 4 supported, got ", bits);
    TORCH_CHECK(group_size == 128,
                "MoE v17_lib v1: only group_size=128 supported, got ",
                group_size);
    TORCH_CHECK(K % group_size == 0,
                "K=", K, " not divisible by group_size=", group_size);

    int M = static_cast<int>(A.size(0));
    int G = static_cast<int>(K / group_size);
    int library_size = static_cast<int>(library.size(0));
    int vals_per_word = (bits == 2) ? 16 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    TORCH_CHECK(library_size <= XFP_V2_LIBRARY_MAX,
                "library_size=", library_size, " > XFP_V2_LIBRARY_MAX=",
                XFP_V2_LIBRARY_MAX);
    TORCH_CHECK(group_lib_id.size(2) == G,
                "group_lib_id last dim ", group_lib_id.size(2),
                " != expected G=", G);

#define LAUNCH_MOE(BITS_) \
    launch_moe_v2<BITS_>(A, B_packed, library, group_lib_id, group_scale, \
                         group_mid, C, sorted_token_ids, expert_ids, topk_weights, \
                         M, static_cast<int>(N), static_cast<int>(K), K_packed, \
                         G, static_cast<int>(group_size), library_size, \
                         static_cast<int>(top_k), \
                         static_cast<int>(flat_per_expert), \
                         static_cast<int>(num_valid_tokens))
    switch (bits) {
        case 2: LAUNCH_MOE(2); break;
        case 4: LAUNCH_MOE(4); break;
        default: TORCH_CHECK(false, "unreachable");
    }
#undef LAUNCH_MOE
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_moe_gemm_v17_lib", &multiquant::xfp_moe_gemm_v17_lib,
          "XFP-V2 MoE GEMM with shared library + per-(expert, N, group) scale/mid");
}
