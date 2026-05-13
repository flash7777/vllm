// SPDX-License-Identifier: Apache-2.0
// XFP-V2 Linear GEMM v17_lib_splitk — V3 reference kernel.
//
// Companion to xfp_gemm_v17_lib.cu (split-N) and xfp_gemm_v17_lib_splitm.cu
// (split-M-internal). This kernel handles K > K_SMEM_MAX_LINEAR=8192 by
// processing the K dimension in K_CHUNK-sized slices, reloading SMEM per
// chunk while accumulators persist in registers across chunks.
//
// Use case: GLM-4.7-Flash shared_expert.down_proj (K=10240),
// GLM-5 shared (K=12288), Qwen3.6-27B intermediate (K=17408).
//
// Selection (xfp_kernel.py):
//   K ≤ 8192 → v17_lib (M=1) or v17_lib_splitm (M ≥ 16)
//   K  > 8192 → v17_lib_splitk (THIS kernel, both M ranges via M_CHUNK)
//
// Performance: ~10–20% slower than hypothetical "infinite-SMEM" V2 path
// due to chunked A-reload; this is the price of supporting K > 8192 on
// workstation Blackwell SMs (sm_120/sm_121, 99 KB SMEM/CTA).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core_v2_splitk.cuh"

namespace multiquant {

template <int BITS, int M_CHUNK, int K_CHUNK>
static void launch_splitk(
    torch::Tensor A, torch::Tensor B_packed,
    torch::Tensor library, torch::Tensor group_lib_id,
    torch::Tensor group_scale, torch::Tensor group_mid,
    torch::Tensor C, int K, int N, int K_packed,
    int G, int group_size, int library_size, int M_total)
{
    dim3 block(XFP_BLOCK_SIZE);
    dim3 grid(
        (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
        (M_total + M_CHUNK - 1) / M_CHUNK
    );

    // Dynamic SMEM: M_CHUNK A-rows of K_CHUNK cols + library
    size_t smem_bytes = (size_t)M_CHUNK * K_CHUNK * sizeof(__nv_bfloat16)
                      + (size_t)library_size * 16 * sizeof(__half);

    // sm_120 (Blackwell consumer/pro) sharedMemPerBlockOptin = 99 KB.
    // Practical M_CHUNK × K_CHUNK combos with K_CHUNK in {2048, 4096}:
    //   K_CHUNK=4096, M_CHUNK=1: s_A=8 KB,  + lib=1 KB → 9 KB total ✓
    //   K_CHUNK=4096, M_CHUNK=2: s_A=16 KB, + lib=1 KB → 17 KB total ✓
    //   K_CHUNK=4096, M_CHUNK=4: s_A=32 KB, + lib=1 KB → 33 KB total ✓
    //   K_CHUNK=2048, M_CHUNK=4: s_A=16 KB, + lib=1 KB → 17 KB total ✓
    static bool s_carveout_set = false;
    if (!s_carveout_set) {
        cudaFuncSetAttribute(
            xfp_gemm_v2_splitk_kernel<BITS, M_CHUNK, K_CHUNK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        s_carveout_set = true;
    }
    TORCH_CHECK(smem_bytes <= 98304,
                "split-K: smem_bytes=", smem_bytes,
                " exceeds 96 KB carveout. M_CHUNK=", M_CHUNK,
                " K_CHUNK=", K_CHUNK, " too large for sm_120.");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v2_splitk_kernel<BITS, M_CHUNK, K_CHUNK>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(library.data_ptr()),
            group_lib_id.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(group_scale.data_ptr()),
            reinterpret_cast<const half*>(group_mid.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, K_packed, G, group_size, library_size, M_total);
}

void xfp_gemm_v17_lib_splitk(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor library,
    torch::Tensor group_lib_id,
    torch::Tensor group_scale,
    torch::Tensor group_mid,
    torch::Tensor C,
    int64_t bits,
    int64_t K,
    int64_t group_size,
    int64_t m_chunk,
    int64_t k_chunk)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && library.is_cuda() &&
                group_lib_id.is_cuda() && group_scale.is_cuda() &&
                group_mid.is_cuda() && C.is_cuda(),
                "xfp_gemm_v17_lib_splitk: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(library.dtype() == torch::kFloat16, "library must be float16");
    TORCH_CHECK(group_lib_id.dtype() == torch::kInt32,
                "group_lib_id must be int32");
    TORCH_CHECK(group_scale.dtype() == torch::kFloat16,
                "group_scale must be float16");
    TORCH_CHECK(group_mid.dtype() == torch::kFloat16,
                "group_mid must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(K % 2 == 0, "K must be even (bf162 vector load)");
    TORCH_CHECK(bits == 2 || bits == 4,
                "split-K v17_lib: only BITS=2 or 4 supported, got ", bits);
    TORCH_CHECK(group_size == 128,
                "split-K v1: only group_size=128 supported, got ", group_size);
    TORCH_CHECK(K % group_size == 0, "K not divisible by group_size");
    TORCH_CHECK(k_chunk == 2048 || k_chunk == 4096,
                "split-K v1: k_chunk must be 2048 or 4096, got ", k_chunk);

    int M = static_cast<int>(A.size(0));
    int N = static_cast<int>(group_scale.size(0));
    int G = static_cast<int>(group_scale.size(1));
    int library_size = static_cast<int>(library.size(0));
    int vals_per_word = (bits == 2) ? 16 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    TORCH_CHECK(library_size <= XFP_V2_LIBRARY_MAX, "library_size > MAX");

    int kK = static_cast<int>(K);
    int gs = static_cast<int>(group_size);

    // Dispatch on (M_CHUNK, K_CHUNK) combinations.
    // M_CHUNK ∈ {1, 2, 4}: 1 for single-stream M=1, 2/4 for batched.
    // K_CHUNK ∈ {2048, 4096}: 4096 default, 2048 fallback for very large M.
#define LAUNCH_SK(BITS_, MC_, KC_) \
        launch_splitk<BITS_, MC_, KC_>(A, B_packed, library, group_lib_id, \
            group_scale, group_mid, C, kK, N, K_packed, G, gs, library_size, M)
#define DISPATCH_MC_KC(BITS_) \
    if (k_chunk == 4096) { \
        switch (m_chunk) { \
            case 1: LAUNCH_SK(BITS_, 1, 4096); break; \
            case 2: LAUNCH_SK(BITS_, 2, 4096); break; \
            case 4: LAUNCH_SK(BITS_, 4, 4096); break; \
            default: TORCH_CHECK(false, "split-K: unsupported m_chunk=", m_chunk); \
        } \
    } else { \
        switch (m_chunk) { \
            case 1: LAUNCH_SK(BITS_, 1, 2048); break; \
            case 2: LAUNCH_SK(BITS_, 2, 2048); break; \
            case 4: LAUNCH_SK(BITS_, 4, 2048); break; \
            default: TORCH_CHECK(false, "split-K: unsupported m_chunk=", m_chunk); \
        } \
    }
    switch (bits) {
        case 2: DISPATCH_MC_KC(2); break;
        case 4: DISPATCH_MC_KC(4); break;
        default: TORCH_CHECK(false, "unreachable");
    }
#undef DISPATCH_MC_KC
#undef LAUNCH_SK
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm_v17_lib_splitk", &multiquant::xfp_gemm_v17_lib_splitk,
          "XFP-V2 Linear GEMM, split-K variant for K > 8192 "
          "(M_CHUNK ∈ {1,2,4}, K_CHUNK ∈ {2048,4096})");
}
