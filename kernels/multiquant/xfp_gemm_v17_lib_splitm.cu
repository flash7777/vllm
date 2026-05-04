// SPDX-License-Identifier: Apache-2.0
// XFP-V2 Linear GEMM v17_lib_splitm — split-M-internal variant.
//
// Companion to xfp_gemm_v17_lib.cu (split-N): handles M ≥ ~8 cases by
// processing M_CHUNK rows internally per block. Caller picks M_CHUNK
// adaptively per K (see Python dispatch).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core_v2_splitm.cuh"

namespace multiquant {

template <int BITS, int M_CHUNK>
static void launch_splitm(
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

    // Dynamic SMEM: M_CHUNK A-rows + library.
    size_t smem_bytes = (size_t)M_CHUNK * K * sizeof(__nv_bfloat16)
                      + (size_t)library_size * 16 * sizeof(__half);

    // sm_120 (Blackwell consumer/pro) sharedMemPerBlockOptin = 99 KB.
    // Set carveout to 96 KB (98304) — safe headroom under device limit.
    // Practical M_CHUNK × K combos:
    //   K=8192: M_CHUNK ≤ 4 (s_A=64 KB)  — LUT picks 2 (32 KB)
    //   K=4096: M_CHUNK ≤ 8 (s_A=64 KB)  — LUT picks 4 (32 KB)
    //   K=2048: M_CHUNK ≤ 16 (s_A=64 KB) — LUT picks 4 (16 KB)
    //   K=1024: M_CHUNK ≤ 16 (s_A=32 KB) — LUT picks 8 (16 KB)
    static bool s_carveout_set = false;
    if (!s_carveout_set) {
        cudaFuncSetAttribute(
            xfp_gemm_v2_splitm_kernel<BITS, M_CHUNK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        s_carveout_set = true;
    }
    TORCH_CHECK(smem_bytes <= 98304,
                "split-M: smem_bytes=", smem_bytes,
                " exceeds 96 KB carveout. M_CHUNK=", M_CHUNK, " K=", K,
                " too large for sm_120.");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_v2_splitm_kernel<BITS, M_CHUNK>
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

void xfp_gemm_v17_lib_splitm(
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
    int64_t m_chunk)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && library.is_cuda() &&
                group_lib_id.is_cuda() && group_scale.is_cuda() &&
                group_mid.is_cuda() && C.is_cuda(),
                "xfp_gemm_v17_lib_splitm: all tensors must be CUDA");
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
    TORCH_CHECK(bits == 4, "split-M v1: only BITS=4 supported");
    TORCH_CHECK(group_size == 128,
                "split-M v1: only group_size=128 supported, got ", group_size);
    TORCH_CHECK(K % group_size == 0, "K not divisible by group_size");

    int M = static_cast<int>(A.size(0));
    int N = static_cast<int>(group_scale.size(0));
    int G = static_cast<int>(group_scale.size(1));
    int library_size = static_cast<int>(library.size(0));
    int K_packed = (static_cast<int>(K) + 7) / 8;  // bits=4 → 8 vals/word

    TORCH_CHECK(library_size <= XFP_V2_LIBRARY_MAX, "library_size > MAX");

    int kK = static_cast<int>(K);
    int gs = static_cast<int>(group_size);

    switch (m_chunk) {
        case 2:
            launch_splitm<4, 2>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, kK, N, K_packed, G, gs, library_size, M);
            break;
        case 4:
            launch_splitm<4, 4>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, kK, N, K_packed, G, gs, library_size, M);
            break;
        case 8:
            launch_splitm<4, 8>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, kK, N, K_packed, G, gs, library_size, M);
            break;
        case 16:
            launch_splitm<4, 16>(A, B_packed, library, group_lib_id, group_scale,
                                 group_mid, C, kK, N, K_packed, G, gs, library_size, M);
            break;
        default:
            TORCH_CHECK(false, "unsupported m_chunk: ", m_chunk,
                        " (supported: 2, 4, 8, 16)");
    }
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm_v17_lib_splitm", &multiquant::xfp_gemm_v17_lib_splitm,
          "XFP-V2 Linear GEMM, split-M-internal variant (M_CHUNK ∈ {2,4,8,16})");
}
