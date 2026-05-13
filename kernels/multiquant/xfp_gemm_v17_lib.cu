// SPDX-License-Identifier: Apache-2.0
// XFP-V2 Linear GEMM v17_lib — per-group + shared codebook library.
//
// Derived from xfp_gemm_v12.cu. V1 stays unchanged; this kernel runs
// the V2 forward path (group_size=128, library_size≤64, BITS=4) on top
// of the same A-row SMEM cache pattern.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core_v2.cuh"

namespace multiquant {

template <int BITS>
static void launch_linear_v2(
    torch::Tensor A, torch::Tensor B_packed,
    torch::Tensor library, torch::Tensor group_lib_id,
    torch::Tensor group_scale, torch::Tensor group_mid,
    torch::Tensor C, int K, int N, int K_packed,
    int G, int group_size, int library_size)
{
    int M = static_cast<int>(A.size(0));

    dim3 block(XFP_BLOCK_SIZE);
    dim3 grid(
        (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
        M
    );

    LinearPolicyV2::Params params{M};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dynamic SMEM: s_A (K * bf16) + s_library (library_size * 16 * fp16).
    constexpr int LUT = (1 << BITS);  // bits-dependent codebook size
    size_t smem_bytes = static_cast<size_t>(K) * sizeof(__nv_bfloat16)
                      + static_cast<size_t>(library_size) * LUT * sizeof(__half);

    // sm_120/121 default SMEM-per-block carveout is 48 KB — opt in to the
    // 99 KB sharedMemPerBlockOptin via cudaFuncSetAttribute. Required for
    // K > 24320 (e.g. K=32768 hypothetical 397B Linear). One-time set is
    // sticky per kernel function. For low-K calls (K=2048, smem=5 KB)
    // occupancy is computed from the actual smem_bytes argument, so this
    // setting does not regress small-K throughput.
    static bool s_attr_set = false;
    if (!s_attr_set) {
        cudaFuncSetAttribute(
            xfp_gemm_v2_templated_kernel<BITS, LinearPolicyV2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        s_attr_set = true;
    }
    TORCH_CHECK(smem_bytes <= 98304,
                "xfp_gemm_v17_lib: smem_bytes=", smem_bytes,
                " exceeds 96 KB carveout (K=", K, ")");

    xfp_gemm_v2_templated_kernel<BITS, LinearPolicyV2>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(library.data_ptr()),
            group_lib_id.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(group_scale.data_ptr()),
            reinterpret_cast<const half*>(group_mid.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, K_packed, G, group_size, library_size, params);
}

void xfp_gemm_v17_lib(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor library,        // [L, LUT_SIZE] fp16
    torch::Tensor group_lib_id,   // [N, G] int32
    torch::Tensor group_scale,    // [N, G] fp16
    torch::Tensor group_mid,      // [N, G] fp16
    torch::Tensor C,
    int64_t bits,
    int64_t K,
    int64_t group_size)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() &&
                library.is_cuda() && group_lib_id.is_cuda() &&
                group_scale.is_cuda() && group_mid.is_cuda() &&
                C.is_cuda(),
                "xfp_gemm_v17_lib: all tensors must be CUDA");
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
    TORCH_CHECK(K % 2 == 0,
                "xfp_gemm_v17_lib: K must be even (bf162 vector load), got K=", K);
    TORCH_CHECK(K <= LinearPolicyV2::K_SMEM_MAX,
                "xfp_gemm_v17_lib: K=", K, " exceeds K_SMEM_MAX=",
                LinearPolicyV2::K_SMEM_MAX);
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4,
                "xfp_gemm_v17_lib: only BITS=2/3/4 supported, got ", bits);
    TORCH_CHECK(group_size == 128,
                "xfp_gemm_v17_lib v1: only group_size=128 supported, got ",
                group_size);
    TORCH_CHECK(K % group_size == 0,
                "xfp_gemm_v17_lib: K=", K, " not divisible by group_size=",
                group_size);

    int N = static_cast<int>(group_scale.size(0));
    int G = static_cast<int>(group_scale.size(1));
    int library_size = static_cast<int>(library.size(0));
    int vals_per_word = (bits == 2) ? 16 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    TORCH_CHECK(library_size <= XFP_V2_LIBRARY_MAX,
                "xfp_gemm_v17_lib: library_size=", library_size,
                " exceeds XFP_V2_LIBRARY_MAX=", XFP_V2_LIBRARY_MAX);
    TORCH_CHECK(group_lib_id.size(0) == N && group_lib_id.size(1) == G,
                "group_lib_id shape mismatch: expected [N=", N, ", G=", G,
                "], got [", group_lib_id.size(0), ", ", group_lib_id.size(1), "]");

    switch (bits) {
        case 2:
            launch_linear_v2<2>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, static_cast<int>(K), N, K_packed,
                                G, static_cast<int>(group_size), library_size);
            break;
        case 3:
            launch_linear_v2<3>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, static_cast<int>(K), N, K_packed,
                                G, static_cast<int>(group_size), library_size);
            break;
        case 4:
            launch_linear_v2<4>(A, B_packed, library, group_lib_id, group_scale,
                                group_mid, C, static_cast<int>(K), N, K_packed,
                                G, static_cast<int>(group_size), library_size);
            break;
        default:
            TORCH_CHECK(false, "unreachable");
    }
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm_v17_lib", &multiquant::xfp_gemm_v17_lib,
          "XFP-V2 Linear GEMM with shared codebook library + per-group "
          "scale/mid (group_size=128, BITS=4)");
}
