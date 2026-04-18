// SPDX-License-Identifier: Apache-2.0
// XFP Linear GEMM v13 — v12 SMEM A-row cache + v11_async cp.async B prefetch.
//
// Combines both core orthogonal optimizations:
//   XFP_CORE_USE_SMEM_A   — block-uniform A-row cooperatively loaded once
//                           into __shared__ s_A[K_SMEM_MAX]; all 8 warps
//                           read A from SMEM (v12).
//   XFP_CORE_USE_CPASYNC  — double-buffered cp.async prefetch of B_packed
//                           groups; issue gi+1 while computing gi (v11_async).
//
// Same template as v11/v12 → bitwise identical output, just with B-prefetch
// overlapping A-SMEM-backed slot compute.
//
// K constraint inherited from v12 (LinearPolicy::K_SMEM_MAX = 8192). For
// larger K, dispatch falls back to v11 (the sync, per-warp-global path).

#define XFP_CORE_USE_SMEM_A
#define XFP_CORE_USE_CPASYNC
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core.cuh"

namespace multiquant {

template <int BITS>
static void launch_linear(
    torch::Tensor A, torch::Tensor B_packed, torch::Tensor codebook,
    torch::Tensor C, int K, int N, int K_packed)
{
    int M = static_cast<int>(A.size(0));

    dim3 block(XFP_BLOCK_SIZE);
    dim3 grid(
        (N + XFP_WARPS_PER_BLOCK - 1) / XFP_WARPS_PER_BLOCK,
        M
    );

    LinearPolicy::Params params{M};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    xfp_gemm_templated_kernel<BITS, LinearPolicy><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K, K_packed, params);
}

void xfp_gemm(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor codebook,
    torch::Tensor C,
    int64_t bits,
    int64_t K)
{
    TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() &&
                codebook.is_cuda() && C.is_cuda(),
                "xfp_gemm v13: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(K % 2 == 0,
                "xfp_gemm v13: K must be even (bf162 vector load), got K=", K);
    TORCH_CHECK(K <= LinearPolicy::K_SMEM_MAX,
                "xfp_gemm v13: K=", K, " exceeds K_SMEM_MAX=",
                LinearPolicy::K_SMEM_MAX, " — use v11 for this shape");

    int N = static_cast<int>(codebook.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;

    if (bits == 2)      launch_linear<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_linear<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_linear<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm v13: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v13: SMEM A-row cache + cp.async B prefetch");
}
