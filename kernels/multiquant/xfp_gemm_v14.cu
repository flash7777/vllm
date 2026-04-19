// SPDX-License-Identifier: Apache-2.0
// XFP Linear GEMM v14 — SMEM A-row cache (v12) + bulk cp.async B preload.
//
// The v13 per-group cp.async pipeline regressed on SM121 because the small
// transfer size (4 B/lane) with a wait_group between every K-group
// serialized the slot compute. v14 tests whether the REVERSE tradeoff —
// issue all n_groups of B up front, ONE commit, ONE wait, then a pure-SMEM
// K-loop — recovers the lost headroom. No TMA yet (that's v15); this just
// isolates the "bulk load vs streamed prefetch" signal.
//
// SMEM per block: 16 KiB (s_A, bf16) + 16 KiB (s_B, u32) = 32 KiB. Safe
// under SM121's 48 KiB/block limit. K cap of 4096 (bits=4) imposed
// host-side; larger K must use v11 fallback.

#define XFP_CORE_USE_SMEM_A
#define XFP_CORE_USE_CPASYNC_BULK
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "xfp_gemm_core.cuh"

namespace multiquant {

static constexpr int V14_K_MAX_LINEAR = 4096;  // bits=4 cap

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
                "xfp_gemm v14: all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(K % 2 == 0,
                "xfp_gemm v14: K must be even (bf162 vector load), got K=", K);
    TORCH_CHECK(K <= LinearPolicy::K_SMEM_MAX,
                "xfp_gemm v14: K=", K, " exceeds s_A K_SMEM_MAX=",
                LinearPolicy::K_SMEM_MAX);

    int N = static_cast<int>(codebook.size(0));
    int vals_per_word = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vals_per_word - 1) / vals_per_word;
    int n_groups = (K_packed + XFP_WARP_SIZE - 1) / XFP_WARP_SIZE;
    TORCH_CHECK(n_groups <= 16,
                "xfp_gemm v14: n_groups=", n_groups,
                " exceeds 16 (B-SMEM cap). bits=", bits, " K=", K,
                " — use v11/v12 for this shape");

    if (bits == 2)      launch_linear<2>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 3) launch_linear<3>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else if (bits == 4) launch_linear<4>(A, B_packed, codebook, C, static_cast<int>(K), N, K_packed);
    else TORCH_CHECK(false, "xfp_gemm v14: unsupported bits=", bits);
}

}  // namespace multiquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &multiquant::xfp_gemm,
          "XFP v14: SMEM A-row cache + bulk cp.async B preload");
}
