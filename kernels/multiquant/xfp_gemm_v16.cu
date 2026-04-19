// SPDX-License-Identifier: Apache-2.0
// XFP Linear GEMM v16 — Tensor Core MMA m16n8k16 with XFP-codebook B-decode.
//
// Same API as v11/v12 (A[M,K] × B_xfp_packed → C[M,N]) but the hot path uses
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for compute instead
// of scalar FMAs. For each (M-tile, N-tile) block:
//
//   1. Load 16×16 bf16 A-fragment cooperatively via scalar loads (ldmatrix
//      would be cleaner but scalar works and lets us focus on correctness
//      first; swap later).
//   2. Decode one 16×8 bf16 B-tile per K=16 step, from XFP packed+codebook
//      via SHFL codebook lookup (same algorithm as v10/v11/v12 inner loop,
//      just writing into the MMA m16n8k16 B-layout in SMEM instead of a
//      per-slot scalar FMA).
//   3. Run one mma.m16n8k16 per K=16 step → 4 fp32 accumulator slots/lane.
//   4. Cast fp32 → bf16 at end, store C.
//
// Grid shape: (ceil(N/8), ceil(M/16)). One warp per block (32 threads).
// M < 16 is padded to 16 internally (caller sees their M, only rows 0..M-1
// of C are written).
//
// K constraint: K must be a multiple of 16 (MMA K-tile) and K ≤ the XFP
// packed format's WARP_SIZE * VALS_PER_WORD = 32*8 = 256 per group. For
// larger K we walk multiple groups.
//
// Bitwise identical to v11 is NOT expected — Tensor-Core FP32 accumulation
// has different rounding than scalar FMA accumulation. Correctness gate is
// cos ≥ 0.9999 vs v11 reference (same bar Marlin meets vs reference).

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace xfp_v16 {

__device__ __forceinline__ uint32_t bf162_as_uint(__nv_bfloat162 v) {
    uint32_t r;
    asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(*(uint32_t*)&v));
    return r;
}

#define WARP_SIZE 32
#define M_TILE 16
#define N_TILE 8
#define K_TILE 16  // MMA K dimension

template <int BITS>
__global__ void xfp_gemm_v16_kernel(
    const __nv_bfloat16* __restrict__ A,    // [M, K] row-major
    const uint32_t* __restrict__ B_packed,   // flat repacked, v11/v12 layout
    const half* __restrict__ codebook,       // [N, LUT_SIZE] fp16
    __nv_bfloat16* __restrict__ C,           // [M, N] row-major
    int M, int N, int K, int K_packed)
{
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : (BITS == 3) ? 10 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    // Block handles one (M_TILE, N_TILE) output tile.
    int n_block = blockIdx.x;       // N-tile index (0..N/8-1)
    int m_block = blockIdx.y;       // M-tile index (0..M/16-1)
    int n_base = n_block * N_TILE;  // starting N col for this block
    int m_base = m_block * M_TILE;  // starting M row for this block
    int lane = threadIdx.x;         // 0..31

    if (n_base >= N) return;

    // Codebook slice [N_TILE, LUT_SIZE] into SMEM. For bits=4 this is 8×16=128 u32 = 512 bytes.
    __shared__ float s_cb[N_TILE * LUT_SIZE];
    if (lane < LUT_SIZE) {
        #pragma unroll
        for (int n = 0; n < N_TILE; n++) {
            int n_col = n_base + n;
            s_cb[n * LUT_SIZE + lane] =
                (n_col < N)
                    ? __half2float(codebook[n_col * LUT_SIZE + lane])
                    : 0.0f;
        }
    }
    __syncwarp();

    // Accumulator — 4 fp32 per lane for m16n8k16.
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Scratch SMEM for B-tile in MMA layout: 16 K-rows × 8 N-cols bf16 = 256 bytes.
    __shared__ __nv_bfloat16 s_B[M_TILE * N_TILE];

    // Loop over K in K_TILE chunks.
    for (int k = 0; k < K; k += K_TILE) {
        // ── Decode 16×8 B-tile into s_B ──
        // Each lane writes 4 bf16 into s_B (128 values / 32 lanes = 4).
        // Lane l handles k_row = l/2, n_cols = (l%2)*4 .. (l%2)*4+3.
        int l_krow = lane / 2;              // 0..15
        int l_ncol_base = (lane % 2) * 4;    // 0 or 4
        int k_abs = k + l_krow;

        #pragma unroll
        for (int dn = 0; dn < 4; dn++) {
            int n_col_local = l_ncol_base + dn;
            int n_col_global = n_base + n_col_local;
            __nv_bfloat16 w_bf;
            if (k_abs < K && n_col_global < N) {
                int kw_global = k_abs / VALS_PER_WORD;
                int slot = k_abs % VALS_PER_WORD;
                int g = kw_global / WARP_SIZE;
                int kw_in_group = kw_global % WARP_SIZE;
                int n_off = n_col_global * WARP_SIZE + kw_in_group;
                // Global read of one packed uint32; can't coalesce across
                // lanes because each lane reads a different n_col/kw combo
                // — that's fine for a POC, optimise later with cp.async+SMEM.
                int offset = g * N * WARP_SIZE + n_off;
                uint32_t packed = (kw_global < K_packed)
                    ? B_packed[offset] : 0u;
                int idx = (int)((packed >> (slot * BITS)) & MASK);
                float w = s_cb[n_col_local * LUT_SIZE + idx];
                w_bf = __float2bfloat16(w);
            } else {
                w_bf = __float2bfloat16(0.0f);
            }
            s_B[l_krow * N_TILE + n_col_local] = w_bf;
        }
        __syncwarp();

        // ── Load A-fragment [16, 16] ──
        uint32_t a_frag[4];
        {
            int row_low  = lane / 4;
            int row_high = row_low + 8;
            int col_base = (lane % 4) * 2;
            auto Aelem = [&] (int row, int col) -> __nv_bfloat16 {
                int r = m_base + row;
                int c = k + col;
                return (r < M && c < K)
                    ? A[r * K + c]
                    : __float2bfloat16(0.0f);
            };
            __nv_bfloat16 a0 = Aelem(row_low,  col_base);
            __nv_bfloat16 a1 = Aelem(row_low,  col_base + 1);
            __nv_bfloat16 a2 = Aelem(row_high, col_base);
            __nv_bfloat16 a3 = Aelem(row_high, col_base + 1);
            __nv_bfloat16 a4 = Aelem(row_low,  col_base + 8);
            __nv_bfloat16 a5 = Aelem(row_low,  col_base + 9);
            __nv_bfloat16 a6 = Aelem(row_high, col_base + 8);
            __nv_bfloat16 a7 = Aelem(row_high, col_base + 9);
            a_frag[0] = bf162_as_uint(__nv_bfloat162(a0, a1));
            a_frag[1] = bf162_as_uint(__nv_bfloat162(a2, a3));
            a_frag[2] = bf162_as_uint(__nv_bfloat162(a4, a5));
            a_frag[3] = bf162_as_uint(__nv_bfloat162(a6, a7));
        }

        // ── Load B-fragment [16, 8] from s_B ──
        uint32_t b_frag[2];
        {
            int col = lane / 4;                   // 0..7
            int row_base = (lane % 4) * 2;         // 0, 2, 4, 6
            __nv_bfloat16 b0 = s_B[(row_base + 0) * N_TILE + col];
            __nv_bfloat16 b1 = s_B[(row_base + 1) * N_TILE + col];
            __nv_bfloat16 b2 = s_B[(row_base + 8) * N_TILE + col];
            __nv_bfloat16 b3 = s_B[(row_base + 9) * N_TILE + col];
            b_frag[0] = bf162_as_uint(__nv_bfloat162(b0, b1));
            b_frag[1] = bf162_as_uint(__nv_bfloat162(b2, b3));
        }

        // ── MMA ──
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_frag[0]), "r"(a_frag[1]),
              "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1])
        );
        __syncwarp();
    }

    // ── Store C-fragment [16, 8] (m16n8k16 C-layout) ──
    // Layout per lane:
    //   c[0], c[1] go to rows (lane/4),   cols (lane%4)*2 + {0,1}
    //   c[2], c[3] go to rows (lane/4)+8, cols (lane%4)*2 + {0,1}
    int c_row_lo = lane / 4;
    int c_row_hi = c_row_lo + 8;
    int c_col_base = (lane % 4) * 2;
    auto store = [&](int row, int col, float v) {
        int r = m_base + row;
        int c = n_base + col;
        if (r < M && c < N) {
            C[r * N + c] = __float2bfloat16(v);
        }
    };
    store(c_row_lo, c_col_base,     acc[0]);
    store(c_row_lo, c_col_base + 1, acc[1]);
    store(c_row_hi, c_col_base,     acc[2]);
    store(c_row_hi, c_col_base + 1, acc[3]);
}

void xfp_gemm(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor codebook,
    torch::Tensor C,
    int64_t bits,
    int64_t K)
{
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat16, "codebook must be float16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(K % K_TILE == 0,
                "v16: K must be multiple of 16 (MMA K-tile), got K=", K);

    int M = static_cast<int>(A.size(0));
    int N = static_cast<int>(codebook.size(0));
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");

    int vpw = (bits == 2) ? 16 : (bits == 3) ? 10 : 8;
    int K_packed = (static_cast<int>(K) + vpw - 1) / vpw;

    dim3 grid((N + N_TILE - 1) / N_TILE,
              (M + M_TILE - 1) / M_TILE);
    dim3 block(WARP_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (bits == 4) {
        xfp_gemm_v16_kernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(codebook.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            M, N, static_cast<int>(K), K_packed);
    } else if (bits == 3) {
        xfp_gemm_v16_kernel<3><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(codebook.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            M, N, static_cast<int>(K), K_packed);
    } else if (bits == 2) {
        xfp_gemm_v16_kernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
            reinterpret_cast<const uint32_t*>(B_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(codebook.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            M, N, static_cast<int>(K), K_packed);
    } else {
        TORCH_CHECK(false, "v16: unsupported bits=", bits);
    }
}

}  // namespace xfp_v16

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfp_gemm", &xfp_v16::xfp_gemm,
          "XFP v16: Tensor-Core MMA m16n8k16 with XFP codebook B-decode");
}
