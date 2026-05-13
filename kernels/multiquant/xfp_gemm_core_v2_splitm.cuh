// SPDX-License-Identifier: Apache-2.0
// XFP-V2 GEMM core — split-M-internal variant.
//
// Difference vs xfp_gemm_core_v2.cuh:
//   - Grid: (ceil(N/W), ceil(M/M_CHUNK))
//   - Each block handles W N-rows × M_CHUNK M-rows internally
//   - Each lane keeps M_CHUNK accumulators (vs 1 in split-N)
//   - Codebook rebuild is amortized over M_CHUNK fmaf-pairs per inner slot
//
// Rationale: at M=64, the split-N kernel rebuilds the same codebook 64×
// per N-row (once per (m,n) block). Split-M-internal builds it once per
// (m_chunk, n)-block and reuses across M_CHUNK rows.
//
// Compile-time constraints:
//   - BITS == 4 (LUT_SIZE = 16). bits=2/3 deferred.
//   - M_CHUNK ∈ {2, 4, 8} (instantiated). Higher M_CHUNK risks reg spill;
//     check with -Xptxas=-v before adding 16.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace multiquant {

#ifndef XFP_WARP_SIZE
#define XFP_WARP_SIZE 32
#endif

#ifndef XFP_WARPS_PER_BLOCK
#define XFP_WARPS_PER_BLOCK 8
#endif

#define XFP_BLOCK_SIZE (XFP_WARP_SIZE * XFP_WARPS_PER_BLOCK)

#ifndef XFP_V2_LIBRARY_MAX
#define XFP_V2_LIBRARY_MAX 64
#endif


template <int BITS, int M_CHUNK>
__global__ void xfp_gemm_v2_splitm_kernel(
    const __nv_bfloat16* __restrict__ A,           // [M_total, K]
    const uint32_t*      __restrict__ B_packed,    // warp-interleaved
    const half*          __restrict__ library,     // [L, LUT_SIZE]
    const int32_t*       __restrict__ group_lib_id,// [N, G]
    const half*          __restrict__ group_scale, // [N, G]
    const half*          __restrict__ group_mid,   // [N, G]
    __nv_bfloat16*       __restrict__ C,           // [M_total, N]
    int N, int K, int K_packed,
    int G, int group_size, int library_size,
    int M_total)
{
    static_assert(BITS == 2 || BITS == 4,
                  "BITS must be 2, 3, or 4");
    static_assert(M_CHUNK >= 2 && M_CHUNK <= 16,
                  "M_CHUNK ∈ [2, 16] supported");
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / XFP_WARP_SIZE;
    int lane    = threadIdx.x % XFP_WARP_SIZE;

    // GROUP_SIZE hardcoded to 128 (host TORCH_CHECK enforces this).
    // For BITS=4: VALS_PER_WORD=8, LANES_PER_GROUP=16 (=LUT_SIZE, coincidence).
    // For BITS=2: VALS_PER_WORD=16, LANES_PER_GROUP=8 (<LUT_SIZE — lane
    // duplicates within each codebook subgroup; shuffle uses
    // cb_lane_offset+idx, only first LUT_SIZE lanes hold distinct values).
    constexpr int GROUP_SIZE      = 128;
    constexpr int LANES_PER_GROUP = GROUP_SIZE / VALS_PER_WORD;
    constexpr int CB_PER_ITER     = XFP_WARP_SIZE / LANES_PER_GROUP;
    const int my_cb_idx           = lane % LUT_SIZE;
    const int lane_group          = lane / LANES_PER_GROUP;
    const int cb_lane_offset      = lane_group * LANES_PER_GROUP;

    // ── Block coords ──
    const int m_base = blockIdx.y * M_CHUNK;
    const int n      = blockIdx.x * XFP_WARPS_PER_BLOCK + warp_id;

    // M_actual: number of valid rows in this M-chunk (≤ M_CHUNK at tail)
    int m_actual = M_total - m_base;
    if (m_actual <= 0) return;
    if (m_actual > M_CHUNK) m_actual = M_CHUNK;

    // ── Dynamic SMEM ──
    // [0]            s_A         : M_CHUNK * K * sizeof(bf16)
    // [s_A end]      s_library   : library_size * LUT_SIZE * sizeof(half)
    extern __shared__ char xfp_v2_splitm_smem[];
    __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(xfp_v2_splitm_smem);
    half*          s_library = reinterpret_cast<half*>(
        xfp_v2_splitm_smem + (size_t)M_CHUNK * K * sizeof(__nv_bfloat16));

    // ── Cooperative library load ──
    {
        int total = library_size * LUT_SIZE;
        for (int i = threadIdx.x; i < total; i += XFP_BLOCK_SIZE) {
            s_library[i] = library[i];
        }
    }
    // ── Cooperative load of M_CHUNK A-rows ──
    // Layout in SMEM: row m at offset m*K. Tail rows beyond m_actual are
    // zero-initialized so the inner loop can unconditionally read M_CHUNK.
    {
        int pair_count = K >> 1;
        for (int m = 0; m < M_CHUNK; m++) {
            __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(
                s_A + (size_t)m * K);
            if (m < m_actual) {
                const __nv_bfloat162* src2 =
                    reinterpret_cast<const __nv_bfloat162*>(
                        A + (size_t)(m_base + m) * K);
                #pragma unroll 4
                for (int i = threadIdx.x; i < pair_count; i += XFP_BLOCK_SIZE) {
                    dst2[i] = src2[i];
                }
            } else {
                __nv_bfloat162 zero2 = __floats2bfloat162_rn(0.0f, 0.0f);
                #pragma unroll 4
                for (int i = threadIdx.x; i < pair_count; i += XFP_BLOCK_SIZE) {
                    dst2[i] = zero2;
                }
            }
        }
    }
    __syncthreads();

    if (n >= N) return;

    // ── Per-row metadata pointers ──
    const int32_t* row_lib_id = group_lib_id + (size_t)n * G;
    const half*    row_scale  = group_scale  + (size_t)n * G;
    const half*    row_mid    = group_mid    + (size_t)n * G;

    // ── M_CHUNK accumulators per lane ──
    float acc[M_CHUNK];
    #pragma unroll
    for (int m = 0; m < M_CHUNK; m++) acc[m] = 0.0f;

    int n_offset = n * XFP_WARP_SIZE + lane;
    int K_packed_safe = K / VALS_PER_WORD;
    int n_full_groups = K_packed_safe / XFP_WARP_SIZE;

    float my_cb_val = 0.0f;

    for (int gi = 0; gi < n_full_groups; gi++) {
        // Codebook rebuild — ONCE per gi (amortized over M_CHUNK acc updates)
        int my_group_idx = gi * CB_PER_ITER + lane_group;
        int   lib_id  = (int) row_lib_id[my_group_idx];
        float scale_f = __half2float(row_scale[my_group_idx]);
        float mid_f   = __half2float(row_mid  [my_group_idx]);
        my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx])
                    * scale_f + mid_f;

        int kw = lane + gi * XFP_WARP_SIZE;
        uint32_t packed = B_packed[gi * N * XFP_WARP_SIZE + n_offset];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            int idx0 = (int)((packed >> (slot * BITS)) & MASK);
            int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
            float w0 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx0);
            float w1 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx1);

            // M_CHUNK fmaf pairs amortizing the rebuild + SHFL above
            #pragma unroll
            for (int m = 0; m < M_CHUNK; m++) {
                __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
                    s_A + (size_t)m * K + k_base + slot);
                float a0 = __bfloat162float(__low2bfloat16(a2));
                float a1 = __bfloat162float(__high2bfloat16(a2));
                acc[m] = fmaf(w0, a0, acc[m]);
                acc[m] = fmaf(w1, a1, acc[m]);
            }
        }
    }

    // ── Warp reduction × M_CHUNK ──
    #pragma unroll
    for (int m = 0; m < M_CHUNK; m++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    // ── Epilogue: lane 0 writes M_CHUNK outputs (mask tail rows) ──
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_CHUNK; m++) {
            int actual_m = m_base + m;
            if (actual_m < M_total) {
                C[(size_t)actual_m * N + n] = __float2bfloat16(acc[m]);
            }
        }
    }
}

}  // namespace multiquant
