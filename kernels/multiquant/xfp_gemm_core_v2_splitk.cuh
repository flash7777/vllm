// SPDX-License-Identifier: Apache-2.0
// XFP-V2 GEMM core — split-K variant (V3 reference).
//
// Difference vs xfp_gemm_core_v2_splitm.cuh:
//   - Adds an outer K-chunk loop wrapping the group iteration
//   - SMEM A-cache holds only K_CHUNK columns (vs full K in splitm)
//   - Accumulators persist across K-chunks (per-warp registers)
//   - Library + output reduction unchanged
//
// Rationale: workstation Blackwell SMs (sm_120/sm_121, 99 KB SMEM/CTA)
// cap K_SMEM_MAX at 8192. For models with K > 8192 (GLM-4.7-Flash 10240,
// GLM-5 12288, Qwen3.6-27B 17408), this kernel processes the K dimension
// in K_CHUNK-sized slices, reloading SMEM per chunk.
//
// Compile-time constraints:
//   - BITS == 4 (LUT_SIZE = 16). bits=2/3 deferred.
//   - K_CHUNK ∈ {2048, 4096} (must be multiple of 256 = 2 groups × group_size 128)
//   - M_CHUNK ∈ {1, 2, 4} (instantiated). Higher M_CHUNK with K_CHUNK=4096
//     uses 32 KB SMEM (M_CHUNK=4 × 4096 × bf16); M_CHUNK=8 + K_CHUNK=4096
//     would need 64 KB (within 99 KB carveout but tighter occupancy).
//
// Performance: 10–20% slower than v17_lib_splitm at K=8192 due to
// the chunked A-reload; the overhead is the price of supporting K > 8192.

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


template <int BITS, int M_CHUNK, int K_CHUNK>
__global__ void xfp_gemm_v2_splitk_kernel(
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
    static_assert(M_CHUNK >= 1 && M_CHUNK <= 4,
                  "M_CHUNK ∈ [1, 4] supported (split-K halves SMEM headroom)");
    static_assert(K_CHUNK == 2048 || K_CHUNK == 4096,
                  "K_CHUNK ∈ {2048, 4096} supported (multiple of 256)");
    constexpr int VALS_PER_WORD  = 8;
    constexpr uint32_t MASK      = 0x0fu;
    constexpr int LUT_SIZE       = (1 << BITS);
    constexpr int K_PACKED_CHUNK = K_CHUNK / VALS_PER_WORD;  // 256 or 512

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

    // M_actual: number of valid rows in this M-chunk
    int m_actual = M_total - m_base;
    if (m_actual <= 0) return;
    if (m_actual > M_CHUNK) m_actual = M_CHUNK;

    // ── Dynamic SMEM ──
    // [0]            s_A         : M_CHUNK * K_CHUNK * sizeof(bf16)
    // [s_A end]      s_library   : library_size * LUT_SIZE * sizeof(half)
    extern __shared__ char xfp_v2_splitk_smem[];
    __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(xfp_v2_splitk_smem);
    half*          s_library = reinterpret_cast<half*>(
        xfp_v2_splitk_smem + (size_t)M_CHUNK * K_CHUNK * sizeof(__nv_bfloat16));

    // ── Cooperative library load (once per block) ──
    {
        int total = library_size * LUT_SIZE;
        for (int i = threadIdx.x; i < total; i += XFP_BLOCK_SIZE) {
            s_library[i] = library[i];
        }
    }
    __syncthreads();

    // n_full_groups: total iter pairs across full K (CB_PER_ITER groups per iter)
    int n_full_groups_total = K_packed / XFP_WARP_SIZE;

    // ── Per-row metadata pointers (set up once per warp) ──
    const int32_t* row_lib_id = nullptr;
    const half*    row_scale  = nullptr;
    const half*    row_mid    = nullptr;
    if (n < N) {
        row_lib_id = group_lib_id + (size_t)n * G;
        row_scale  = group_scale  + (size_t)n * G;
        row_mid    = group_mid    + (size_t)n * G;
    }

    // ── M_CHUNK accumulators per lane (persist across K-chunks!) ──
    float acc[M_CHUNK];
    #pragma unroll
    for (int m = 0; m < M_CHUNK; m++) acc[m] = 0.0f;

    int n_offset = n * XFP_WARP_SIZE + lane;
    const int n_chunks = (K + K_CHUNK - 1) / K_CHUNK;

    // ════════════════════════════════════════════════════════════════
    // Outer K-chunk loop
    // ════════════════════════════════════════════════════════════════
    for (int kc = 0; kc < n_chunks; kc++) {
        const int k_offset      = kc * K_CHUNK;
        const int k_size_actual = (k_offset + K_CHUNK <= K) ? K_CHUNK : (K - k_offset);
        // gi iter range covered by this K-chunk:
        const int gi_start = (k_offset / VALS_PER_WORD) / XFP_WARP_SIZE;
        const int gi_end   = ((k_offset + k_size_actual) / VALS_PER_WORD + XFP_WARP_SIZE - 1)
                                / XFP_WARP_SIZE;
        // Clamp gi_end to global limit (K may not align to K_CHUNK)
        const int gi_end_clamped = (gi_end < n_full_groups_total) ? gi_end : n_full_groups_total;

        // ── Cooperative A-chunk load ──
        // Layout: s_A[m * K_CHUNK + (k - k_offset)]
        // Tail rows beyond m_actual zero-initialized; tail K beyond k_size_actual zero
        {
            int pair_count = K_CHUNK >> 1;
            for (int m = 0; m < M_CHUNK; m++) {
                __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(
                    s_A + (size_t)m * K_CHUNK);
                if (m < m_actual) {
                    const __nv_bfloat162* src2 =
                        reinterpret_cast<const __nv_bfloat162*>(
                            A + (size_t)(m_base + m) * K + k_offset);
                    int valid_pairs = k_size_actual >> 1;
                    #pragma unroll 4
                    for (int i = threadIdx.x; i < pair_count; i += XFP_BLOCK_SIZE) {
                        if (i < valid_pairs) {
                            dst2[i] = src2[i];
                        } else {
                            dst2[i] = __floats2bfloat162_rn(0.0f, 0.0f);
                        }
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

        if (n >= N) {
            __syncthreads();  // keep all warps in lock-step on next iter A-load
            continue;
        }

        // ── Inner gi loop, restricted to this K-chunk ──
        for (int gi = gi_start; gi < gi_end_clamped; gi++) {
            int my_group_idx = gi * CB_PER_ITER + lane_group;
            int   lib_id  = (int) row_lib_id[my_group_idx];
            float scale_f = __half2float(row_scale[my_group_idx]);
            float mid_f   = __half2float(row_mid  [my_group_idx]);
            float my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx])
                                * scale_f + mid_f;

            int kw = lane + gi * XFP_WARP_SIZE;
            uint32_t packed = B_packed[gi * N * XFP_WARP_SIZE + n_offset];
            int k_base_global = kw * VALS_PER_WORD;
            int k_base_chunk  = k_base_global - k_offset;  // index into s_A

            #pragma unroll
            for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
                int idx0 = (int)((packed >> (slot * BITS)) & MASK);
                int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
                float w0 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx0);
                float w1 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx1);

                #pragma unroll
                for (int m = 0; m < M_CHUNK; m++) {
                    __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
                        s_A + (size_t)m * K_CHUNK + k_base_chunk + slot);
                    float a0 = __bfloat162float(__low2bfloat16(a2));
                    float a1 = __bfloat162float(__high2bfloat16(a2));
                    acc[m] = fmaf(w0, a0, acc[m]);
                    acc[m] = fmaf(w1, a1, acc[m]);
                }
            }
        }
        __syncthreads();  // before next chunk's A-load overwrites SMEM
    }

    if (n >= N) return;

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
