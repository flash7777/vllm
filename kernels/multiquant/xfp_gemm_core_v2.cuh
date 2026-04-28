// SPDX-License-Identifier: Apache-2.0
// XFP-V2 GEMM core — per-group + shared codebook library variant.
//
// Derived from xfp_gemm_core.cuh (v12 pattern). V1 stays unchanged;
// this header is a parallel implementation for the V2 forward path so
// V1 hot-loop performance is fully preserved.
//
// Key differences vs V1 hot loop (BITS=4, GROUP_SIZE=128 sweet spot):
//   - Library is loaded once per block into __shared__ s_library
//     (LIBRARY_SIZE * LUT_SIZE half values, ~1 KB for {32, 16}).
//   - Each warp covers WARP_SIZE * VALS_PER_WORD = 256 weights per outer
//     iter; for GROUP_SIZE=128 that means 2 codebooks per outer iter
//     (lanes 0..15 → group A, lanes 16..31 → group B).
//   - Each lane reads its OWN group's lib_id, scale, mid directly from
//     global (coalesced 32-byte chunks per warp).
//   - my_cb_val = library[lib_id][lane%LUT_SIZE] * scale + mid.
//   - SHFL lookup uses (cb_lane_offset + idx) so each lane-subgroup
//     reads centroids from its own subgroup of lanes.
//
// Compile-time constraints:
//   - BITS == 4 in this v17_lib v1 (LUT_SIZE = 16). bits=2 / bits=3
//     paths are deferred to a later iteration.
//   - LIBRARY_SIZE ≤ 64 (SMEM budget cap, ~2 KB).

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

// SMEM budget cap for the library (independent of LIBRARY_SIZE arg).
// 64 prototypes × 16 cents × 2 B = 2 KB, comfortable for any tile.
#ifndef XFP_V2_LIBRARY_MAX
#define XFP_V2_LIBRARY_MAX 64
#endif


// ─── V2 inner loop template — Linear policy with library lookup ─────

template <int BITS, class PolicyV2>
__device__ __forceinline__ void xfp_gemm_core_v2(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t*      __restrict__ B_packed_base,
    const half*          __restrict__ library,            // [LIBRARY_SIZE * LUT_SIZE]
    const int32_t*       __restrict__ group_lib_id_base,   // [N * G] (per-row x per-group)
    const half*          __restrict__ group_scale_base,    // [N * G]
    const half*          __restrict__ group_mid_base,      // [N * G]
    __nv_bfloat16*       __restrict__ C,
    int N, int K, int K_packed,
    int G, int group_size, int library_size,
    typename PolicyV2::Params params)
{
    static_assert(BITS == 4,
                  "xfp_gemm_core_v2: only BITS=4 supported in v17_lib v1");
    constexpr int VALS_PER_WORD = 8;     // BITS=4
    constexpr uint32_t MASK = 0x0fu;
    constexpr int LUT_SIZE = 16;

    int warp_id = threadIdx.x / XFP_WARP_SIZE;
    int lane    = threadIdx.x % XFP_WARP_SIZE;

    // Lane-subgroup partitioning. For BITS=4, LUT_SIZE=16, WARP_SIZE=32:
    // 2 lane-subgroups (codebooks) per warp outer iter → CB_PER_ITER = 2.
    constexpr int CB_PER_ITER     = XFP_WARP_SIZE / LUT_SIZE;  // = 2
    const int my_cb_idx           = lane % LUT_SIZE;            // 0..15
    const int lane_group          = lane / LUT_SIZE;            // 0 or 1
    const int cb_lane_offset      = lane_group * LUT_SIZE;      // 0 or 16

    // ── Cooperative library load into SMEM ──
    __shared__ half s_library[XFP_V2_LIBRARY_MAX * LUT_SIZE];
    {
        int total = library_size * LUT_SIZE;
        for (int i = threadIdx.x; i < total; i += XFP_BLOCK_SIZE) {
            s_library[i] = library[i];
        }
    }
    // Block-uniform A-row load (mirrors v12). Library cooperative copy
    // and A-row copy share the same __syncthreads barrier below.
    const __nv_bfloat16* block_A = PolicyV2::block_A_row(A, K, params);
    if (block_A == nullptr) return;
    __shared__ __nv_bfloat16 s_A[PolicyV2::K_SMEM_MAX];
    {
        int pair_count = K >> 1;
        const __nv_bfloat162* __restrict__ src2 =
            reinterpret_cast<const __nv_bfloat162*>(block_A);
        __nv_bfloat162* dst2 =
            reinterpret_cast<__nv_bfloat162*>(s_A);
        #pragma unroll 4
        for (int i = threadIdx.x; i < pair_count; i += XFP_BLOCK_SIZE) {
            dst2[i] = src2[i];
        }
    }
    __syncthreads();

    auto ctx = PolicyV2::template prologue<BITS, LUT_SIZE>(
        A, B_packed_base, library, group_lib_id_base, group_scale_base,
        group_mid_base, N, K, warp_id, lane, params);
    if (!ctx.active) return;

    const __nv_bfloat16* __restrict__ A_src = s_A;

    float acc = 0.0f;
    int n_offset = ctx.n * XFP_WARP_SIZE + lane;

    // ── Per-warp SMEM cache for this row's group metadata ──
    // [WARPS_PER_BLOCK][G][3]  // (lib_id, scale_f, mid_f) per group.
    // Storing as float lets us avoid __half2float in the hot loop
    // and pack {scale, mid} as float32 too (cheap precision: scale/mid
    // come from fp16 source already).
    constexpr int META_PER_GROUP = 3;        // lib_id (as float), scale, mid
    constexpr int META_MAX_G     = PolicyV2::K_SMEM_MAX / 64;  // K/group_size cap (≥ G ever observed)
    __shared__ float s_meta[XFP_WARPS_PER_BLOCK * META_MAX_G * META_PER_GROUP];
    float* my_warp_meta = s_meta + warp_id * META_MAX_G * META_PER_GROUP;
    {
        const int32_t* row_lib_id = ctx.group_lib_id + (size_t)ctx.n * G;
        const half*    row_scale  = ctx.group_scale  + (size_t)ctx.n * G;
        const half*    row_mid    = ctx.group_mid    + (size_t)ctx.n * G;
        for (int g = lane; g < G; g += XFP_WARP_SIZE) {
            my_warp_meta[g * META_PER_GROUP + 0] = (float) row_lib_id[g];
            my_warp_meta[g * META_PER_GROUP + 1] = __half2float(row_scale[g]);
            my_warp_meta[g * META_PER_GROUP + 2] = __half2float(row_mid[g]);
        }
    }
    __syncwarp();  // warp-local: all lanes see this warp's meta

    int K_packed_safe = K / VALS_PER_WORD;
    int n_full_groups = K_packed_safe / XFP_WARP_SIZE;

    // Storage for the lane's own codebook entry (refreshed every outer iter)
    float my_cb_val = 0.0f;

    // Fast path: no bounds checks
    for (int gi = 0; gi < n_full_groups; gi++) {
        // ── V2 PATCH: per-lane-group codebook reload ──
        // Each outer iter spans CB_PER_ITER groups; this lane belongs to
        // group (gi * CB_PER_ITER + lane_group). Read from per-warp SMEM
        // (was: 3× global loads per iter — that was the +37% bottleneck).
        int my_group_idx = gi * CB_PER_ITER + lane_group;
        const float* meta = my_warp_meta + my_group_idx * META_PER_GROUP;
        int lib_id   = (int) meta[0];
        float scale_f = meta[1];
        float mid_f   = meta[2];
        my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx])
                    * scale_f + mid_f;

        // ── Existing hot loop (unchanged except SHFL src lane) ──
        int kw = lane + gi * XFP_WARP_SIZE;
        uint32_t packed = ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset];
        int k_base = kw * VALS_PER_WORD;

        #pragma unroll
        for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
            __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
                A_src + k_base + slot);
            float a0 = __bfloat162float(__low2bfloat16(a2));
            float a1 = __bfloat162float(__high2bfloat16(a2));
            int idx0 = (int)((packed >> (slot * BITS)) & MASK);
            int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
            // V2: SHFL src lane = cb_lane_offset + idx
            float w0 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx0);
            float w1 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx1);
            acc = fmaf(w0, a0, acc);
            acc = fmaf(w1, a1, acc);
        }
    }

    // Tail: remaining packed words past K_packed_safe (rare; K_packed
    // typically equals K_packed_safe for shapes where K is a multiple
    // of VALS_PER_WORD which is asserted host-side for v17_lib).
    // For now just bail if not aligned; future revision can mirror v12's
    // tail handling.
    int n_groups_total = (K_packed + XFP_WARP_SIZE - 1) / XFP_WARP_SIZE;
    // (no-op if n_full_groups == n_groups_total)
    (void)n_groups_total;

    // Warp reduction (same as v12)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        PolicyV2::epilogue(C, N, ctx, acc);
    }
}


// ─── Linear V2 policy ────────────────────────────────────────────────

struct LinearPolicyV2 {
    static constexpr int K_SMEM_MAX = 8192;

    struct Params {
        int M;
    };

    struct Ctx {
        bool active;
        int n;
        int m;
        const __nv_bfloat16* A_row;
        const uint32_t*      B_packed;
        const half*          library;
        const int32_t*       group_lib_id;
        const half*          group_scale;
        const half*          group_mid;
    };

    __device__ static const __nv_bfloat16* block_A_row(
        const __nv_bfloat16* A, int K, const Params& p)
    {
        int m = blockIdx.y;
        return (m < p.M) ? (A + (size_t)m * K) : nullptr;
    }

    template <int BITS, int LUT>
    __device__ static Ctx prologue(
        const __nv_bfloat16* A,
        const uint32_t*      B,
        const half*          library_,
        const int32_t*       group_lib_id_,
        const half*          group_scale_,
        const half*          group_mid_,
        int N, int K,
        int warp_id, int /*lane*/,
        Params p)
    {
        int n = blockIdx.x * XFP_WARPS_PER_BLOCK + warp_id;
        int m = blockIdx.y;
        Ctx c{};
        c.active = (m < p.M) && (n < N);
        if (!c.active) return c;
        c.n = n;
        c.m = m;
        c.A_row        = A + m * K;
        c.B_packed     = B;
        c.library      = library_;
        c.group_lib_id = group_lib_id_;
        c.group_scale  = group_scale_;
        c.group_mid    = group_mid_;
        return c;
    }

    __device__ static void epilogue(
        __nv_bfloat16* C, int N, const Ctx& ctx, float acc)
    {
        C[ctx.m * N + ctx.n] = __float2bfloat16(acc);
    }
};


// ─── V2 kernel entry point ──────────────────────────────────────────

template <int BITS, class PolicyV2>
__global__ void xfp_gemm_v2_templated_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint32_t*      __restrict__ B_packed,
    const half*          __restrict__ library,
    const int32_t*       __restrict__ group_lib_id,
    const half*          __restrict__ group_scale,
    const half*          __restrict__ group_mid,
    __nv_bfloat16*       __restrict__ C,
    int N, int K, int K_packed,
    int G, int group_size, int library_size,
    typename PolicyV2::Params params)
{
    xfp_gemm_core_v2<BITS, PolicyV2>(
        A, B_packed, library, group_lib_id, group_scale, group_mid,
        C, N, K, K_packed, G, group_size, library_size, params);
}

}  // namespace multiquant
