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
    static_assert(BITS == 2 || BITS == 3 || BITS == 4,
                  "xfp_gemm_core_v2: BITS must be 2, 3, or 4");
    // BITS=2 → 16 vals/word (LUT=4), BITS=3 → 10 vals/word (LUT=8),
    // BITS=4 → 8 vals/word (LUT=16). BITS=3 wastes 2 bits/word.
    constexpr int VALS_PER_WORD = (BITS == 2) ? 16 : 8;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr int LUT_SIZE = (1 << BITS);

    int warp_id = threadIdx.x / XFP_WARP_SIZE;
    int lane    = threadIdx.x % XFP_WARP_SIZE;

    // Lane-subgroup partitioning. For BITS=4, LUT_SIZE=16, WARP_SIZE=32:
    // 2 lane-subgroups (codebooks) per warp outer iter → CB_PER_ITER = 2.
    // GROUP_SIZE hardcoded to 128 (host TORCH_CHECK enforces this).
    // For BITS=4: VALS_PER_WORD=8, LANES_PER_GROUP=16 (=LUT_SIZE, coincidence).
    // For BITS=2: VALS_PER_WORD=16, LANES_PER_GROUP=8 (<LUT_SIZE — lane
    // duplicates within each codebook subgroup; shuffle uses
    // cb_lane_offset+idx, only first LUT_SIZE lanes hold distinct values).
    constexpr int GROUP_SIZE      = 128;
    constexpr int LANES_PER_GROUP = GROUP_SIZE / VALS_PER_WORD;
    constexpr int CB_PER_ITER     = XFP_WARP_SIZE / LANES_PER_GROUP;
    const int my_cb_idx           = lane % LUT_SIZE;            // 0..15
    const int lane_group          = lane / LANES_PER_GROUP;            // 0 or 1
    const int cb_lane_offset      = lane_group * LANES_PER_GROUP;      // 0 or 16

    // ── Dynamic SMEM layout ──
    // [0]            s_A        : K * sizeof(bf16)
    // [s_A end]      s_library  : library_size * LUT_SIZE * sizeof(half)
    //
    // Codebook prebuild was tested (storing G fully-rebuilt codebooks per
    // warp in SMEM) including K=8192 explicitly: fp32 +44.5% median
    // (vs per-iter +35.4%) — at K=8192 the s_codebooks alone is 32 KB,
    // dropping occupancy 6→2 blocks/SM. fp16 storage hits 2-way SMEM bank
    // conflicts (16-bit reads on 32-bit banks). Per-iter rebuild wins for
    // every K tested. The 3-LDG metadata fetch is L1-cached (only 2
    // distinct addresses per warp per outer iter); the cb_val FMA is
    // structural and unavoidable in this design.
    extern __shared__ char xfp_v2_smem[];
    size_t off_A   = 0;
    size_t off_lib = off_A + (size_t)K * sizeof(__nv_bfloat16);
    __nv_bfloat16* s_A       = reinterpret_cast<__nv_bfloat16*>(
        xfp_v2_smem + off_A);
    half*          s_library = reinterpret_cast<half*>(
        xfp_v2_smem + off_lib);

    // ── Cooperative library load ──
    {
        int total = library_size * LUT_SIZE;
        for (int i = threadIdx.x; i < total; i += XFP_BLOCK_SIZE) {
            s_library[i] = library[i];
        }
    }
    // ── Block-uniform A-row load (mirrors v12) ──
    const __nv_bfloat16* block_A = PolicyV2::block_A_row(A, K, params);
    if (block_A == nullptr) return;
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

    int K_packed_safe = K / VALS_PER_WORD;
    int n_full_groups = K_packed_safe / XFP_WARP_SIZE;

    // Per-row metadata pointers — read directly per outer iter, L1-cached.
    const int32_t* row_lib_id = ctx.group_lib_id + (size_t)ctx.n * G;
    const half*    row_scale  = ctx.group_scale  + (size_t)ctx.n * G;
    const half*    row_mid    = ctx.group_mid    + (size_t)ctx.n * G;

    float my_cb_val = 0.0f;

    if constexpr (BITS == 3) {
        // ── V3 BITS=3 hot loop: per-group + lane-padding + SMEM-direct ──
        // 13 lanes/group, 2 groups/warp = 26 active lanes; lanes 26..31 idle.
        // Codebook lookup via s_library (SMEM-direct, no shuffle subgroup).
        // 2 padding slots/group end skipped via abs_slot < GROUP_SIZE check.
        // Memory: K_PACKED_PER_GROUP = 13 (vs 16 for BITS=4) → 17% saving.
        constexpr int LANES_PER_GROUP_V3 = 13;
        constexpr int CB_PER_ITER_V3     = 2;
        constexpr int ACTIVE_LANES_V3    = CB_PER_ITER_V3 * LANES_PER_GROUP_V3;  // 26

        const bool lane_active_v3 = (lane < ACTIVE_LANES_V3);
        const int  lane_grp_v3    = lane / LANES_PER_GROUP_V3;   // 0 or 1
        const int  lane_in_grp_v3 = lane % LANES_PER_GROUP_V3;   // 0..12

        const int n_warp_iters_v3 = G / CB_PER_ITER_V3;  // G must be even

        for (int gi = 0; gi < n_warp_iters_v3; gi++) {
            if (!lane_active_v3) continue;

            int   my_group_idx = gi * CB_PER_ITER_V3 + lane_grp_v3;
            int   lib_id       = (int) row_lib_id[my_group_idx];
            float scale_f      = __half2float(row_scale[my_group_idx]);
            float mid_f        = __half2float(row_mid  [my_group_idx]);

            const half* my_lib = s_library + (size_t)lib_id * LUT_SIZE;

            // Stride per warp-iter = ACTIVE_LANES_V3 (26), N-stride likewise.
            uint32_t packed = ctx.B_packed[
                (size_t)gi      * (size_t)N * ACTIVE_LANES_V3
              + (size_t)ctx.n   *               ACTIVE_LANES_V3
              + lane
            ];

            int k_base = my_group_idx * GROUP_SIZE
                       + lane_in_grp_v3 * VALS_PER_WORD;

            #pragma unroll
            for (int slot = 0; slot < VALS_PER_WORD; slot++) {
                int abs_slot = lane_in_grp_v3 * VALS_PER_WORD + slot;
                if (abs_slot >= GROUP_SIZE) break;  // 2 padding slots/group

                int idx = (int)((packed >> (slot * BITS)) & MASK);
                float w_norm = __half2float(my_lib[idx]);
                float w      = w_norm * scale_f + mid_f;

                __nv_bfloat16 a_bf = A_src[k_base + slot];
                float a            = __bfloat162float(a_bf);
                acc = fmaf(w, a, acc);
            }
        }
    } else {
        // ── BITS=2/4: original hot loop (unchanged) ──
        for (int gi = 0; gi < n_full_groups; gi++) {
            int my_group_idx = gi * CB_PER_ITER + lane_group;
            int   lib_id  = (int) row_lib_id[my_group_idx];
            float scale_f = __half2float(row_scale[my_group_idx]);
            float mid_f   = __half2float(row_mid  [my_group_idx]);
            my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx])
                        * scale_f + mid_f;

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
                float w0 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx0);
                float w1 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx1);
                acc = fmaf(w0, a0, acc);
                acc = fmaf(w1, a1, acc);
            }
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
    // Lifted from 8192 → 32768 to support dense FFN down_proj of
    // Qwen3.6-27B (K=17408) and GLM-4.7-Flash (K=10240). Inner loop
    // (`n_full_groups = K_packed / WARP_SIZE`) is K-agnostic; SMEM
    // scales as 2K bytes (K=17408 → 35 KB, K=32768 → 66 KB) — fits in
    // sm_120/121 96 KB carveout. Occupancy drops from 6 to 2-3 blocks
    // per SM at K=17408, accepted trade for unblocking dense models.
    static constexpr int K_SMEM_MAX = 32768;

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


// ─── MoE V2 policy ───────────────────────────────────────────────────
//
// Mirrors LinearPolicyV2 but indexes B_packed and per-group metadata by
// expert_id (from sorted_token_ids/expert_ids). Output is per-token via
// the same sorted scheme as v12 MoE.

struct MoEPolicyV2 {
    static constexpr int K_SMEM_MAX = 4096;

    struct Params {
        const int32_t* sorted_token_ids;   // [num_tokens_padded]
        const int32_t* expert_ids;         // [num_token_blocks]
        const float*   topk_weights;       // [M * top_k] or nullptr
        int M;
        int top_k;
        int flat_per_expert;               // int32 words per expert in B_packed
        int num_valid_tokens;
        int G;                             // K / group_size — needed for prologue offsets
    };

    struct Ctx {
        bool active;
        int n;
        int token_id;
        const uint32_t*      B_packed;
        const half*          library;
        const int32_t*       group_lib_id;  // pre-offset to expert
        const half*          group_scale;
        const half*          group_mid;
        const float*         topk_weights;
    };

    __device__ static const __nv_bfloat16* block_A_row(
        const __nv_bfloat16* A, int K, const Params& p)
    {
        int tb = blockIdx.y;
        int expert_id = p.expert_ids[tb];
        if (expert_id < 0) return nullptr;
        int token_id = p.sorted_token_ids[tb];
        if (token_id >= p.num_valid_tokens) return nullptr;
        int orig = token_id / p.top_k;
        if (orig >= p.M) return nullptr;
        return A + (size_t)orig * K;
    }

    template <int BITS, int LUT>
    __device__ static Ctx prologue(
        const __nv_bfloat16* /*A*/,
        const uint32_t*      B,
        const half*          library_,
        const int32_t*       group_lib_id_,
        const half*          group_scale_,
        const half*          group_mid_,
        int N, int /*K*/,
        int warp_id, int /*lane*/,
        Params p)
    {
        Ctx c{};
        int n  = blockIdx.x * XFP_WARPS_PER_BLOCK + warp_id;
        int tb = blockIdx.y;

        int expert_id = p.expert_ids[tb];
        if (expert_id < 0) return c;
        int token_id = p.sorted_token_ids[tb];
        if (token_id >= p.num_valid_tokens) return c;
        int orig = token_id / p.top_k;
        if (orig >= p.M || n >= N) return c;

        c.active = true;
        c.n = n;
        c.token_id = token_id;
        c.B_packed = B + (size_t)expert_id * p.flat_per_expert;
        c.library  = library_;
        // Per-expert metadata stride: e × N × G entries
        size_t e_off = (size_t)expert_id * (size_t)N * (size_t)p.G;
        c.group_lib_id = group_lib_id_ + e_off;
        c.group_scale  = group_scale_  + e_off;
        c.group_mid    = group_mid_    + e_off;
        c.topk_weights = p.topk_weights;
        return c;
    }

    __device__ static void epilogue(
        __nv_bfloat16* C, int N, const Ctx& ctx, float acc)
    {
        if (ctx.topk_weights != nullptr) {
            acc *= ctx.topk_weights[ctx.token_id];
        }
        C[(size_t)ctx.token_id * N + ctx.n] = __float2bfloat16(acc);
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
