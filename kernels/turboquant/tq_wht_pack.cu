// SPDX-License-Identifier: Apache-2.0
// TurboQuant v2: Fused WHT Pack Kernel
//
// Single kernel: raw float vector → WHT → amax → quantize → bitpack → uint8
// Replaces ~25 Python tensor-op launches with 1 CUDA launch.
//
// Grid: (N, D/32) — one warp per WHT block per vector
// Block: (32) — exactly one warp
//
// Same WHT transform as tq_wht_decode.cu (warp_wht_forward).
// Same packed format: [qs(8) | qr(4) | gamma(2)] per 32 values = 14 bytes.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace turboquant {

#define WARP_SIZE 32
#define WHT_STAGES 5

// Fixed sign-flip pattern (must match Python wht.py and tq_wht_decode.cu)
__constant__ float PACK_WHT_SIGNS[32] = {
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1,
};

// 3-bit thresholds for N(0,1) Lloyd-Max quantization
__constant__ float WHT_THRESHOLDS_3BIT[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f,
};

// WHT forward via warp shuffles (identical to decode kernel)
__device__ __forceinline__ float warp_wht_forward(float val, int lane) {
    val *= PACK_WHT_SIGNS[lane];
    for (int stage = 0; stage < WHT_STAGES; stage++) {
        int mask = 1 << stage;
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        if (lane & mask)
            val = other - val;
        else
            val = val + other;
    }
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    return val;
}

// Warp-reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

// 3-bit threshold quantization: returns index 0-7
__device__ __forceinline__ int threshold_quantize_3bit(float x) {
    int idx = 0;
    if (x > WHT_THRESHOLDS_3BIT[0]) idx = 1;
    if (x > WHT_THRESHOLDS_3BIT[1]) idx = 2;
    if (x > WHT_THRESHOLDS_3BIT[2]) idx = 3;
    if (x > WHT_THRESHOLDS_3BIT[3]) idx = 4;
    if (x > WHT_THRESHOLDS_3BIT[4]) idx = 5;
    if (x > WHT_THRESHOLDS_3BIT[5]) idx = 6;
    if (x > WHT_THRESHOLDS_3BIT[6]) idx = 7;
    return idx;
}

// Fused WHT Pack Kernel (3-bit)
// Input: [N, D] float32, Output: [N, packed_size] uint8
// packed_size = (D/32) * 14 bytes
template <int HEAD_DIM, int BLOCK_SIZE = 32>
__global__ void tq_wht_pack_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    int N,
    int packed_size
) {
    constexpr int BYTES_PER_BLOCK = 14;  // 8 qs + 4 qr + 2 gamma

    int vec_idx = blockIdx.x;
    int wht_block = blockIdx.y;
    int lane = threadIdx.x;

    if (vec_idx >= N) return;

    // 1. Load bfloat16 input → float
    float val = __bfloat162float(input[vec_idx * HEAD_DIM + wht_block * BLOCK_SIZE + lane]);

    // 2. WHT forward
    val = warp_wht_forward(val, lane);

    // 3. Per-block amax (warp reduce)
    float amax = warp_reduce_max(fabsf(val));
    amax = fmaxf(amax, 1e-10f);

    // 4. Gamma + normalize + quantize
    float gamma = amax / 2.1519f;
    float normalized = val / gamma;
    int idx = threshold_quantize_3bit(normalized);

    // 5. Cooperative bitpack
    // Output offset for this WHT block
    uint8_t* out = output + vec_idx * packed_size + wht_block * BYTES_PER_BLOCK;

    // 5a. Lower 2 bits → qs[8 bytes]: 4 indices per byte
    int low2 = idx & 0x3;
    // All lanes read all low2 values; only group leaders write
    {
        int base = (lane / 4) * 4;  // group start lane
        int b0 = __shfl_sync(0xffffffff, low2, base);
        int b1 = __shfl_sync(0xffffffff, low2, base + 1);
        int b2 = __shfl_sync(0xffffffff, low2, base + 2);
        int b3 = __shfl_sync(0xffffffff, low2, base + 3);
        if (lane % 4 == 0) {
            out[lane / 4] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
        }
    }

    // 5b. Upper 1 bit → qr[4 bytes]: 8 bits per byte
    int hi1 = (idx >> 2) & 1;
    {
        int base = (lane / 8) * 8;
        int byte_val = 0;
        for (int k = 0; k < 8; k++)
            byte_val |= (__shfl_sync(0xffffffff, hi1, base + k) << k);
        if (lane % 8 == 0) {
            out[8 + lane / 8] = (uint8_t)(byte_val & 0xFF);
        }
    }

    // 5c. Gamma as FP16 (2 bytes) — lane 0 writes
    if (lane == 0) {
        __half gamma_h = __float2half(gamma);
        uint16_t gamma_u16 = *reinterpret_cast<uint16_t*>(&gamma_h);
        out[12] = (uint8_t)(gamma_u16 & 0xFF);
        out[13] = (uint8_t)((gamma_u16 >> 8) & 0xFF);
    }
}


// C++ wrapper
void tq_wht_pack(
    torch::Tensor input,    // [N, D] bfloat16 — zero-copy from vLLM
    torch::Tensor output    // [N, packed_size] uint8 (pre-allocated)
) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    int N = input.size(0);
    int D = input.size(1);
    int packed_size = output.size(1);

    dim3 grid(N, D / 32);
    dim3 block(32);  // one warp
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define PACK_LAUNCH(HD) \
        tq_wht_pack_kernel<HD><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            output.data_ptr<uint8_t>(), N, packed_size)

    if (D == 256) { PACK_LAUNCH(256); }
    else if (D == 128) { PACK_LAUNCH(128); }
    else if (D == 64) { PACK_LAUNCH(64); }
    else if (D == 512) { PACK_LAUNCH(512); }
    else { TORCH_CHECK(false, "tq_wht_pack: unsupported D=", D); }
    #undef PACK_LAUNCH
}

}  // namespace turboquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq_wht_pack", &turboquant::tq_wht_pack,
          "Fused WHT pack: float → WHT → quantize → bitpack → uint8");
}
