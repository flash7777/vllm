// SPDX-License-Identifier: Apache-2.0
// TurboQuant fused round-trip CUDA kernel.
//
// Replaces the PyTorch tq_round_trip_keys() with a single fused kernel.
// Operations: normalize → rotate(Pi) → quantize → reconstruct → residual →
//             QJL(S) → combine. All via tiled GEMV in shared memory.
//
// Grid: (num_tokens, num_kv_heads)
// Block: (BLOCK_SIZE threads)
// Shared memory: ~36 KB for head_size=256

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>

#include "tq_round_trip.cuh"

namespace turboquant {

// Configuration
constexpr int BLOCK_SIZE = 256;  // threads per block
constexpr int WARP_SIZE = 32;
constexpr int MAX_CENTROIDS = 16;  // max 4-bit = 16 centroids

// Convert half/bf16 to float
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<c10::Half>(c10::Half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float to_float<float>(float val) {
  return val;
}

// Convert float to half/bf16
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ c10::Half from_float<c10::Half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float val) {
  return static_cast<at::BFloat16>(val);
}

template <>
__device__ __forceinline__ float from_float<float>(float val) {
  return val;
}


// ==========================================================================
// Main fused kernel
// ==========================================================================
template <typename scalar_t, int HEAD_SIZE, int N_CENTROIDS>
__global__ void turboquant_round_trip_kernel(
    const scalar_t* __restrict__ key_in,   // [N, H, D]
    scalar_t* __restrict__ key_out,         // [N, H, D]
    const float* __restrict__ Pi,           // [D, D]
    const float* __restrict__ S,            // [D, D]
    const float* __restrict__ centroids,    // [N_CENTROIDS]
    int num_heads
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int D = HEAD_SIZE;
    constexpr int TILE_ROWS = BLOCK_SIZE;  // 256 threads = 256 rows per tile
    constexpr int NUM_TILES = (D + TILE_ROWS - 1) / TILE_ROWS;

    // Shared memory layout:
    //   [0..D-1]:          x_hat (normalized key)
    //   [D..2D-1]:         work buffer (rotated / x_mse / residual)
    //   [2D..2D+7]:        warp reduction scratch
    //   [2D+8..2D+8+NC-1]: centroids (cached)
    extern __shared__ float smem[];
    float* s_x_hat = smem;                    // D floats
    float* s_work = smem + D;                 // D floats
    float* s_reduce = smem + 2 * D;           // 8 floats (warp scratch)
    float* s_centroids = smem + 2 * D + 8;    // N_CENTROIDS floats

    // Base addresses
    const int key_offset = (token_idx * num_heads + head_idx) * D;

    // Initialize shared memory
    if (tid < 8) s_reduce[tid] = 0.0f;
    if (tid < N_CENTROIDS) s_centroids[tid] = centroids[tid];

    // ====== Stage 1: Load key + compute L2 norm ======
    float local_val = 0.0f;
    float local_sq = 0.0f;
    if (tid < D) {
        local_val = to_float(key_in[key_offset + tid]);
        local_sq = local_val * local_val;
    }

    // Block-level reduce for L2 norm
    if (tid < 8) s_reduce[tid] = 0.0f;
    __syncthreads();
    float norm_sq = block_reduce_sum(local_sq, s_reduce, tid, BLOCK_SIZE);
    __shared__ float s_vec_norm;
    if (tid == 0) {
        s_vec_norm = sqrtf(norm_sq + 1e-8f);
    }
    __syncthreads();

    // Normalize and store in shared memory
    if (tid < D) {
        s_x_hat[tid] = local_val / s_vec_norm;
    }
    __syncthreads();

    // ====== Stage 2: Rotate y = x_hat @ Pi^T ======
    // y[i] = sum_j x_hat[j] * Pi^T[j, i] = sum_j x_hat[j] * Pi[i, j]
    // Pi is row-major: Pi[i, j] = Pi[i * D + j]
    // So y[tid] = sum_j x_hat[j] * Pi[tid, j] = sum_j Pi[tid * D + j] * x_hat[j]
    float y_val = 0.0f;
    if (tid < D) {
        for (int j = 0; j < D; j++) {
            y_val += Pi[tid * D + j] * s_x_hat[j];
        }
    }
    // y_val now holds the rotated coordinate for this thread's dimension
    __syncthreads();

    // ====== Stage 3: Scalar quantize to nearest centroid ======
    int best_idx = 0;
    float best_dist = 1e30f;
    float best_centroid = 0.0f;
    if (tid < D) {
        for (int c = 0; c < N_CENTROIDS; c++) {
            float dist = fabsf(y_val - s_centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
                best_centroid = s_centroids[c];
            }
        }
    }
    // best_centroid = y_hat[tid] = centroids[best_idx]

    // Store y_hat in shared memory for unrotation
    if (tid < D) {
        s_work[tid] = best_centroid;
    }
    __syncthreads();

    // ====== Stage 4: Unrotate x_mse = y_hat @ Pi ======
    // x_mse[i] = sum_j y_hat[j] * Pi[j, i] = sum_j Pi[j * D + i] * y_hat[j]
    float x_mse_val = 0.0f;
    if (tid < D) {
        for (int j = 0; j < D; j++) {
            x_mse_val += Pi[j * D + tid] * s_work[j];
        }
    }
    __syncthreads();

    // ====== Stage 5: Residual + norm ======
    float r_val = 0.0f;
    if (tid < D) {
        r_val = s_x_hat[tid] - x_mse_val;
    }

    // Compute residual L2 norm
    float r_sq = (tid < D) ? r_val * r_val : 0.0f;
    if (tid < 8) s_reduce[tid] = 0.0f;
    __syncthreads();
    float gamma_sq = block_reduce_sum(r_sq, s_reduce, tid, BLOCK_SIZE);
    __shared__ float s_gamma;
    if (tid == 0) {
        s_gamma = sqrtf(gamma_sq + 1e-8f);
    }
    __syncthreads();

    // Store residual in shared memory for QJL projection
    if (tid < D) {
        s_work[tid] = r_val;
    }
    __syncthreads();

    // ====== Stage 6: QJL projection: proj = r @ S^T, signs = sign(proj) ======
    // proj[i] = sum_j r[j] * S^T[j, i] = sum_j r[j] * S[i, j] = sum_j S[i*D+j] * r[j]
    float proj_val = 0.0f;
    if (tid < D) {
        for (int j = 0; j < D; j++) {
            proj_val += S[tid * D + j] * s_work[j];
        }
    }
    float sign_val = (proj_val >= 0.0f) ? 1.0f : -1.0f;

    // Store signs in shared memory for QJL correction GEMV
    if (tid < D) {
        s_work[tid] = sign_val;
    }
    __syncthreads();

    // ====== Stage 7: QJL correction: x_qjl = signs @ S ======
    // x_qjl[i] = sum_j signs[j] * S[j, i] = sum_j S[j * D + i] * signs[j]
    float x_qjl_val = 0.0f;
    if (tid < D) {
        for (int j = 0; j < D; j++) {
            x_qjl_val += S[j * D + tid] * s_work[j];
        }
    }

    // Scale by correction factor
    const float correction_scale = sqrtf(M_PI_2) / static_cast<float>(D);
    x_qjl_val *= correction_scale * s_gamma;

    // ====== Stage 8: Combine and write output ======
    if (tid < D) {
        float result = s_vec_norm * (x_mse_val + x_qjl_val);
        key_out[key_offset + tid] = from_float<scalar_t>(result);
    }
}


// ==========================================================================
// C++ wrapper with dispatch
// ==========================================================================
void turboquant_round_trip(
    torch::Tensor& key,
    torch::Tensor& Pi,
    torch::Tensor& S,
    torch::Tensor& centroids,
    torch::Tensor& output,
    int head_size,
    int n_centroids
) {
    const int num_tokens = key.size(0);
    const int num_heads = key.size(1);

    // Shared memory: 2*D + 8 + N_CENTROIDS floats
    const int smem_bytes = (2 * head_size + 8 + n_centroids) * sizeof(float);

    dim3 grid(num_tokens, num_heads);
    dim3 block(BLOCK_SIZE);

    // Helper lambda to launch kernel for a given scalar_t
    auto launch = [&]<typename scalar_t>() {
        if (head_size == 128 && n_centroids == 4) {
            turboquant_round_trip_kernel<scalar_t, 128, 4>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 128 && n_centroids == 8) {
            turboquant_round_trip_kernel<scalar_t, 128, 8>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 256 && n_centroids == 4) {
            turboquant_round_trip_kernel<scalar_t, 256, 4>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else if (head_size == 256 && n_centroids == 8) {
            turboquant_round_trip_kernel<scalar_t, 256, 8>
                <<<grid, block, smem_bytes>>>(
                    key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    Pi.data_ptr<float>(), S.data_ptr<float>(),
                    centroids.data_ptr<float>(), num_heads);
        } else {
            TORCH_CHECK(false,
                "TurboQuant: unsupported head_size=", head_size,
                " n_centroids=", n_centroids);
        }
    };

    auto dtype = key.scalar_type();
    if (dtype == at::ScalarType::Half) {
        launch.template operator()<c10::Half>();
    } else if (dtype == at::ScalarType::BFloat16) {
        launch.template operator()<at::BFloat16>();
    } else {
        TORCH_CHECK(false, "TurboQuant: unsupported dtype. Expected Half or BFloat16.");
    }
}

}  // namespace turboquant
