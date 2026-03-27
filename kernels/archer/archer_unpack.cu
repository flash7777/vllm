// SPDX-License-Identifier: Apache-2.0
// Archer bit-unpack kernel — extract MSE indices + QJL signs from packed uint8.
//
// Replaces the Python loops that are the decompress bottleneck.
// Does NOT do rotation or GEMM — those use cuBLAS (torch.mm).
//
// Grid:  (num_rows,)
// Block: (BLOCK_SIZE,)  — each thread unpacks multiple coordinates
//
// Outputs:
//   indices: [num_rows, D] int32 — MSE centroid indices
//   signs:   [num_rows, D] float32 — QJL signs (+1/-1)
//   row_norms: [num_rows] float32
//   res_norms: [num_rows] float32

#include <torch/extension.h>
#include <cuda_fp16.h>

namespace archer {

constexpr int BLOCK_SIZE = 256;

template <int D, int MSE_BITS>
__global__ void archer_unpack_kernel(
    const uint8_t* __restrict__ packed,   // [num_rows, packed_size]
    int32_t* __restrict__ indices,        // [num_rows, D]
    float* __restrict__ signs,            // [num_rows, D]
    float* __restrict__ row_norms,        // [num_rows]
    float* __restrict__ res_norms,        // [num_rows]
    int num_rows,
    int packed_size
) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;

    constexpr int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    constexpr int QJL_BYTES = (D + 7) / 8;
    constexpr int COORDS_PER_BYTE = 8 / MSE_BITS;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    const uint8_t* row_packed = packed + row * packed_size;
    int32_t* row_indices = indices + row * D;
    float* row_signs = signs + row * D;

    // Unpack MSE indices — each thread handles multiple coordinates
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int byte_idx = j / COORDS_PER_BYTE;
        int bit_pos = (j % COORDS_PER_BYTE) * MSE_BITS;
        row_indices[j] = (row_packed[byte_idx] >> bit_pos) & MASK;
    }

    // Unpack QJL signs
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int byte_idx = MSE_BYTES + j / 8;
        int bit_pos = j % 8;
        row_signs[j] = ((row_packed[byte_idx] >> bit_pos) & 1) ? 1.0f : -1.0f;
    }

    // Thread 0 unpacks norms
    if (tid == 0) {
        int no = MSE_BYTES + QJL_BYTES;
        uint16_t vn_bits = row_packed[no] | (row_packed[no + 1] << 8);
        uint16_t rn_bits = row_packed[no + 2] | (row_packed[no + 3] << 8);
        row_norms[row] = __half2float(*reinterpret_cast<const __half*>(&vn_bits));
        res_norms[row] = __half2float(*reinterpret_cast<const __half*>(&rn_bits));
    }
}

// Host launcher
void archer_unpack(
    torch::Tensor packed,       // [num_rows, packed_size] uint8
    torch::Tensor indices,      // [num_rows, D] int32
    torch::Tensor signs,        // [num_rows, D] float32
    torch::Tensor row_norms,    // [num_rows] float32
    torch::Tensor res_norms,    // [num_rows] float32
    int head_size,
    int mse_bits
) {
    int num_rows = packed.size(0);
    int packed_size = packed.size(1);

    #define LAUNCH(D, BITS) \
        archer_unpack_kernel<D, BITS><<<num_rows, BLOCK_SIZE>>>( \
            packed.data_ptr<uint8_t>(), \
            indices.data_ptr<int32_t>(), \
            signs.data_ptr<float>(), \
            row_norms.data_ptr<float>(), \
            res_norms.data_ptr<float>(), \
            num_rows, packed_size)

    if (mse_bits == 2) {
        if      (head_size == 128)   LAUNCH(128, 2);
        else if (head_size == 256)   LAUNCH(256, 2);
        else if (head_size == 512)   LAUNCH(512, 2);
        else if (head_size == 768)   LAUNCH(768, 2);
        else if (head_size == 1024)  LAUNCH(1024, 2);
        else if (head_size == 1536)  LAUNCH(1536, 2);
        else if (head_size == 2048)  LAUNCH(2048, 2);
        else if (head_size == 4096)  LAUNCH(4096, 2);
        else if (head_size == 5120)  LAUNCH(5120, 2);
        else if (head_size == 10240) LAUNCH(10240, 2);
        else TORCH_CHECK(false, "Unsupported head_size: ", head_size);
    } else if (mse_bits == 3) {
        if      (head_size == 128)   LAUNCH(128, 3);
        else if (head_size == 256)   LAUNCH(256, 3);
        else if (head_size == 512)   LAUNCH(512, 3);
        else if (head_size == 768)   LAUNCH(768, 3);
        else if (head_size == 1024)  LAUNCH(1024, 3);
        else if (head_size == 1536)  LAUNCH(1536, 3);
        else if (head_size == 2048)  LAUNCH(2048, 3);
        else if (head_size == 4096)  LAUNCH(4096, 3);
        else if (head_size == 5120)  LAUNCH(5120, 3);
        else if (head_size == 10240) LAUNCH(10240, 3);
        else TORCH_CHECK(false, "Unsupported head_size: ", head_size);
    } else {
        TORCH_CHECK(false, "Unsupported mse_bits: ", mse_bits);
    }

    #undef LAUNCH
}

}  // namespace archer

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("archer_unpack", &archer::archer_unpack,
          "Archer unpack: packed uint8 → indices + signs + norms");
}
