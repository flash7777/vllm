# SPDX-License-Identifier: Apache-2.0
"""Archer CUDA kernel bindings — JIT compiled on first use."""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_kernel = None


@lru_cache(maxsize=1)
def _load_kernel():
    """JIT compile archer_decompress kernel."""
    global _kernel
    if _kernel is not None:
        return _kernel

    src_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "csrc", "quantization", "archer"
    )
    # Also check container path
    if not os.path.exists(src_dir):
        src_dir = "/opt/archer_build"

    src_file = os.path.join(src_dir, "archer_decompress.cu")
    if not os.path.exists(src_file):
        logger.warning("Archer CUDA source not found at %s", src_file)
        return None

    try:
        from torch.utils.cpp_extension import load
        _kernel = load(
            name="archer_decompress",
            sources=[src_file],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("Archer CUDA kernel compiled successfully")
        return _kernel
    except Exception as e:
        logger.warning("Archer CUDA kernel compilation failed: %s", e)
        return None


def cuda_decompress(
    packed_W: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    head_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """Decompress packed uint8 weights using CUDA kernel.

    Returns decompressed tensor, or None if kernel unavailable.
    """
    kernel = _load_kernel()
    if kernel is None:
        return None

    num_rows = packed_W.shape[0]
    n_centroids = centroids.shape[0]
    W_out = torch.empty(num_rows, head_size, dtype=out_dtype,
                        device=packed_W.device)

    try:
        kernel.archer_decompress(
            packed_W.contiguous(),
            W_out,
            Pi.float().contiguous(),
            S.float().contiguous(),
            centroids.float().contiguous(),
            head_size,
            n_centroids,
        )
        return W_out
    except Exception as e:
        logger.debug("Archer CUDA decompress failed: %s", e)
        return None
