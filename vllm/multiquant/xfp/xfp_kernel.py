# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the XFP fused GEMM CUDA kernel.

Compiles kernels/multiquant/xfp_gemm.cu once per process via
torch.utils.cpp_extension.load. One .so bedient alle Bitbreiten —
der Dispatch auf BITS passiert im C++ wrapper.

Path resolution mirrors _load_mq_gemm in
vllm/multiquant/weight_quant/mq_sub4_linear.py: try the source tree
first (for development), fall back to /opt/mq_kernels (container).
"""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


# Resolved once at module import time. No os.path calls in _load_xfp_gemm
# or in the kernel apply path — torch.compile/Dynamo does not graph through
# posix.path functions, so any lazy path resolution inside the forward
# pass triggers a graph break (see gb0007 on pytorch docs).
def _resolve_kernel_dir() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(
        os.path.join(here, "..", "..", "..", "kernels", "multiquant")
    )
    # Prefer v4 kernel if available, fall back to v2/v1
    for name in ("xfp_gemm_v4.cu", "xfp_gemm.cu"):
        if os.path.exists(os.path.join(src_dir, name)):
            return src_dir
    fallback = "/opt/mq_kernels"
    for name in ("xfp_gemm_v4.cu", "xfp_gemm.cu"):
        if os.path.exists(os.path.join(fallback, name)):
            return fallback
    return None


_KERNEL_SRC_DIR: Optional[str] = _resolve_kernel_dir()
_xfp_gemm_kernel = None
_load_attempted = False


# Pre-compute the full source path so _load_xfp_gemm is trivially callable
# from anywhere without touching the filesystem API.
# Prefer v4 kernel (warp-per-element, register LUT) if available
def _find_kernel_cu() -> Optional[str]:
    if _KERNEL_SRC_DIR is None:
        return None
    for name in ("xfp_gemm_v4.cu", "xfp_gemm.cu"):
        p = os.path.join(_KERNEL_SRC_DIR, name)
        if os.path.exists(p):
            return p
    return None


_KERNEL_CU_PATH: Optional[str] = _find_kernel_cu()


def _load_xfp_gemm(bits: int):
    """JIT compile (first call) and return the xfp_gemm kernel module.

    `bits` is only validated; the compiled module dispatches 2/3/4 at
    runtime via the C++ wrapper in xfp_gemm.cu.

    IMPORTANT: this function must not use any os.path calls after the
    cached-kernel fast path. torch.compile/Dynamo graphs through this
    on the forward pass and cannot trace posix path normalization.
    """
    global _xfp_gemm_kernel, _load_attempted

    if _xfp_gemm_kernel is not None:
        return _xfp_gemm_kernel

    if bits not in (2, 3, 4):
        raise ValueError(
            f"XFP kernel: unsupported bits={bits}, must be in {{2,3,4}}"
        )

    if _load_attempted:
        return None
    _load_attempted = True

    if _KERNEL_CU_PATH is None:
        logger.warning(
            "XFP kernel: xfp_gemm.cu not found in source tree "
            "or /opt/mq_kernels; kernel unavailable"
        )
        return None

    try:
        from torch.utils.cpp_extension import load
        _xfp_gemm_kernel = load(
            name="xfp_gemm",
            sources=[_KERNEL_CU_PATH],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("XFP fused GEMM CUDA kernel compiled and loaded from %s",
                    _KERNEL_SRC_DIR)
        return _xfp_gemm_kernel
    except Exception as e:
        logger.warning("XFP kernel JIT compile FAILED: %s", e)
        return None
