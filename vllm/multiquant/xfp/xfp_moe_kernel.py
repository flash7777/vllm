# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the XFP fused MoE GEMM CUDA kernel."""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

_xfp_moe_kernel = None
_load_attempted = False


def _find_kernel_cu() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(
        os.path.join(here, "..", "..", "..", "kernels", "multiquant")
    )
    for d in [src_dir, "/opt/mq_kernels"]:
        p = os.path.join(d, "xfp_moe_gemm.cu")
        if os.path.exists(p):
            return p
    return None


_KERNEL_CU_PATH: Optional[str] = _find_kernel_cu()


def _load_xfp_moe_gemm():
    global _xfp_moe_kernel, _load_attempted

    if _xfp_moe_kernel is not None:
        return _xfp_moe_kernel

    if _load_attempted:
        return None
    _load_attempted = True

    if _KERNEL_CU_PATH is None:
        logger.warning("XFP MoE kernel: xfp_moe_gemm.cu not found")
        return None

    try:
        from torch.utils.cpp_extension import load
        _xfp_moe_kernel = load(
            name="xfp_moe_gemm",
            sources=[_KERNEL_CU_PATH],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("XFP fused MoE GEMM kernel compiled from %s",
                     os.path.dirname(_KERNEL_CU_PATH))
        return _xfp_moe_kernel
    except Exception as e:
        logger.warning("XFP MoE kernel JIT compile FAILED: %s", e)
        return None
