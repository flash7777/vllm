# SPDX-License-Identifier: Apache-2.0
"""MultiQuant: Generic KV-cache quantization framework for vLLM.

Supports multiple quantization algorithms (TurboQuant, RotorQuant, etc.)
via a unified interface and registry.

Usage:
    from vllm.multiquant import get_kv_quantizer_config, is_multiquant_dtype

    if is_multiquant_dtype("tq3"):
        config = get_kv_quantizer_config("tq3", head_dim=128)
"""

from vllm.multiquant.registry import (
    get_kv_quantizer,
    get_kv_quantizer_config,
    get_registered_dtypes,
    is_multiquant_dtype,
    register_kv_quantizer,
)


def _cleanup_multiquant_globals() -> None:
    """Reset module-level kernel handles and singletons at shutdown.

    JIT-compiled CUDA extensions (xfp_gemm, xfp_moe_gemm, mq_gemm_int{2,3})
    are cached as module-level globals once loaded. On GB10 UMA the CUDA
    context they bind keeps driver pages pinned well past ``podman stop``
    unless we drop those refs before ``torch.cuda.empty_cache()``. Same
    reasoning for the policy/cache singletons, which can otherwise pin
    stats tensors.

    Callers (e.g. GPUWorker.shutdown) invoke this right before
    ``gc.collect()`` + ``torch.cuda.empty_cache()``. Safe to call
    repeatedly — all resets are idempotent. If the extension modules
    were never imported (e.g. non-MultiQuant model), the imports no-op
    silently through the try/except.
    """
    try:
        from vllm.multiquant.xfp import xfp_kernel, xfp_moe_kernel
        xfp_kernel._xfp_gemm_kernel = None
        xfp_kernel._xfp_gemm_v11_fallback = None
        xfp_kernel._load_attempted = False
        xfp_moe_kernel._xfp_moe_kernel = None
        xfp_moe_kernel._load_attempted = False
    except ImportError:
        pass
    try:
        from vllm.multiquant.weight_quant import mq_sub4_linear
        mq_sub4_linear._mq_gemm_int2 = None
        mq_sub4_linear._mq_gemm_int3 = None
    except ImportError:
        pass
    try:
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        MultiQuantWeightCache.set_active(None)
    except ImportError:
        pass
    try:
        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        MultiQuantPolicyRegistry._active = None
    except ImportError:
        pass


__all__ = [
    "_cleanup_multiquant_globals",
    "get_kv_quantizer",
    "get_kv_quantizer_config",
    "get_registered_dtypes",
    "is_multiquant_dtype",
    "register_kv_quantizer",
]
