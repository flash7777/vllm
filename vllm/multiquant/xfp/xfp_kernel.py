# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the XFP fused GEMM CUDA kernel.

Compiles kernels/multiquant/xfp_gemm*.cu once per process via
torch.utils.cpp_extension.load. One .so bedient alle Bitbreiten —
der Dispatch auf BITS passiert im C++ wrapper.

v12 (default) uses a static SMEM A-row cache with a compile-time
K_SMEM_MAX = 8192 (Linear). Shapes with K > 8192 (e.g. attn kv_b 17408)
fall back to v11 via ``dispatch_linear_gemm`` at call time.

Path resolution mirrors _load_mq_gemm in
vllm/multiquant/weight_quant/mq_sub4_linear.py: try the source tree
first (for development), fall back to /opt/mq_kernels (container).
"""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Keep in sync with LinearPolicy::K_SMEM_MAX in xfp_gemm_core.cuh.
K_SMEM_MAX_LINEAR = 8192


def _resolve_kernel_dir() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.normpath(
        os.path.join(here, "..", "..", "..", "kernels", "multiquant")
    )
    force = os.environ.get("XFP_KERNEL", "")
    if force == "v8":
        preferred = ("xfp_gemm_v8.cu", "xfp_gemm.cu")
    elif force == "v9":
        preferred = ("xfp_gemm_v9.cu",)
    elif force == "v10":
        preferred = ("xfp_gemm_v10.cu",)
    elif force == "v11":
        preferred = ("xfp_gemm_v11.cu",)
    else:
        # v12 default: static SMEM A-row cache. Needs v11 as fallback for
        # K > 8192 shapes, so we try v12 first and also keep v11 around.
        preferred = ("xfp_gemm_v12.cu", "xfp_gemm_v11.cu",
                     "xfp_gemm_v10.cu", "xfp_gemm_v8.cu", "xfp_gemm.cu")
    for name in preferred:
        if os.path.exists(os.path.join(src_dir, name)):
            return src_dir
    fallback = "/opt/mq_kernels"
    for name in preferred:
        if os.path.exists(os.path.join(fallback, name)):
            return fallback
    return None


_KERNEL_SRC_DIR: Optional[str] = _resolve_kernel_dir()

# Primary kernel (v12 by default). Used for all calls unless K exceeds
# K_SMEM_MAX_LINEAR, in which case dispatch_linear_gemm falls back to v11.
_xfp_gemm_kernel = None
# Optional v11 fallback module, populated only when the primary is v12.
_xfp_gemm_v11_fallback = None
_load_attempted = False


def _find_kernel_cu(primary_only: bool = False) -> Optional[str]:
    if _KERNEL_SRC_DIR is None:
        return None
    force = os.environ.get("XFP_KERNEL", "")
    if force == "v8":
        names = ("xfp_gemm_v8.cu", "xfp_gemm.cu")
    elif force == "v9":
        names = ("xfp_gemm_v9.cu",)
    elif force == "v10":
        names = ("xfp_gemm_v10.cu",)
    elif force == "v11":
        names = ("xfp_gemm_v11.cu",)
    else:
        names = ("xfp_gemm_v12.cu", "xfp_gemm_v11.cu",
                 "xfp_gemm_v10.cu", "xfp_gemm_v8.cu", "xfp_gemm.cu")
    for name in names:
        p = os.path.join(_KERNEL_SRC_DIR, name)
        if os.path.exists(p):
            return p
    return None


_KERNEL_CU_PATH: Optional[str] = _find_kernel_cu()


def _kname_from_path(path: str) -> str:
    for tag in ("v12", "v11", "v10", "v9", "v8"):
        if tag in path:
            return f"xfp_gemm_{tag}"
    return "xfp_gemm"


def _compile_one(cu_path: str):
    from torch.utils.cpp_extension import load
    return load(
        name=_kname_from_path(cu_path),
        sources=[cu_path],
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


def _load_xfp_gemm(bits: int):
    """JIT compile (first call) and return the xfp_gemm kernel module.

    When the primary kernel is v12, additionally JIT-compiles v11 so
    dispatch_linear_gemm can use it as a K>8192 fallback.

    IMPORTANT: this function must not use any os.path calls after the
    cached-kernel fast path. torch.compile/Dynamo graphs through this
    on the forward pass and cannot trace posix path normalization.
    """
    global _xfp_gemm_kernel, _xfp_gemm_v11_fallback, _load_attempted

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
            "XFP kernel: xfp_gemm*.cu not found in source tree "
            "or /opt/mq_kernels; kernel unavailable"
        )
        return None

    try:
        _xfp_gemm_kernel = _compile_one(_KERNEL_CU_PATH)
        primary_tag = _kname_from_path(_KERNEL_CU_PATH)
        logger.info("XFP GEMM kernel compiled (%s) from %s",
                    primary_tag, _KERNEL_SRC_DIR)

        # If primary is v12, compile v11 as K>8192 fallback.
        if "v12" in _KERNEL_CU_PATH and _KERNEL_SRC_DIR is not None:
            v11_path = os.path.join(_KERNEL_SRC_DIR, "xfp_gemm_v11.cu")
            if os.path.exists(v11_path):
                try:
                    _xfp_gemm_v11_fallback = _compile_one(v11_path)
                    logger.info(
                        "XFP Linear: v12 primary (static SMEM A-row cache, "
                        "K_SMEM_MAX=%d) + v11 fallback armed for K>%d",
                        K_SMEM_MAX_LINEAR, K_SMEM_MAX_LINEAR,
                    )
                except Exception as e:
                    logger.warning(
                        "XFP Linear v11 fallback compile FAILED: %s — "
                        "layers with K>%d will RAISE at runtime, not fall back",
                        e, K_SMEM_MAX_LINEAR,
                    )
            else:
                logger.warning(
                    "XFP Linear v11 fallback source missing (%s) — "
                    "layers with K>%d will RAISE at runtime, not fall back",
                    v11_path, K_SMEM_MAX_LINEAR,
                )
        else:
            # User forced a non-v12 primary (e.g., XFP_KERNEL=v11). No SMEM
            # A-row cache is active. Surface this loudly so a missing speedup
            # is traceable to the env override.
            force = os.environ.get("XFP_KERNEL", "")
            logger.warning(
                "XFP Linear: primary=%s (XFP_KERNEL=%r). v12 SMEM A-row "
                "cache is NOT active — per-warp global A-reads in use.",
                primary_tag, force or "<auto, v12 not found>",
            )

        return _xfp_gemm_kernel
    except Exception as e:
        logger.warning("XFP kernel JIT compile FAILED: %s", e)
        return None


# Track (K, N) pairs that already hit the v11 fallback path, so we log
# each distinct shape exactly once rather than every forward pass.
_FALLBACK_WARNED_SHAPES: set = set()


def dispatch_linear_gemm(x, packed, cb, C, bits: int, K: int) -> None:
    """Call the right Linear xfp_gemm for the given K.

    K <= K_SMEM_MAX_LINEAR  → primary kernel (v12 static SMEM cache)
    K  > K_SMEM_MAX_LINEAR  → v11 fallback (direct global A-row reads)

    Both kernels have identical numerical output; the only difference is
    the A-source path. Dispatch adds one Python-level branch per call
    (negligible vs the kernel time).

    Every distinct K that triggers the v11 fallback is logged once at
    WARN level so a disappointing tok/s can be traced back to a layer
    bypassing the SMEM A-row cache.
    """
    if _xfp_gemm_kernel is None:
        raise RuntimeError(
            "XFP kernel not loaded; call _load_xfp_gemm(bits) first."
        )

    use_fallback = (
        _xfp_gemm_v11_fallback is not None
        and K > K_SMEM_MAX_LINEAR
    )

    if use_fallback:
        shape_key = (int(K), int(C.shape[-1]), int(bits))
        if shape_key not in _FALLBACK_WARNED_SHAPES:
            _FALLBACK_WARNED_SHAPES.add(shape_key)
            logger.warning(
                "XFP Linear fallback → v11 for K=%d N=%d bits=%d "
                "(> K_SMEM_MAX=%d, no SMEM A-row cache for this layer)",
                K, C.shape[-1], bits, K_SMEM_MAX_LINEAR,
            )
        _xfp_gemm_v11_fallback.xfp_gemm(x, packed, cb, C, int(bits), int(K))
    elif K > K_SMEM_MAX_LINEAR and _xfp_gemm_v11_fallback is None:
        # Primary must be v12 (since K>8192 would've been fine for v11) but
        # fallback failed to load → runtime error, don't silently miscompute.
        raise RuntimeError(
            f"XFP Linear: K={K} > K_SMEM_MAX={K_SMEM_MAX_LINEAR} and no "
            f"v11 fallback is loaded. Set XFP_KERNEL=v11 to use v11 as "
            f"primary, or rebuild image with v11 source present."
        )
    else:
        _xfp_gemm_kernel.xfp_gemm(x, packed, cb, C, int(bits), int(K))


# ─── XFP-V2 kernel modules (v17_lib + v17_lib_splitm + v17_lib_splitk) ──

_xfp_v17_lib_module = None
_xfp_v17_lib_splitm_module = None
_xfp_v17_lib_splitk_module = None
_xfp_v2_load_attempted = False


def _load_xfp_v2_kernels():
    """JIT-load v17_lib (split-N), v17_lib_splitm and v17_lib_splitk on
    first call.

    Returns (v17_lib_module, v17_lib_splitm_module, v17_lib_splitk_module).
    Any of the three may be None if compile fails — caller must check.

    splitk (V3) is the K-axis split kernel for K > 8192 (Q3.6-27B
    down_proj K=17408, GLM-4.7-Flash down_proj K=10240). Replaces the
    V2a "split-N fallback" path which only worked at 1-2 blocks/SM.
    """
    global _xfp_v17_lib_module, _xfp_v17_lib_splitm_module
    global _xfp_v17_lib_splitk_module, _xfp_v2_load_attempted

    if _xfp_v2_load_attempted:
        return (_xfp_v17_lib_module, _xfp_v17_lib_splitm_module,
                _xfp_v17_lib_splitk_module)
    _xfp_v2_load_attempted = True

    if _KERNEL_SRC_DIR is None:
        logger.warning("XFP-V2: kernel source dir not resolved")
        return None, None, None

    from torch.utils.cpp_extension import load

    def _try(name: str, file: str):
        path = os.path.join(_KERNEL_SRC_DIR, file)
        if not os.path.exists(path):
            logger.warning("XFP-V2: %s missing at %s", file, path)
            return None
        try:
            return load(
                name=name,
                sources=[path],
                extra_cuda_cflags=[
                    "-O3", "-std=c++17", "--use_fast_math",
                    "-gencode=arch=compute_120,code=sm_120",
                    "-gencode=arch=compute_121,code=sm_121",
                    "-diag-suppress=177,3288",
                ],
                verbose=False,
            )
        except Exception as e:
            logger.warning("XFP-V2: %s compile FAILED: %s", file, e)
            return None

    _xfp_v17_lib_module = _try("xfp_gemm_v17_lib", "xfp_gemm_v17_lib.cu")
    _xfp_v17_lib_splitm_module = _try(
        "xfp_gemm_v17_lib_splitm", "xfp_gemm_v17_lib_splitm.cu")
    _xfp_v17_lib_splitk_module = _try(
        "xfp_gemm_v17_lib_splitk", "xfp_gemm_v17_lib_splitk.cu")

    if _xfp_v17_lib_module is not None:
        logger.info("XFP-V2: v17_lib (split-N) compiled")
    if _xfp_v17_lib_splitm_module is not None:
        logger.info("XFP-V2: v17_lib_splitm compiled")
    if _xfp_v17_lib_splitk_module is not None:
        logger.info("XFP-V2: v17_lib_splitk (V3, K>8192) compiled")
    return (_xfp_v17_lib_module, _xfp_v17_lib_splitm_module,
            _xfp_v17_lib_splitk_module)


# Threshold for switching from split-N to split-M-internal kernel.
# Below this M, split-N (v17_lib) is faster or equivalent. Above, split-M
# amortizes the per-Group codebook rebuild over M_CHUNK accumulators.
# Aligned with cudagraph_capture_sizes[0..n] — first non-trivial batch is 8;
# at M=8 split-M is mixed (some shapes regress slightly), at M=16+ it's a
# clear win for K ≤ 4096 and a strong improvement for K=8192.
XFP_V2_SPLITM_THRESHOLD_M = 16


def _select_v2_m_chunk(K: int) -> int:
    """Pick M_CHUNK based on K to keep s_A SMEM ≤ ~32 KB.

    s_A = M_CHUNK * K * 2 bytes. Target 32 KB for ~3 blocks/SM occupancy.
    Empirically validated by Phase B bench across realistic shapes.
    """
    if K <= 1024:
        return 8
    if K <= 4096:
        return 4
    return 2  # K up to 8192


# SMEM-Limit für split-M: M_CHUNK * K * bf16 + library + meta muss in
# ~96 KB carveout passen. Bei K=8192 mit M_CHUNK=2 → s_A = 32 KB OK.
# Bei K > 8192 nimmt der V3 split-K Kernel (v17_lib_splitk) über.
XFP_V2_SPLITM_K_MAX = 8192


def _select_v3_splitk_chunks(M: int) -> tuple:
    """Pick (M_CHUNK, K_CHUNK) for V3 split-K kernel.

    K_CHUNK=4096 default (best throughput per RTX bench). M_CHUNK
    matches effective M to avoid wasted accumulator slots.
    """
    if M == 1:
        m_chunk = 1
    elif M <= 4:
        m_chunk = 2
    else:
        m_chunk = 4
    return m_chunk, 4096


def _xfp_v3_enabled() -> bool:
    """V3 split-K path is opt-in via XFP_V3=1. Default off — fall back
    to V2a single-block-per-SM split-N for K > 8192.
    """
    return os.environ.get("XFP_V3", "0").lower() in ("1", "true", "yes", "on")


_xfp_v2_dispatch_seen: set = set()  # First-occurrence cache (M_bucket, K, path)


def _xfp_v2_log_dispatch(M: int, K: int, path: str, m_chunk: int = 0) -> None:
    """Log V2 dispatch decision once per (M-bucket, K, path) combination.

    Default: first-seen logging — emits one INFO per shape so we can
    confirm in benchmark logs which kernel served which layer-class.
    Set ``XFP_V2_LOG=verbose`` to log every call (very chatty).
    """
    if os.environ.get("XFP_V2_LOG", "").lower() in ("verbose", "v"):
        if path == "splitm":
            logger.info(
                "XFP-V2 dispatch: M=%d K=%d → splitm M_CHUNK=%d",
                M, K, m_chunk)
        else:
            logger.info("XFP-V2 dispatch: M=%d K=%d → split-N (v17_lib)", M, K)
        return
    # Bucketed first-occurrence
    if M == 1:
        m_b = "M=1"
    elif M <= 4:
        m_b = "M=2-4"
    elif M < XFP_V2_SPLITM_THRESHOLD_M:
        m_b = f"M=5-{XFP_V2_SPLITM_THRESHOLD_M-1}"
    elif M < 64:
        m_b = f"M={XFP_V2_SPLITM_THRESHOLD_M}-63"
    elif M < 256:
        m_b = "M=64-255"
    else:
        m_b = "M>=256"
    key = (m_b, K, path)
    if key in _xfp_v2_dispatch_seen:
        return
    _xfp_v2_dispatch_seen.add(key)
    if path == "splitm":
        logger.info(
            "XFP-V2 dispatch [first-seen]: %s K=%d -> splitm M_CHUNK=%d",
            m_b, K, m_chunk)
    elif path == "splitk":
        logger.info(
            "XFP-V2 dispatch [first-seen]: %s K=%d -> splitk V3 "
            "(M_CHUNK=%d, K_CHUNK=4096)",
            m_b, K, m_chunk)
    elif path == "split-N-fallback":
        logger.warning(
            "XFP-V2 dispatch [first-seen]: %s K=%d -> split-N FALLBACK "
            "(K > %d, splitk unavailable)",
            m_b, K, XFP_V2_SPLITM_K_MAX)
    else:
        logger.info(
            "XFP-V2 dispatch [first-seen]: %s K=%d -> split-N (v17_lib)",
            m_b, K)


def dispatch_v2_linear_gemm(
    x_bf16, packed_repacked, library, lib_id, scale, mid, C,
    bits: int, K: int, group_size: int,
) -> None:
    """Dispatch V2 Linear GEMM to split-N, split-M or split-K kernel.

    Caller is responsible for repacking ``packed`` (warp-interleaved
    layout via xfp_repack) and ensuring all tensors are contiguous + on
    the same device. C is pre-allocated [M, N] bf16.

    Decision matrix:
      - K  > 8192, XFP_V3=1:             splitk (V3)       -- opt-in,
                                                              chunked K
      - K  > 8192, default:              split-N (V2a)     -- single
                                                              block/SM,
                                                              tested path
      - M < THRESHOLD (16), K <= 8192:   split-N (v17_lib) -- top perf M=1
      - M >= THRESHOLD,    K <= 8192:    splitm (split-M)  -- batched
    Logging: first-seen (M-bucket, K, path) emits one INFO line. Set
    ``XFP_V2_LOG=verbose`` to log every call.
    """
    v17_lib, v17_splitm, v17_splitk = _load_xfp_v2_kernels()
    if v17_lib is None and v17_splitm is None and v17_splitk is None:
        raise RuntimeError("XFP-V2 kernels not loaded")

    M = int(x_bf16.shape[0])

    # K > 8192: V3 split-K opt-in via XFP_V3=1, otherwise V2a fallback
    # (single-block-per-SM split-N) — V2a is the tested path.
    if K > XFP_V2_SPLITM_K_MAX:
        if _xfp_v3_enabled() and v17_splitk is not None:
            m_chunk, k_chunk = _select_v3_splitk_chunks(M)
            _xfp_v2_log_dispatch(M, K, "splitk", m_chunk)
            v17_splitk.xfp_gemm_v17_lib_splitk(
                x_bf16, packed_repacked, library, lib_id, scale, mid, C,
                int(bits), int(K), int(group_size),
                int(m_chunk), int(k_chunk),
            )
            return
        # V2a tested fallback (Q3.6-27B K=17408, GLM K=10240).
        if v17_lib is None:
            raise RuntimeError(
                "XFP-V2: K=%d > 8192 but v17_lib not loaded" % K)
        _xfp_v2_log_dispatch(M, K, "split-N-fallback")
        v17_lib.xfp_gemm_v17_lib(
            x_bf16, packed_repacked, library, lib_id, scale, mid, C,
            int(bits), int(K), int(group_size),
        )
        return

    # K <= 8192: split-N or split-M based on M.
    use_splitm = (
        M >= XFP_V2_SPLITM_THRESHOLD_M
        and v17_splitm is not None
    )
    if not use_splitm:
        if v17_lib is None:
            raise RuntimeError(
                "XFP-V2: v17_lib (split-N) not loaded for M=%d" % M)
        _xfp_v2_log_dispatch(M, K, "split-N")
        v17_lib.xfp_gemm_v17_lib(
            x_bf16, packed_repacked, library, lib_id, scale, mid, C,
            int(bits), int(K), int(group_size),
        )
    else:
        m_chunk = _select_v2_m_chunk(K)
        _xfp_v2_log_dispatch(M, K, "splitm", m_chunk)
        v17_splitm.xfp_gemm_v17_lib_splitm(
            x_bf16, packed_repacked, library, lib_id, scale, mid, C,
            int(bits), int(K), int(group_size), int(m_chunk),
        )
