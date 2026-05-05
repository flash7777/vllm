# SPDX-License-Identifier: Apache-2.0
"""GPUWorker shutdown hook for MultiQuant.

Extracted from `vllm/v1/worker/gpu_worker.py:Worker.shutdown` to keep
the core file's patch surface minimal across vLLM upgrades. The core
shutdown method now contains a single call into this module instead
of 159 lines of inline cleanup logic.

Why this is needed: vLLM's executor-shutdown window (between
SIGTERM and the executor join) does NOT proactively release the
loaded model. On podman + GB10 UMA, the SIGTERM grace period
expires before Python finalizers run; CUDA pages stay mapped and
the next container start fails with OOM.

What this does (4 phases):
  1. Drop model + runner buffers — shrink each Parameter/buffer
     storage to zero bytes so the caching allocator can reclaim
     the underlying CUDA memory immediately, regardless of how
     many wrapper objects (CUDAGraphWrapper, UBatchWrapper,
     LoRA, etc.) still hold refs to the .data tensor.
  2. Reset torch._dynamo — its AOT cache pins compiled closures
     that reference model Parameters.
  3. Cleanup MultiQuant module-level globals (JIT-compiled C++
     extension handles, kernel singletons).
  4. GC + empty_cache + cudaDeviceReset — last-resort tear-down
     of the primary CUDA context to unmap UVM pages.
"""

from __future__ import annotations

import ctypes
import gc

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def release_model_and_caches(worker) -> None:
    """Release model + KV cache memory at GPUWorker shutdown.

    Called from `Worker.shutdown` after the upstream cleanup of
    kv_transfer / profiler / weight_transfer_engine has run.

    Mutates ``worker.model_runner`` to ``None``.
    """
    _alloc_before = 0
    if torch.cuda.is_available():
        try:
            _alloc_before = torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            pass

    logger.info(
        "[shutdown] GPUWorker releasing model + CUDA caches "
        "(CUDA alloc=%d MiB)", _alloc_before,
    )

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None:
        # Ref-breaking alone (setting .model = None) does not free CUDA
        # memory: compiled CUDA graphs, torch.ops closures, dynamo AOT
        # cache, and nn.Module._parameters/_modules dicts all keep
        # dangling refs to individual Parameters. Instead of fighting
        # the ref graph, shrink each Parameter's storage to zero bytes
        # in place — allocator frees the underlying blocks immediately
        # while the (now empty) Parameter wrappers can safely outlive
        # us until process exit. Same trick the cache-hit path in
        # multiquant/xfp/online_{linear,moe}.py uses.
        inner_model = getattr(model_runner, "model", None)
        if inner_model is not None:
            try:
                # Walk wrappers (CUDAGraphWrapper, UBatchWrapper, LoRA) to
                # reach the real nn.Module. Wrappers stash the wrapped
                # model under a variety of attribute names; grab any that
                # expose .parameters().
                candidates = [inner_model]
                for attr in ("model", "module", "wrapped", "_orig_mod"):
                    sub = getattr(inner_model, attr, None)
                    if sub is not None and hasattr(sub, "parameters"):
                        candidates.append(sub)
                seen = set()
                n_params = 0
                bytes_shrunk = 0
                for cand in candidates:
                    for p in cand.parameters():
                        if id(p) in seen:
                            continue
                        seen.add(id(p))
                        sz = p.data.element_size() * p.data.numel()
                        bytes_shrunk += sz
                        p.data = torch.empty(
                            0, device=p.data.device, dtype=p.data.dtype
                        )
                        n_params += 1
                    for name, buf in list(cand.named_buffers()):
                        if id(buf) in seen:
                            continue
                        seen.add(id(buf))
                        sz = buf.element_size() * buf.numel()
                        bytes_shrunk += sz
                        buf.data = torch.empty(
                            0, device=buf.device, dtype=buf.dtype
                        )
                        n_params += 1
                logger.info(
                    "[shutdown] shrunk %d model params/buffers, freed "
                    "%.1f GiB of nominal storage",
                    n_params, bytes_shrunk / (1024 ** 3),
                )
            except Exception as e:
                logger.warning("[shutdown] param shrink failed: %s", e)
            model_runner.model = None
        # Runner-side buffers that also pin storage.
        for attr in (
            "kv_caches", "attn_metadata_builder",
            "persistent_batch", "sampler", "draft_model_runner",
        ):
            if hasattr(model_runner, attr):
                setattr(model_runner, attr, None)
        worker.model_runner = None

    # torch.compile AOT cache + Dynamo cache hold compiled Python
    # closures that reference model Parameters — without reset they
    # pin the very tensors we just tried to drop.
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
    except Exception:
        pass

    # Reset MultiQuant module-level kernel handles + singletons. The
    # JIT-compiled C++ extensions they cache hold CUDA-context-bound
    # workspaces; dropping the refs here lets empty_cache reclaim them.
    try:
        from vllm.multiquant import _cleanup_multiquant_globals
        _cleanup_multiquant_globals()
    except ImportError:
        pass

    # Two GC passes: first collects the wrapper chain, the second
    # picks up cycles the first pass exposed (compiled graphs often
    # have mutual refs to the model modules).
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            _alloc_after = torch.cuda.memory_allocated() // (1024 * 1024)
            logger.info(
                "[shutdown] CUDA alloc after cleanup: %d MiB (freed %d MiB)",
                _alloc_after, _alloc_before - _alloc_after,
            )
        except Exception:
            pass

        # Last-resort: force-destroy the primary CUDA context via
        # cudaDeviceReset. On GB10 UMA the driver does NOT return
        # UVM pages to the OS kernel on process exit alone —
        # empty_cache releases the PyTorch caching-allocator pool
        # but the driver still tracks the pages as live. Explicit
        # cudaDeviceReset() should (a) tear down the primary
        # context, (b) unmap all UVM pages, (c) let the OS reclaim
        # the physical memory. Safe here because we've already
        # dropped the model and the worker is about to exit.
        _cuda_device_reset()


def _cuda_device_reset() -> None:
    """Best-effort cudaDeviceReset() via libcudart."""
    try:
        cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        cudart.cudaDeviceSynchronize.restype = ctypes.c_int
        cudart.cudaDeviceReset.restype = ctypes.c_int
        cudart.cudaDeviceSynchronize()
        rc = cudart.cudaDeviceReset()
        logger.info("[shutdown] cudaDeviceReset -> rc=%d", int(rc))
        return
    except OSError:
        pass
    except Exception as e:
        logger.warning("[shutdown] cudaDeviceReset failed: %s", e)
        return

    # Fallback: try fully-qualified library names.
    try:
        for cand in (
            "libcudart.so.13", "libcudart.so.12",
            "libcudart.so.11", "libcudart.so.13.0",
        ):
            try:
                cudart = ctypes.CDLL(cand)
                cudart.cudaDeviceReset.restype = ctypes.c_int
                rc = cudart.cudaDeviceReset()
                logger.info(
                    "[shutdown] cudaDeviceReset via %s -> rc=%d",
                    cand, int(rc),
                )
                return
            except OSError:
                continue
        logger.warning(
            "[shutdown] libcudart not found; cudaDeviceReset skipped")
    except Exception as e:
        logger.warning("[shutdown] cudaDeviceReset fallback failed: %s", e)
