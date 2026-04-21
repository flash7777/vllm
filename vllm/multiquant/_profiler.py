"""NVTX range helper gated on VLLM_NVTX_PROFILE=1.

vLLM's model forward is decorated with `@support_torch_compile`, and
`torch.cuda.nvtx.range_push/pop` is not traceable by dynamo (it returns an
int via a C function, which triggers graph break gb0208). We decorate the
push/pop helpers with `@torch._dynamo.disable` so dynamo treats them as
opaque function calls — the NVTX events still fire at runtime, but tracing
steps around them.

No-op when VLLM_NVTX_PROFILE is not set, so production paths stay untouched.

Usage:
    from vllm.multiquant._profiler import nvtx

    with nvtx(f"layer_{idx}/attn"):
        self.self_attn(...)
"""

from __future__ import annotations

import os

_ENABLED = os.environ.get("VLLM_NVTX_PROFILE") == "1"


if _ENABLED:
    import torch

    @torch._dynamo.disable(recursive=False)
    def _push(name: str) -> None:
        torch.cuda.nvtx.range_push(name)

    @torch._dynamo.disable(recursive=False)
    def _pop() -> None:
        torch.cuda.nvtx.range_pop()

    class _NvtxRange:
        __slots__ = ("name",)

        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            _push(self.name)
            return self

        def __exit__(self, exc_type, exc, tb):
            _pop()
            return False

    def nvtx(name: str):
        return _NvtxRange(name)

else:

    class _NoOp:
        __slots__ = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    _NOOP_INSTANCE = _NoOp()

    def nvtx(name: str):
        return _NOOP_INSTANCE


def is_enabled() -> bool:
    return _ENABLED
