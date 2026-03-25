# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend — compressed KV-cache.

Inherits from FlashInfer but overrides:
1. get_kv_cache_shape → uses tq_packed_size as head_size (5× smaller cache)
2. Store: compress K and V via TQ, pack into uint8 cache
3. Read: decompress from uint8 cache into temp BF16 buffer for FlashInfer

The actual attention computation is done by FlashInfer on the
decompressed BF16 buffer — no custom attention kernel needed.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
)

logger = init_logger(__name__)


class TurboQuantAttentionBackend(FlashInferBackend):
    """TQ backend: compressed KV-cache via FlashInfer."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3",
        "tq4",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "tq3",
    ) -> tuple[int, ...]:
        """Return compressed cache shape.

        head_size is replaced by tq_packed_size (e.g. 100 for TQ3/D=256).
        dtype will be uint8 → each element is 1 byte.
        """
        from vllm.turboquant.config import TurboQuantConfig
        if cache_dtype_str.startswith("tq"):
            tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
            packed_head = tq_config.cache_head_size
            return (num_blocks, 2, block_size, num_kv_heads, packed_head)
        # Fallback to FlashInfer default
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in ("tq3", "tq4")

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return FlashInferBackend.supports_attn_type(attn_type)

    @staticmethod
    def get_impl_cls() -> type["TurboQuantImpl"]:
        return TurboQuantImpl

    # Inherit everything else from FlashInfer:
    # get_builder_cls, get_supported_kernel_block_sizes, etc.


class TurboQuantImpl(FlashInferImpl):
    """TQ attention: compressed cache + on-demand decompression."""

    def __init__(self, *args, **kwargs):
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self._tq_cache_dtype = kv_cache_dtype
        self._tq_enabled = kv_cache_dtype.startswith("tq")

        # FlashInfer needs "auto" as dtype for its internal setup
        if self._tq_enabled:
            kwargs["kv_cache_dtype"] = "auto"
        super().__init__(*args, **kwargs)

    # TODO: Override do_kv_cache_update to compress K+V into packed cache
    # TODO: Override forward to decompress K+V before FlashInfer attention
