# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend.

Uses FlashInfer with FP8 cache as the base. Adds:
1. TQ round-trip on keys (quality improvement)
2. Packed K side-buffer (for future compressed score kernel)

--kv-cache-dtype tq3 → internally uses fp8 cache + tq round-trip.
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
    """TQ backend = FlashInfer + TQ round-trip."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["tq3", "tq4"]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype=None) -> bool:
        return kv_cache_dtype in ("tq3", "tq4") if kv_cache_dtype else False

    @staticmethod
    def get_impl_cls():
        return TurboQuantImpl

    # Inherit everything else from FlashInfer:
    # get_kv_cache_shape, get_builder_cls, get_supported_kernel_block_sizes, etc.


class TurboQuantImpl(FlashInferImpl):
    """FlashInfer FP8 + TQ round-trip on keys."""

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """TQ round-trip on keys, then standard FlashInfer FP8 store."""
        if getattr(layer, '_kv_storage_dtype', None) and hasattr(layer, '_tq_Pi'):
            from vllm.turboquant.quantizer import tq_round_trip_keys
            key = tq_round_trip_keys(key, layer)

        # Standard FlashInfer FP8 cache write
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
