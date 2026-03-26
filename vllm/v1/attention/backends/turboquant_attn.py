# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend.

Decode: Custom compressed score kernel reads directly from packed K-cache.
Prefill: FlashInfer (unchanged).
V-cache: BF16 in standard vLLM cache (FlashInfer reads it).
K-cache: Side-buffer with packed TQ data + BF16 in standard cache for prefill.
"""

import math
from typing import ClassVar, Optional

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
)

logger = init_logger(__name__)


class TurboQuantAttentionBackend(FlashInferBackend):
    """TQ backend — inherits FlashInfer, adds compressed decode."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["tq3", "tq4"]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in ("tq3", "tq4")

    @staticmethod
    def get_impl_cls() -> type["TurboQuantImpl"]:
        return TurboQuantImpl

    @staticmethod
    def get_builder_cls():
        return TurboQuantMetadataBuilder

    # Inherit get_kv_cache_shape from FlashInfer (BF16, normal size)


class TurboQuantMetadataBuilder(FlashInferBackend.get_builder_cls()):
    """Accept TQ cache dtype by mapping to model dtype for FlashInfer."""

    def __init__(self, *args, **kwargs):
        kv_cache_spec = args[0] if args else kwargs.get("kv_cache_spec")
        if kv_cache_spec is not None and kv_cache_spec.dtype == torch.uint8:
            from dataclasses import replace
            patched = replace(kv_cache_spec, dtype=torch.bfloat16)
            if args:
                args = (patched,) + args[1:]
            else:
                kwargs["kv_cache_spec"] = patched
        super().__init__(*args, **kwargs)


class TurboQuantImpl(FlashInferImpl):
    """TQ decode: compressed score kernel for K, FlashInfer for prefill."""

    def __init__(self, *args, **kwargs):
        # Extract kv_cache_dtype (7th positional arg)
        if len(args) > 6:
            kv_cache_dtype = args[6]
        else:
            kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")

        self._tq_cache_dtype = kv_cache_dtype
        self._tq_enabled = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq")

        # FlashInfer needs "auto"
        if self._tq_enabled:
            if len(args) > 6:
                args = list(args)
                args[6] = "auto"
                args = tuple(args)
            else:
                kwargs["kv_cache_dtype"] = "auto"

        super().__init__(*args, **kwargs)

        if self._tq_enabled:
            from vllm.turboquant.config import TurboQuantConfig
            self._tq_config = TurboQuantConfig.from_cache_dtype(
                self._tq_cache_dtype, self.head_size)
            self._tq_k_packed_cache = None  # Lazy init

    # do_kv_cache_update: inherited from FlashInferImpl (standard path).
    # TQ round-trip is applied in unified_kv_cache_update hook BEFORE this.

    def _pack_keys_to_side_buffer(self, key, layer, kv_cache, slot_mapping):
        """Pack keys into compressed TQ format in a side buffer."""
        D = self.head_size
        packed_size = self._tq_config.key_packed_size
        mse_bits = self._tq_config.mse_bits
        num_blocks = kv_cache.shape[0]
        block_size = kv_cache.shape[2]
        num_kv_heads = key.shape[1]

        # Lazy init side buffer
        if self._tq_k_packed_cache is None or self._tq_k_packed_cache.shape[0] != num_blocks:
            self._tq_k_packed_cache = torch.zeros(
                num_blocks, block_size, num_kv_heads, packed_size,
                dtype=torch.uint8, device=key.device)

        # Get cached matrices
        device = key.device
        if not hasattr(layer, '_tq_Pi_f32'):
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()

        Pi, S, centroids = layer._tq_Pi_f32, layer._tq_S_f32, layer._tq_c_f32
        num_tokens = key.shape[0]

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            bi = slot // block_size
            bo = slot % block_size
            for h in range(num_kv_heads):
                packed = self._pack_one_key(key[i, h], Pi, S, centroids, D, mse_bits)
                self._tq_k_packed_cache[bi, bo, h, :len(packed)] = packed

    @staticmethod
    def _pack_one_key(key_vec, Pi, S, centroids, D, mse_bits):
        """Pack one key vector to TQ format."""
        mse_bytes = (D * mse_bits + 7) // 8
        qjl_bytes = (D + 7) // 8

        x = key_vec.float()
        vn = x.norm(); xh = x / (vn + 1e-8)
        rot = xh @ Pi.T
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
        xm = centroids[idx.long()] @ Pi
        r = xh - xm; rn = r.norm()
        signs = (r @ S.T >= 0).to(torch.uint8)

        packed = torch.zeros(mse_bytes + qjl_bytes + 4, dtype=torch.uint8, device=key_vec.device)
        # Pack MSE indices
        if mse_bits == 2:
            for j in range(0, D, 4):
                v = 0
                for k in range(min(4, D - j)):
                    v |= (idx[j + k].item() & 0x3) << (k * 2)
                packed[j // 4] = v
        # Pack signs
        for j in range(0, D, 8):
            v = 0
            for k in range(min(8, D - j)):
                v |= (signs[j + k].item() & 1) << k
            packed[mse_bytes + j // 8] = v
        # Norms
        packed[mse_bytes + qjl_bytes:mse_bytes + qjl_bytes + 2] = vn.half().reshape(1).view(torch.uint8)
        packed[mse_bytes + qjl_bytes + 2:mse_bytes + qjl_bytes + 4] = rn.half().reshape(1).view(torch.uint8)
        return packed

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode: compressed score kernel. Prefill: FlashInfer."""
        # For now, always use FlashInfer (compressed decode TODO)
        # The compressed score kernel is validated standalone.
        # Integration into the FlashInfer forward flow requires
        # splitting prefill/decode and replacing the decode path.
        return super().forward(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale)
