# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend for vLLM.

Wraps FlashAttentionBackend, adding TQ quantize→dequant round-trip on keys.
Cache format stays FP16-compatible (FlashAttention reads it directly).
Quality impact is measured by the lossy TQ round-trip on cached keys.

Phase 1: Proves integration + measures quality. No memory savings yet.
Phase 2+: Custom cache format with actual compression.
"""

from typing import ClassVar

import torch

from vllm.attention import AttentionType
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class TurboQuantAttentionBackend(FlashAttentionBackend):
    """TQ backend: delegates everything to FlashAttention,
    adds TQ round-trip on keys for quality measurement."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3",
        "tq4",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantImpl"]:
        return TurboQuantImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        # Reuse FlashAttention's metadata builder entirely
        return FlashAttentionMetadataBuilder

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        if kv_cache_dtype in ("tq3", "tq4"):
            return True
        # Also accept what FlashAttention accepts (for fallback)
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    # Inherit everything else from FlashAttentionBackend:
    # get_kv_cache_shape, get_kv_cache_stride_order, get_supported_kernel_block_sizes, etc.


class TurboQuantImpl(FlashAttentionImpl):
    """TQ attention: FlashAttention + TQ round-trip on keys.

    On every forward pass, keys go through:
      key → TQ.quantize → TQ.dequantize → lossy_key
    Then lossy_key is stored in the standard FlashAttention KV cache.
    Attention computation is identical to FlashAttention.
    """

    def __init__(self, *args, **kwargs):
        # TQ dtypes map to "auto" for FlashAttention internals
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self._tq_cache_dtype = kv_cache_dtype
        if kv_cache_dtype.startswith("tq"):
            kwargs["kv_cache_dtype"] = "auto"
        super().__init__(*args, **kwargs)
        # Restore for reference
        self._tq_enabled = self._tq_cache_dtype.startswith("tq")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with TQ round-trip on keys."""
        if self._tq_enabled and hasattr(layer, "_tq_Pi"):
            key = self._tq_round_trip(key, layer)

        return super().forward(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    @torch.no_grad()
    def _tq_round_trip(
        self, key: torch.Tensor, layer: torch.nn.Module
    ) -> torch.Tensor:
        """Apply TQ quantize→dequant to keys (lossy compression round-trip).

        Args:
            key: (num_tokens, num_kv_heads, head_size)
            layer: attention layer with _tq_Pi, _tq_S, _tq_centroids buffers

        Returns:
            Reconstructed key with TQ compression artifacts.
        """
        Pi = layer._tq_Pi        # (head_size, head_size)
        S = layer._tq_S          # (head_size, head_size)
        centroids = layer._tq_centroids  # (n_centroids,)

        orig_dtype = key.dtype
        num_tokens, num_heads, head_size = key.shape
        flat = key.reshape(-1, head_size).float()

        # 1. Normalize
        vec_norm = torch.norm(flat, dim=-1, keepdim=True)
        x_hat = flat / (vec_norm + 1e-8)

        # 2. Rotate
        rotated = x_hat @ Pi.T

        # 3. Scalar quantize to nearest centroid
        diffs = rotated.unsqueeze(-1) - centroids  # (N, D, n_centroids)
        indices = diffs.abs().argmin(dim=-1)  # (N, D)

        # 4. Reconstruct MSE
        y_hat = centroids[indices]
        x_mse = y_hat @ Pi  # back to original space

        # 5. Residual + QJL
        residual = x_hat - x_mse
        res_norm = torch.norm(residual, dim=-1, keepdim=True)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        # 6. QJL reconstruction
        import math
        m = head_size
        correction_scale = math.sqrt(math.pi / 2) / m
        x_qjl = correction_scale * res_norm * (signs @ S)

        # 7. Combine
        x_recon = vec_norm * (x_mse + x_qjl)

        return x_recon.reshape(num_tokens, num_heads, head_size).to(orig_dtype)
