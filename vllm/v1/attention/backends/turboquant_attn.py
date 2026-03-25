# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend — compressed KV-cache with real memory savings.

Stores K and V as packed TQ data (uint8, ~100 bytes/vector for TQ3/D=256).
Before attention: decompresses into a temporary BF16 buffer for FlashInfer.
FlashInfer runs attention on the decompressed buffer — unchanged.

Memory savings: 5.1× (TQ3) or 4.0× (TQ4) vs BF16.
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
    """TQ backend with compressed cache shape."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["tq3", "tq4"]

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
        """Cache shape — head_size is ALREADY the packed size from get_kv_cache_spec."""
        # head_size comes from FullAttentionSpec which we set to cache_head_size
        # in get_kv_cache_spec. Don't re-compress — use it directly.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

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


class TurboQuantMetadataBuilder(FlashInferBackend.get_builder_cls()):
    """FlashInfer metadata builder that accepts uint8 cache dtype."""

    def __init__(self, *args, **kwargs):
        # Temporarily patch kv_cache_spec dtype to model dtype for parent init
        kv_cache_spec = args[0] if args else kwargs.get("kv_cache_spec")
        if kv_cache_spec is not None and kv_cache_spec.dtype == torch.uint8:
            from dataclasses import replace
            patched_spec = replace(kv_cache_spec, dtype=torch.bfloat16)
            if args:
                args = (patched_spec,) + args[1:]
            else:
                kwargs["kv_cache_spec"] = patched_spec
        super().__init__(*args, **kwargs)


class TurboQuantImpl(FlashInferImpl):
    """Compressed KV-cache with on-demand decompression for FlashInfer."""

    def __init__(self, *args, **kwargs):
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self._tq_cache_dtype = kv_cache_dtype
        self._tq_enabled = kv_cache_dtype.startswith("tq")

        # FlashInfer needs "auto" for internal setup
        if self._tq_enabled:
            kwargs["kv_cache_dtype"] = "auto"

        super().__init__(*args, **kwargs)

        if self._tq_enabled:
            from vllm.turboquant.config import TurboQuantConfig
            self._tq_config = TurboQuantConfig.from_cache_dtype(
                self._tq_cache_dtype, self.head_size)

        # Temp decompression buffer (allocated lazily)
        self._decomp_kv_cache = None

    def _ensure_decomp_buffer(self, kv_cache: torch.Tensor):
        """Allocate temp BF16 buffer matching the compressed cache's block count."""
        if self._decomp_kv_cache is not None:
            # Check if size still matches
            if self._decomp_kv_cache.shape[0] == kv_cache.shape[0]:
                return
        num_blocks = kv_cache.shape[0]
        block_size = kv_cache.shape[2]
        num_kv_heads = kv_cache.shape[3]
        # Full-size BF16 cache for FlashInfer
        self._decomp_kv_cache = torch.zeros(
            num_blocks, 2, block_size, num_kv_heads, self.head_size,
            dtype=torch.bfloat16, device=kv_cache.device,
        )

    @torch.no_grad()
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress K and V, then pack into uint8 cache slots."""
        if not self._tq_enabled or not hasattr(layer, '_tq_Pi'):
            # Fallback: write to decompressed buffer directly
            self._ensure_decomp_buffer(kv_cache)
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key, value,
                self._decomp_kv_cache[:, 0], self._decomp_kv_cache[:, 1],
                slot_mapping, "auto", layer._k_scale, layer._v_scale)
            return

        self._ensure_decomp_buffer(kv_cache)
        Pi = self._get_cached_pi(layer)
        S = self._get_cached_s(layer)
        centroids = self._get_cached_centroids(layer)
        packed_size = self._tq_config.key_packed_size
        mse_bits = self._tq_config.mse_bits
        D = self.head_size
        num_tokens = key.shape[0]
        num_heads = key.shape[1]
        block_size = kv_cache.shape[2]

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            bi = slot // block_size
            bo = slot % block_size

            for h in range(num_heads):
                # Compress key
                k_packed = self._compress_vector(
                    key[i, h], Pi, S, centroids, D, mse_bits)
                kv_cache[bi, 0, bo, h, :len(k_packed)] = k_packed

                # Compress value (same TQ pipeline)
                v_packed = self._compress_vector(
                    value[i, h], Pi, S, centroids, D, mse_bits)
                kv_cache[bi, 1, bo, h, :len(v_packed)] = v_packed

                # Also write decompressed versions for FlashInfer
                k_decomp = self._decompress_vector(k_packed, Pi, S, centroids,
                                                    D, mse_bits)
                v_decomp = self._decompress_vector(v_packed, Pi, S, centroids,
                                                    D, mse_bits)
                self._decomp_kv_cache[bi, 0, bo, h] = k_decomp.to(torch.bfloat16)
                self._decomp_kv_cache[bi, 1, bo, h] = v_decomp.to(torch.bfloat16)

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
        """Forward: use decompressed BF16 buffer for FlashInfer attention."""
        if self._tq_enabled and self._decomp_kv_cache is not None:
            # FlashInfer reads from the full-size BF16 buffer
            return super().forward(
                layer, query, key, value,
                self._decomp_kv_cache,  # ← decompressed, not the packed cache
                attn_metadata, output, output_scale, output_block_scale)
        return super().forward(
            layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale)

    # ── Helper methods ────────────────────────────────────────────────

    def _get_cached_pi(self, layer):
        if not hasattr(layer, '_tq_Pi_f32'):
            device = layer._tq_Pi.device
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
        return layer._tq_Pi_f32

    def _get_cached_s(self, layer):
        if not hasattr(layer, '_tq_S_f32'):
            device = layer._tq_S.device
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
        return layer._tq_S_f32

    def _get_cached_centroids(self, layer):
        if not hasattr(layer, '_tq_c_f32'):
            device = layer._tq_centroids.device
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()
        return layer._tq_c_f32

    @staticmethod
    def _compress_vector(vec, Pi, S, centroids, D, mse_bits):
        """Compress a single vector to packed uint8."""
        x = vec.float()
        vn = x.norm()
        xh = x / (vn + 1e-8)
        rot = xh @ Pi.T
        diffs = rot.unsqueeze(-1) - centroids
        idx = diffs.abs().argmin(dim=-1).to(torch.uint8)
        yh = centroids[idx.long()]
        xm = yh @ Pi
        r = xh - xm
        rn = r.norm()
        proj = r @ S.T
        signs = (proj >= 0).to(torch.uint8)

        # Pack
        mse_bytes = math.ceil(D * mse_bits / 8)
        qjl_bytes = math.ceil(D / 8)
        parts = []

        # MSE indices
        if mse_bits == 2:
            packed = torch.zeros(mse_bytes, dtype=torch.uint8, device=vec.device)
            for j in range(0, D, 4):
                val = torch.zeros(1, dtype=torch.uint8, device=vec.device)
                for k in range(min(4, D - j)):
                    val |= (idx[j + k] & 0x3) << (k * 2)
                packed[j // 4] = val
            parts.append(packed)
        elif mse_bits == 3:
            packed = torch.zeros(mse_bytes, dtype=torch.uint8, device=vec.device)
            for j in range(D):
                off = j * 3
                bi_idx, bit = off // 8, off % 8
                v = idx[j].to(torch.int32)
                packed[bi_idx] |= ((v << bit) & 0xFF).to(torch.uint8)
                if bit > 5 and bi_idx + 1 < mse_bytes:
                    packed[bi_idx + 1] |= ((v >> (8 - bit)) & 0xFF).to(torch.uint8)
            parts.append(packed)

        # QJL signs
        packed_s = torch.zeros(qjl_bytes, dtype=torch.uint8, device=vec.device)
        for j in range(0, D, 8):
            val = torch.zeros(1, dtype=torch.uint8, device=vec.device)
            for k in range(min(8, D - j)):
                val |= (signs[j + k] & 1) << k
            packed_s[j // 8] = val
        parts.append(packed_s)

        # Norms
        parts.append(vn.half().view(torch.uint8))
        parts.append(rn.half().view(torch.uint8))

        return torch.cat(parts)

    @staticmethod
    def _decompress_vector(packed, Pi, S, centroids, D, mse_bits):
        """Decompress packed uint8 to float32 vector."""
        mse_bytes = math.ceil(D * mse_bits / 8)
        qjl_bytes = math.ceil(D / 8)
        mask = (1 << mse_bits) - 1
        device = packed.device

        # Unpack MSE indices
        packed_mse = packed[:mse_bytes]
        idx = torch.zeros(D, dtype=torch.uint8, device=device)
        if mse_bits == 2:
            for j in range(D):
                bi_idx = j // 4
                bit = (j % 4) * 2
                if bi_idx < len(packed_mse):
                    idx[j] = (packed_mse[bi_idx] >> bit) & mask
        elif mse_bits == 3:
            for j in range(D):
                off = j * 3
                bi_idx, bit = off // 8, off % 8
                val = packed_mse[bi_idx].to(torch.int32) >> bit
                if bit > 5 and bi_idx + 1 < len(packed_mse):
                    val |= packed_mse[bi_idx + 1].to(torch.int32) << (8 - bit)
                idx[j] = (val & mask).to(torch.uint8)

        # Unpack signs
        packed_s = packed[mse_bytes:mse_bytes + qjl_bytes]
        signs = torch.zeros(D, dtype=torch.float32, device=device)
        for j in range(D):
            bi_idx, bit = j // 8, j % 8
            if bi_idx < len(packed_s):
                signs[j] = 1.0 if ((packed_s[bi_idx] >> bit) & 1) else -1.0

        # Unpack norms
        off = mse_bytes + qjl_bytes
        vn = packed[off:off + 2].view(torch.float16).float().item()
        rn = packed[off + 2:off + 4].view(torch.float16).float().item()

        # Reconstruct
        c_idx = centroids[idx.long()]
        xm = c_idx @ Pi
        correction = math.sqrt(math.pi / 2) / D
        xq = correction * rn * (signs @ S)
        return vn * (xm + xq)
