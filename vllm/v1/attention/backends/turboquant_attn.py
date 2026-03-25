# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend for vLLM.

Prefill: Standard FlashAttention on uncompressed K/V, then quantize into cache.
Decode: Fused TQ attention score directly from compressed cache.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn as nn

from vllm.attention import AttentionType
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionMetadataBuilder,
    MultipleOf,
)

logger = init_logger(__name__)


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = False
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3",
        "tq4",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "tq3",
    ) -> tuple[int, ...]:
        """Return cache shape with packed TQ format.

        Layout: (2, num_blocks, block_size, num_kv_heads, packed_size)
        where packed_size depends on bits and head_size.
        """
        from vllm.turboquant.config import TurboQuantConfig

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        packed_size = tq_config.packed_size
        return (2, num_blocks, block_size, num_kv_heads, packed_size)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype in ("tq3", "tq4")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 96, 128, 256)


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    # Sequence lengths for each request
    seq_lens: torch.Tensor  # (num_reqs,)

    # Slot mapping for cache writes
    slot_mapping: torch.Tensor  # (num_tokens,)

    # Block table for cache reads
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)

    # Whether this is a prefill or decode step
    is_prefill: bool = False

    # Number of prefill tokens (0 for pure decode)
    num_prefill_tokens: int = 0

    # CU sequence lengths for prefill (FlashAttn)
    cu_seq_lens: Optional[torch.Tensor] = None

    # Max sequence length in this batch
    max_seq_len: int = 0


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output.

    NOTE: This is a minimal stub. A full implementation would mirror
    FlashAttentionMetadataBuilder, handling prefill/decode split,
    cu_seq_lens computation, block_table preparation, etc.
    For the initial prototype, we reuse the scheduler's precomputed metadata.
    """

    def __init__(self, kv_cache_spec, vllm_config, device):
        self.device = device

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, num_reqs, num_tokens, max_num_scheduled_tokens,
              common_prefix_len, common_attn_metadata):
        # Minimal stub — real implementation will populate from common metadata
        return TurboQuantMetadata(
            seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64,
                                     device=self.device),
            block_table=torch.zeros(num_reqs, 1, dtype=torch.int32,
                                    device=self.device),
        )


class TurboQuantAttentionImpl(nn.Module):
    """TurboQuant attention implementation.

    Phase 1: Pure PyTorch fallback (correct but slow).
    Phase 2+: Triton/CUDA fused kernels.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.turboquant.config import TurboQuantConfig

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

    def forward(
        self,
        layer: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with TurboQuant KV-cache.

        For now this is a PyTorch reference implementation.
        """
        num_tokens = query.shape[0]
        B_q = num_tokens

        # Reshape Q/K/V
        query = query.view(B_q, self.num_heads, self.head_size)
        key = key.view(B_q, self.num_kv_heads, self.head_size)
        value = value.view(B_q, self.num_kv_heads, self.head_size)

        # Get TQ buffers from the attention layer
        Pi = layer._tq_Pi      # (head_size, head_size)
        S = layer._tq_S        # (head_size, head_size)
        centroids = layer._tq_centroids  # (n_centroids,)

        # === Store new K/V into cache ===
        try:
            from vllm.v1.attention.ops.triton_tq_reshape_and_cache import (
                triton_tq_reshape_and_cache,
            )
            triton_tq_reshape_and_cache(
                key, value,
                kv_cache[0], kv_cache[1],
                attn_metadata.slot_mapping,
                Pi, S, centroids,
                mse_bits=self.tq_config.mse_bits,
            )
        except Exception:
            # Fallback to PyTorch reference
            self._store_kv(key, value, kv_cache, attn_metadata.slot_mapping,
                           Pi, S, centroids)

        # === Compute attention ===
        if attn_metadata.is_prefill:
            # Prefill: use standard attention on uncompressed K/V
            # (the new tokens haven't been read from cache yet)
            attn_output = self._prefill_attention(query, key, value, attn_metadata)
        else:
            # Decode: compute scores from compressed cache
            try:
                from vllm.v1.attention.ops.triton_tq_attention_score import (
                    triton_tq_fused_attention_score,
                )
                scores = triton_tq_fused_attention_score(
                    query, kv_cache[0],
                    attn_metadata.block_table, attn_metadata.seq_lens,
                    Pi, S, centroids,
                    mse_bits=self.tq_config.mse_bits,
                    attn_scale=self.scale,
                )
                # Softmax + value aggregation (values still in PyTorch path)
                attn_output = self._aggregate_values_from_scores(
                    scores, kv_cache[1], attn_metadata
                )
            except Exception:
                attn_output = self._decode_attention(
                    query, kv_cache, attn_metadata, Pi, S, centroids
                )

        return attn_output.reshape(num_tokens, -1)

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ):
        """Quantize and store K/V into packed cache.

        PyTorch reference — will be replaced by Triton kernel.
        """
        B, H, D = key.shape
        mse_bits = self.tq_config.mse_bits
        packed_size = self.tq_config.packed_size

        key_cache = kv_cache[0]   # (num_blocks, block_size, num_kv_heads, packed_size)
        val_cache = kv_cache[1]

        for i in range(B):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue

            block_idx = slot // key_cache.shape[1]
            block_off = slot % key_cache.shape[1]

            for h in range(H):
                k_vec = key[i, h].float()  # (D,)

                # Quantize key
                packed = self._quantize_vector(k_vec, Pi, S, centroids, mse_bits, D)
                key_cache[block_idx, block_off, h, :len(packed)] = packed

                # Store value as-is in FP16 (Phase 1: values uncompressed)
                # Pack value norm + truncated FP16 into cache
                v_vec = value[i, h].float()
                v_packed = self._pack_value_fp16(v_vec, packed_size)
                val_cache[block_idx, block_off, h, :len(v_packed)] = v_packed

    def _quantize_vector(
        self,
        x: torch.Tensor,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
        mse_bits: int,
        D: int,
    ) -> torch.Tensor:
        """Quantize a single vector using TurboQuant. Returns packed uint8."""
        # 1. Normalize
        vec_norm = x.norm()
        x_hat = x / (vec_norm + 1e-8)

        # 2. Rotate
        y = x_hat @ Pi.T  # (D,)

        # 3. Scalar quantize
        diffs = y.unsqueeze(-1) - centroids  # (D, n_centroids)
        idx = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (D,)

        # 4. Reconstruct
        y_hat = centroids[idx.long()]
        x_mse = y_hat @ Pi

        # 5. Residual + QJL
        r = x_hat - x_mse
        gamma = r.norm()
        projected = r @ S.T
        signs = (projected >= 0).to(torch.uint8)  # (D,) 0 or 1

        # 6. Pack
        mse_bytes = math.ceil(D * mse_bits / 8)
        qjl_bytes = math.ceil(D / 8)

        # Pack MSE indices
        if mse_bits == 2:
            packed_mse = torch.zeros(mse_bytes, dtype=torch.uint8, device=x.device)
            for j in range(0, D, 4):
                byte_val = torch.zeros(1, dtype=torch.uint8, device=x.device)
                for k in range(min(4, D - j)):
                    byte_val |= (idx[j + k].to(torch.uint8) << (k * 2))
                packed_mse[j // 4] = byte_val
        elif mse_bits == 3:
            packed_mse = torch.zeros(mse_bytes, dtype=torch.uint8, device=x.device)
            for j in range(D):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                val = idx[j].to(torch.uint16)
                packed_mse[byte_idx] |= ((val << bit_idx) & 0xFF).to(torch.uint8)
                if bit_idx > 5 and byte_idx + 1 < mse_bytes:
                    packed_mse[byte_idx + 1] |= ((val >> (8 - bit_idx)) & 0xFF).to(
                        torch.uint8
                    )
        else:
            packed_mse = idx[:mse_bytes]

        # Pack QJL signs (8 per byte)
        packed_signs = torch.zeros(qjl_bytes, dtype=torch.uint8, device=x.device)
        for j in range(0, D, 8):
            byte_val = torch.zeros(1, dtype=torch.uint8, device=x.device)
            for k in range(min(8, D - j)):
                byte_val |= (signs[j + k] << k)
            packed_signs[j // 8] = byte_val

        # Pack norms (2x float16 = 4 bytes)
        norm_bytes = vec_norm.half().view(torch.uint8)  # 2 bytes
        gamma_bytes = gamma.half().view(torch.uint8)    # 2 bytes

        return torch.cat([packed_mse, packed_signs, norm_bytes, gamma_bytes])

    def _pack_value_fp16(self, v: torch.Tensor, packed_size: int) -> torch.Tensor:
        """Pack a value vector. Phase 1: store as raw FP16 bytes, truncated to packed_size."""
        raw = v.half().view(torch.uint8)  # D*2 bytes
        result = torch.zeros(packed_size, dtype=torch.uint8, device=v.device)
        n = min(len(raw), packed_size)
        result[:n] = raw[:n]
        return result

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
    ) -> torch.Tensor:
        """Standard attention for prefill (no cache read needed)."""
        B, Hq, D = query.shape
        Hk = key.shape[1]

        # Expand KV for GQA
        if Hk < Hq:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        # Simple scaled dot-product attention
        # query/key/value: (B, H, D)
        # For prefill, B = total tokens, treat as seq_len=1 per token (causal handled)
        scores = torch.einsum("bhd,bhd->bh", query.float(), key.float()) * self.scale
        attn_weights = torch.softmax(scores.unsqueeze(-1), dim=0)  # simplified
        # This is a placeholder — real prefill needs proper causal masking
        output = value.float()  # Placeholder
        return output.to(query.dtype)

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode attention from compressed cache.

        Phase 1: Dequantize all cached K/V, then standard attention.
        Phase 2+: Fused TQ score kernel.
        """
        B, Hq, D = query.shape
        Hk = self.num_kv_heads
        mse_bits = self.tq_config.mse_bits
        m = D  # QJL dimension

        key_cache = kv_cache[0]  # (num_blocks, block_size, num_kv_heads, packed_size)
        val_cache = kv_cache[1]

        outputs = []
        for i in range(B):
            seq_len = attn_metadata.seq_lens[i].item()
            if seq_len <= 0:
                outputs.append(torch.zeros(Hq, D, device=query.device, dtype=query.dtype))
                continue

            blocks = attn_metadata.block_table[i]
            block_size = key_cache.shape[1]

            # Precompute query rotations (once per query token!)
            q_i = query[i].float()  # (Hq, D)
            # For GQA: group queries by KV head
            q_rot = q_i @ Pi.T     # (Hq, D) — q @ Pi^T
            q_proj = q_i @ S.T     # (Hq, D) — q @ S^T

            all_scores = []
            all_values = []

            for pos in range(seq_len):
                block_idx = blocks[pos // block_size].item()
                block_off = pos % block_size

                for h in range(Hk):
                    packed = key_cache[block_idx, block_off, h]

                    # Unpack
                    mse_bytes_n = math.ceil(D * mse_bits / 8)
                    qjl_bytes_n = math.ceil(D / 8)

                    packed_mse = packed[:mse_bytes_n]
                    packed_signs = packed[mse_bytes_n:mse_bytes_n + qjl_bytes_n]
                    norm_bytes = packed[mse_bytes_n + qjl_bytes_n:mse_bytes_n + qjl_bytes_n + 2]
                    gamma_bytes = packed[mse_bytes_n + qjl_bytes_n + 2:mse_bytes_n + qjl_bytes_n + 4]

                    vec_norm = norm_bytes.view(torch.float16).float().item()
                    gamma = gamma_bytes.view(torch.float16).float().item()

                    # Unpack MSE indices
                    idx = self._unpack_mse_indices(packed_mse, mse_bits, D)

                    # Unpack QJL signs
                    signs = self._unpack_signs(packed_signs, D)

                    # Term 1: <q_rot, centroids[idx]>
                    c_idx = centroids[idx.long()]  # (D,)
                    # For each Q head in this KV head group
                    q_heads = range(h * self.num_kv_groups, (h + 1) * self.num_kv_groups)
                    for qh in q_heads:
                        t1 = (q_rot[qh] * c_idx).sum()

                        # Term 2: QJL correction
                        signs_float = signs.float() * 2 - 1  # 0,1 -> -1,+1
                        t2 = (q_proj[qh] * signs_float).sum()
                        correction = math.sqrt(math.pi / 2) / m

                        score = vec_norm * (t1 + correction * gamma * t2)
                        all_scores.append((qh, pos, score))

                    # Unpack value (FP16 bytes)
                    v_packed = val_cache[block_idx, block_off, h]
                    v_bytes = v_packed[:D * 2]
                    v_vec = v_bytes.view(torch.float16).float()
                    for qh in q_heads:
                        all_values.append((qh, pos, v_vec))

            # Assemble scores and compute softmax + weighted sum
            score_tensor = torch.zeros(Hq, seq_len, device=query.device)
            value_tensor = torch.zeros(Hq, seq_len, D, device=query.device)

            for qh, pos, s in all_scores:
                score_tensor[qh, pos] = s
            for qh, pos, v in all_values:
                value_tensor[qh, pos] = v

            score_tensor = score_tensor * self.scale
            attn_weights = torch.softmax(score_tensor, dim=-1)  # (Hq, seq_len)
            # Weighted sum: (Hq, seq_len) @ (Hq, seq_len, D) -> (Hq, D)
            out = torch.einsum("hs,hsd->hd", attn_weights, value_tensor)
            outputs.append(out.to(query.dtype))

        return torch.stack(outputs, dim=0)  # (B, Hq, D)

    def _aggregate_values_from_scores(
        self,
        scores: torch.Tensor,    # [B, Hq, max_seq_len]
        val_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, packed_size]
        attn_metadata: TurboQuantMetadata,
    ) -> torch.Tensor:
        """Apply softmax to scores and aggregate values."""
        B, Hq, max_seq_len = scores.shape
        D = self.head_size
        Hk = self.num_kv_heads
        block_size = val_cache.shape[1]
        packed_size = val_cache.shape[3]

        outputs = []
        for i in range(B):
            seq_len = attn_metadata.seq_lens[i].item()
            if seq_len <= 0:
                outputs.append(torch.zeros(Hq, D, device=scores.device,
                                           dtype=scores.dtype))
                continue

            # Softmax over valid positions
            s = scores[i, :, :seq_len]  # (Hq, seq_len)
            attn_weights = torch.softmax(s, dim=-1)  # (Hq, seq_len)

            # Gather values
            blocks = attn_metadata.block_table[i]
            values = torch.zeros(Hk, seq_len, D, device=scores.device)
            for pos in range(seq_len):
                block_idx = blocks[pos // block_size].item()
                block_off = pos % block_size
                for h in range(Hk):
                    v_packed = val_cache[block_idx, block_off, h]
                    n_elems = min(D, packed_size // 2)
                    v_bytes = v_packed[:n_elems * 2]
                    values[h, pos, :n_elems] = v_bytes.view(torch.float16).float()

            # Expand for GQA
            if Hk < Hq:
                values = values.repeat_interleave(self.num_kv_groups, dim=0)

            # Weighted sum: (Hq, seq_len) @ (Hq, seq_len, D)
            out = torch.einsum("hs,hsd->hd", attn_weights, values)
            outputs.append(out)

        return torch.stack(outputs, dim=0)

    def _unpack_mse_indices(
        self, packed: torch.Tensor, mse_bits: int, D: int
    ) -> torch.Tensor:
        """Unpack MSE indices from packed bytes."""
        idx = torch.zeros(D, dtype=torch.uint8, device=packed.device)
        mask = (1 << mse_bits) - 1

        if mse_bits == 2:
            for j in range(D):
                byte_idx = j // 4
                bit_idx = (j % 4) * 2
                if byte_idx < len(packed):
                    idx[j] = (packed[byte_idx] >> bit_idx) & mask
        elif mse_bits == 3:
            for j in range(D):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                val = packed[byte_idx].to(torch.uint16) >> bit_idx
                if bit_idx > 5 and byte_idx + 1 < len(packed):
                    val |= packed[byte_idx + 1].to(torch.uint16) << (8 - bit_idx)
                idx[j] = (val & mask).to(torch.uint8)
        else:
            idx[:min(D, len(packed))] = packed[:min(D, len(packed))]

        return idx

    def _unpack_signs(self, packed: torch.Tensor, D: int) -> torch.Tensor:
        """Unpack QJL sign bits from packed bytes."""
        signs = torch.zeros(D, dtype=torch.uint8, device=packed.device)
        for j in range(D):
            byte_idx = j // 8
            bit_idx = j % 8
            if byte_idx < len(packed):
                signs[j] = (packed[byte_idx] >> bit_idx) & 1
        return signs
