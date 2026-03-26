# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention backend — compressed KV-cache, custom decode attention.

Cache: uint8, head_size = tq_packed_size (28 bytes for TQ3/D=64).
Decode: Compressed score kernel reads packed K directly. No decompression.
Prefill: Standard causal attention on raw K/V (no cache read needed).
Store: Custom packing into uint8 slots (no reshape_and_cache_flash).
"""

import math
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.logger import init_logger

try:
    from vllm.attention import AttentionType
except ImportError:
    from vllm.v1.attention.backend import AttentionType

from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
)

logger = init_logger(__name__)


class TurboQuantAttentionBackend(FlashInferBackend):
    """TQ backend with compressed cache."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["tq3", "tq4"]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size,
        cache_dtype_str="tq3",
    ) -> tuple[int, ...]:
        # head_size is ALREADY tq_packed_size from get_kv_cache_spec
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype=None) -> bool:
        return kv_cache_dtype in ("tq3", "tq4") if kv_cache_dtype else False

    @staticmethod
    def get_impl_cls():
        return TurboQuantImpl

    @staticmethod
    def get_builder_cls():
        return TurboQuantMetadataBuilder


class TurboQuantMetadataBuilder(FlashInferBackend.get_builder_cls()):
    """Patch spec to BF16 + real head_size for FlashInfer autotuner."""
    def __init__(self, *args, **kwargs):
        spec = args[0] if args else kwargs.get("kv_cache_spec")
        if spec is not None and spec.dtype == torch.uint8:
            from dataclasses import replace
            from vllm.turboquant.config import TurboQuantConfig
            # Recover real head_size from the packed head_size
            # The packed head_size was set in get_kv_cache_spec.
            # We need the ORIGINAL head_size for FlashInfer.
            # TQ3/D=64: packed=28, TQ3/D=128: packed=52, TQ3/D=256: packed=100
            # Reverse: find head_dim where TurboQuantConfig.key_packed_size == spec.head_size
            real_head = None
            for d in [64, 96, 128, 192, 256]:
                for bits_name in ["tq3", "tq4"]:
                    tc = TurboQuantConfig.from_cache_dtype(bits_name, d)
                    if tc.key_packed_size == spec.head_size:
                        real_head = d
                        break
                if real_head:
                    break
            if real_head is None:
                real_head = spec.head_size  # fallback
            patched = replace(spec, dtype=torch.bfloat16,
                              head_size=real_head, head_size_v=real_head)
            if args:
                args = (patched,) + args[1:]
            else:
                kwargs["kv_cache_spec"] = patched
        super().__init__(*args, **kwargs)


class TurboQuantImpl(FlashInferImpl):
    """TQ decode attention from compressed cache."""

    def __init__(self, *args, **kwargs):
        kv_cache_dtype = args[6] if len(args) > 6 else kwargs.get("kv_cache_dtype", "auto")
        self._tq_cache_dtype = kv_cache_dtype
        self._tq_enabled = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq")

        # FlashInfer needs "auto" for its internal setup
        if self._tq_enabled:
            if len(args) > 6:
                args = list(args); args[6] = "auto"; args = tuple(args)
            else:
                kwargs["kv_cache_dtype"] = "auto"
        super().__init__(*args, **kwargs)

        if self._tq_enabled:
            from vllm.turboquant.config import TurboQuantConfig
            self._tq_config = TurboQuantConfig.from_cache_dtype(
                self._tq_cache_dtype, self.head_size)
            self._score_kernel = None  # Lazy JIT load

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Pack K+V into compressed uint8 cache slots."""
        if not self._tq_enabled or not hasattr(layer, '_tq_Pi'):
            # Fallback to FlashInfer standard
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key, value, kv_cache[:, 0], kv_cache[:, 1],
                slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale)
            return

        D = self.head_size
        mse_bits = self._tq_config.mse_bits
        packed_size = self._tq_config.key_packed_size
        block_size = kv_cache.shape[2]
        device = key.device

        # Get cached matrices
        if not hasattr(layer, '_tq_Pi_f32'):
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()
        Pi, S, centroids = layer._tq_Pi_f32, layer._tq_S_f32, layer._tq_c_f32

        num_tokens, num_heads = key.shape[0], key.shape[1]
        mse_bytes = (D * mse_bits + 7) // 8
        qjl_bytes = (D + 7) // 8

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            bi, bo = slot // block_size, slot % block_size
            for h in range(num_heads):
                # Pack key
                k_packed = self._pack_vector(key[i, h], Pi, S, centroids, D, mse_bits, mse_bytes, qjl_bytes)
                kv_cache[bi, 0, bo, h, :len(k_packed)] = k_packed
                # Pack value
                v_packed = self._pack_vector(value[i, h], Pi, S, centroids, D, mse_bits, mse_bytes, qjl_bytes)
                kv_cache[bi, 1, bo, h, :len(v_packed)] = v_packed

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        """Decode: compressed score kernel. Prefill: raw Q·K·V."""
        assert output is not None

        if attn_metadata is None:
            return output.fill_(0)

        if not self._tq_enabled:
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, output_block_scale)

        num_decode = attn_metadata.num_decode_tokens
        num_prefill = attn_metadata.num_prefill_tokens

        # --- Prefill: standard scaled dot-product on raw K/V ---
        if num_prefill > 0:
            pq = query[num_decode:]
            pk = key[num_decode:]
            pv = value[num_decode:]
            # Simple causal attention (no cache read needed for initial prefill)
            scores = torch.matmul(pq.float(), pk.float().transpose(-2, -1))
            scores = scores * (1.0 / math.sqrt(self.head_size))
            # Causal mask
            L = scores.shape[-2]
            mask = torch.triu(torch.full((L, L), float('-inf'), device=scores.device), diagonal=1)
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            prefill_out = torch.matmul(weights, pv.float())
            output[num_decode:] = prefill_out.reshape(num_prefill, -1).to(output.dtype)

        # --- Decode: compressed score kernel ---
        if num_decode > 0:
            dq = query[:num_decode]  # (num_decode, num_heads, head_size)
            self._compressed_decode(layer, dq, kv_cache, attn_metadata, output)

        return output

    def _compressed_decode(self, layer, query, kv_cache, attn_metadata, output):
        """Compute decode attention from compressed KV cache."""
        D = self.head_size
        device = query.device
        num_q = query.shape[0]
        num_heads = query.shape[1]

        Pi = layer._tq_Pi_f32
        S = layer._tq_S_f32
        centroids = layer._tq_c_f32
        mse_bits = self._tq_config.mse_bits
        mse_bytes = (D * mse_bits + 7) // 8
        qjl_bytes = (D + 7) // 8
        correction = math.sqrt(math.pi / 2) / D
        attn_scale = 1.0 / math.sqrt(D)
        block_size = kv_cache.shape[2]
        packed_size = kv_cache.shape[4]
        mask = (1 << mse_bits) - 1

        # Precompute q_rot, q_proj
        q_rot = query.float() @ Pi.T   # (num_q, num_heads, D)
        q_proj = query.float() @ S.T   # (num_q, num_heads, D)

        # Get seq_lens and block_table from metadata
        seq_lens = attn_metadata.seq_lens_cpu if hasattr(attn_metadata, 'seq_lens_cpu') else attn_metadata.seq_lens
        block_table = attn_metadata.block_table_tensor if hasattr(attn_metadata, 'block_table_tensor') else attn_metadata.block_table

        # Per-query decode attention
        for qi in range(num_q):
            seq_len = seq_lens[qi].item() if torch.is_tensor(seq_lens) else seq_lens[qi]

            for h in range(num_heads):
                # Online softmax
                m_prev = float('-inf')
                d_prev = 0.0
                acc = torch.zeros(D, device=device)

                for pos in range(seq_len):
                    bi = pos // block_size
                    bo = pos % block_size
                    phys_block = block_table[qi, bi].item()

                    # Score from compressed K
                    k_packed = kv_cache[phys_block, 0, bo, h]
                    score = self._score_from_packed(
                        q_rot[qi, h], q_proj[qi, h], k_packed,
                        centroids, D, mse_bits, mse_bytes, qjl_bytes,
                        mask, correction, attn_scale)

                    # Decompress V
                    v_packed = kv_cache[phys_block, 1, bo, h]
                    v_val = self._decompress_vector(
                        v_packed, Pi, S, centroids,
                        D, mse_bits, mse_bytes, qjl_bytes, mask, correction)

                    # Online softmax
                    m_new = max(m_prev, score)
                    exp_prev = math.exp(m_prev - m_new)
                    exp_cur = math.exp(score - m_new)
                    d_new = d_prev * exp_prev + exp_cur
                    acc = acc * exp_prev + exp_cur * v_val
                    m_prev, d_prev = m_new, d_new

                if d_prev > 0:
                    output[qi, h * D:(h + 1) * D] = (acc / d_prev).to(output.dtype)

    @staticmethod
    def _pack_vector(vec, Pi, S, centroids, D, mse_bits, mse_bytes, qjl_bytes):
        x = vec.float(); vn = x.norm(); xh = x / (vn + 1e-8)
        rot = xh @ Pi.T
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
        xm = centroids[idx.long()] @ Pi; r = xh - xm; rn = r.norm()
        signs = (r @ S.T >= 0).to(torch.uint8)

        packed = torch.zeros(mse_bytes + qjl_bytes + 4, dtype=torch.uint8, device=vec.device)
        if mse_bits == 2:
            for j in range(0, D, 4):
                v = 0
                for k in range(min(4, D - j)):
                    v |= (idx[j+k].item() & 0x3) << (k*2)
                packed[j//4] = v
        for j in range(0, D, 8):
            v = 0
            for k in range(min(8, D - j)):
                v |= (signs[j+k].item() & 1) << k
            packed[mse_bytes + j//8] = v
        packed[mse_bytes+qjl_bytes:mse_bytes+qjl_bytes+2] = vn.half().reshape(1).view(torch.uint8)
        packed[mse_bytes+qjl_bytes+2:mse_bytes+qjl_bytes+4] = rn.half().reshape(1).view(torch.uint8)
        return packed

    @staticmethod
    def _score_from_packed(q_rot, q_proj, packed, centroids, D, mse_bits,
                           mse_bytes, qjl_bytes, mask, correction, attn_scale):
        """Compute attention score from packed K-vector."""
        term1 = 0.0
        if mse_bits == 2:
            for b in range(mse_bytes):
                bv = packed[b].item()
                for k in range(4):
                    j = b*4+k
                    if j >= D: break
                    idx = (bv >> (k*2)) & mask
                    term1 += q_rot[j].item() * centroids[idx].item()

        term2 = 0.0
        for b in range(qjl_bytes):
            bv = packed[mse_bytes + b].item()
            for k in range(8):
                j = b*8+k
                if j >= D: break
                sv = 1.0 if ((bv >> k) & 1) else -1.0
                term2 += q_proj[j].item() * sv

        import struct
        no = mse_bytes + qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]

        return vn * (term1 + correction * rn * term2) * attn_scale

    @staticmethod
    def _decompress_vector(packed, Pi, S, centroids, D, mse_bits,
                           mse_bytes, qjl_bytes, mask, correction):
        """Decompress packed vector to float32."""
        device = packed.device
        idx = torch.zeros(D, dtype=torch.long, device=device)
        if mse_bits == 2:
            for j in range(D):
                b, k = j//4, j%4
                idx[j] = (packed[b].item() >> (k*2)) & mask

        signs = torch.zeros(D, dtype=torch.float32, device=device)
        for j in range(D):
            b, k = j//8, j%8
            signs[j] = 1.0 if ((packed[mse_bytes+b].item() >> k) & 1) else -1.0

        import struct
        no = mse_bytes + qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]

        c_idx = centroids[idx]
        xm = c_idx @ Pi
        xq = correction * rn * (signs @ S)
        return vn * (xm + xq)
