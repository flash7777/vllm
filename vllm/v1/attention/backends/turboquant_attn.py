# SPDX-License-Identifier: Apache-2.0
"""TurboQuant custom attention backend — compressed uint8 KV-cache.

Standalone backend, does NOT inherit FlashInfer.
Reads/writes compressed TQ data directly. No shadow cache.

Cache: (num_blocks, 2, block_size, num_kv_heads, packed_size) uint8
Decode: compressed score + online softmax + V decompress (Python, CUDA later)
Prefill: naive causal attention on raw K/V
"""

import math
import struct
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)

try:
    from vllm.attention import AttentionType
except ImportError:
    from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)


# ============================================================
# Metadata
# ============================================================

@dataclass
class TQMetadata(AttentionMetadata):
    seq_lens: torch.Tensor          # [batch]
    block_table: torch.Tensor       # [batch, max_blocks]
    slot_mapping: torch.Tensor      # [num_tokens]
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    max_seq_len: int = 0
    query_start_loc: Optional[torch.Tensor] = None


# ============================================================
# Backend
# ============================================================

class TurboQuantAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["tq3", "tq4"]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype=None) -> bool:
        return kv_cache_dtype in ("tq3", "tq4") if kv_cache_dtype else False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True  # packed_size is passed as head_size

    @staticmethod
    def get_impl_cls():
        return TurboQuantImpl

    @staticmethod
    def get_builder_cls():
        return TQMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="tq3"):
        # head_size is already packed_size from get_kv_cache_spec
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


# ============================================================
# Metadata Builder
# ============================================================

class TQMetadataBuilder(AttentionMetadataBuilder[TQMetadata]):

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        # Determine prefill vs decode
        num_tokens = cam.num_actual_tokens
        num_reqs = cam.num_reqs
        # Heuristic: if max_query_len > 1, there are prefill tokens
        num_prefill = 0
        num_decode = num_tokens
        if cam.max_query_len > 1:
            num_prefill = num_tokens
            num_decode = 0

        return TQMetadata(
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
            max_seq_len=cam.max_seq_len,
            query_start_loc=cam.query_start_loc,
        )


# ============================================================
# Impl
# ============================================================

class TurboQuantImpl:
    """Custom TQ attention — no FlashInfer dependency."""

    supports_quant_query_input: bool = False
    can_return_lse_for_decode: bool = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="tq3",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size  # This is the REAL head_dim, not packed
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        from vllm.turboquant.config import TurboQuantConfig
        self._tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self._packed_size = self._tq_config.key_packed_size
        self._mse_bits = self._tq_config.mse_bits
        self._mse_bytes = (head_size * self._mse_bits + 7) // 8
        self._qjl_bytes = (head_size + 7) // 8
        self._mask = (1 << self._mse_bits) - 1
        self._correction = math.sqrt(math.pi / 2) / head_size

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Pack K+V into compressed uint8 cache."""
        if self.kv_sharing_target_layer_name is not None:
            return

        D = self.head_size
        device = key.device
        block_size = kv_cache.shape[2]

        Pi, S, centroids = self._get_matrices(layer, device)

        num_tokens, num_heads = key.shape[0], key.shape[1]
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            bi, bo = slot // block_size, slot % block_size
            for h in range(num_heads):
                kv_cache[bi, 0, bo, h, :self._packed_size] = self._pack(
                    key[i, h], Pi, S, centroids, D)
                kv_cache[bi, 1, bo, h, :self._packed_size] = self._pack(
                    value[i, h], Pi, S, centroids, D)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        """Decode: compressed attention. Prefill: naive causal."""
        if output is None:
            output = torch.empty(query.shape[0], self.num_heads * self.head_size,
                                 device=query.device, dtype=query.dtype)
        if attn_metadata is None:
            return output.fill_(0)

        D = self.head_size
        device = query.device
        Pi, S, centroids = self._get_matrices(layer, device)
        block_size = kv_cache.shape[2]

        num_prefill = attn_metadata.num_prefill_tokens
        num_decode = attn_metadata.num_decode_tokens

        # --- Prefill: naive causal on raw K/V ---
        if num_prefill > 0:
            pq = query[num_decode:].reshape(-1, self.num_heads, D)
            pk = key[num_decode:].reshape(-1, self.num_kv_heads, D)
            pv = value[num_decode:].reshape(-1, self.num_kv_heads, D)

            if self.num_kv_groups > 1:
                pk = pk.repeat_interleave(self.num_kv_groups, dim=1)
                pv = pv.repeat_interleave(self.num_kv_groups, dim=1)

            L = pq.shape[0]
            scores = torch.bmm(
                pq.transpose(0, 1).float(),
                pk.transpose(0, 1).float().transpose(-2, -1)
            ) * self.scale
            causal_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0)
            weights = F.softmax(scores, dim=-1)
            prefill_out = torch.bmm(weights, pv.transpose(0, 1).float())
            output[num_decode:] = prefill_out.transpose(0, 1).reshape(
                num_prefill, -1).to(output.dtype)

        # --- Decode: compressed score + V decompress ---
        if num_decode > 0:
            dq = query[:num_decode].reshape(num_decode, self.num_heads, D)
            seq_lens = attn_metadata.seq_lens
            block_table = attn_metadata.block_table

            for qi in range(num_decode):
                sl = seq_lens[qi].item()
                q_rot = dq[qi].float() @ Pi.T   # (num_heads, D)
                q_proj = dq[qi].float() @ S.T

                for h in range(self.num_heads):
                    kv_h = h // self.num_kv_groups

                    m_prev = float('-inf')
                    d_prev = 0.0
                    acc = torch.zeros(D, device=device)

                    for t in range(sl):
                        bi_log = t // block_size
                        bo = t % block_size
                        bi_phys = block_table[qi, bi_log].item()

                        # Score from packed K
                        score = self._score_packed(
                            q_rot[h], q_proj[h],
                            kv_cache[bi_phys, 0, bo, kv_h],
                            centroids)

                        # Decompress V
                        v_val = self._unpack(
                            kv_cache[bi_phys, 1, bo, kv_h],
                            Pi, S, centroids)

                        # Online softmax
                        m_new = max(m_prev, score)
                        ep = math.exp(m_prev - m_new)
                        ec = math.exp(score - m_new)
                        d_new = d_prev * ep + ec
                        acc = acc * ep + ec * v_val
                        m_prev, d_prev = m_new, d_new

                    if d_prev > 0:
                        output[qi, h * D:(h + 1) * D] = (acc / d_prev).to(output.dtype)

        return output

    # --- Helpers ---

    def _get_matrices(self, layer, device):
        if not hasattr(layer, '_tq_Pi_f32'):
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()
        return layer._tq_Pi_f32, layer._tq_S_f32, layer._tq_c_f32

    def _pack(self, vec, Pi, S, centroids, D):
        x = vec.float(); vn = x.norm(); xh = x / (vn + 1e-8)
        rot = xh @ Pi.T
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1).to(torch.uint8)
        xm = centroids[idx.long()] @ Pi; r = xh - xm; rn = r.norm()
        signs = (r @ S.T >= 0).to(torch.uint8)

        packed = torch.zeros(self._packed_size, dtype=torch.uint8, device=vec.device)
        if self._mse_bits == 2:
            for j in range(0, D, 4):
                v = 0
                for k in range(min(4, D - j)):
                    v |= (idx[j+k].item() & 0x3) << (k*2)
                packed[j//4] = v
        for j in range(0, D, 8):
            v = 0
            for k in range(min(8, D - j)):
                v |= (signs[j+k].item() & 1) << k
            packed[self._mse_bytes + j//8] = v
        no = self._mse_bytes + self._qjl_bytes
        packed[no:no+2] = vn.half().reshape(1).view(torch.uint8)
        packed[no+2:no+4] = rn.half().reshape(1).view(torch.uint8)
        return packed

    def _score_packed(self, q_rot, q_proj, packed, centroids):
        D = self.head_size
        t1 = 0.0
        if self._mse_bits == 2:
            for b in range(self._mse_bytes):
                bv = packed[b].item()
                for k in range(4):
                    j = b*4+k
                    if j >= D: break
                    t1 += q_rot[j].item() * centroids[(bv >> (k*2)) & self._mask].item()
        t2 = 0.0
        for b in range(self._qjl_bytes):
            bv = packed[self._mse_bytes + b].item()
            for k in range(8):
                j = b*8+k
                if j >= D: break
                t2 += q_proj[j].item() * (1.0 if ((bv >> k) & 1) else -1.0)
        no = self._mse_bytes + self._qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]
        return vn * (t1 + self._correction * rn * t2) * self.scale

    def _unpack(self, packed, Pi, S, centroids):
        D = self.head_size
        idx = torch.zeros(D, dtype=torch.long, device=packed.device)
        if self._mse_bits == 2:
            for j in range(D):
                b, k = j//4, j%4
                idx[j] = (packed[b].item() >> (k*2)) & self._mask
        signs = torch.zeros(D, dtype=torch.float32, device=packed.device)
        for j in range(D):
            b, k = j//8, j%8
            signs[j] = 1.0 if ((packed[self._mse_bytes+b].item() >> k) & 1) else -1.0
        no = self._mse_bytes + self._qjl_bytes
        vn = struct.unpack('e', bytes([packed[no].item(), packed[no+1].item()]))[0]
        rn = struct.unpack('e', bytes([packed[no+2].item(), packed[no+3].item()]))[0]
        c_idx = centroids[idx]
        xm = c_idx @ Pi
        xq = self._correction * rn * (signs @ S)
        return vn * (xm + xq)
