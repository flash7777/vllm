# SPDX-License-Identifier: Apache-2.0
"""MultiQuant attention backend — compressed uint8 KV-cache.

Generic backend for any KV-cache quantizer registered in the MultiQuant
registry (TurboQuant, RotorQuant, etc.). Does NOT inherit FlashInfer.

Cache: (num_blocks, 2, block_size, num_kv_heads, packed_size) uint8
Decode: compressed score + online softmax + V decompress
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

class MultiQuantAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3", "tq4", "rq2", "rq3", "rq4",
    ]

    @staticmethod
    def get_name() -> str:
        return "MULTIQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype=None) -> bool:
        if not kv_cache_dtype:
            return False
        from vllm.multiquant.registry import is_multiquant_dtype
        return is_multiquant_dtype(kv_cache_dtype)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True  # packed_size is passed as head_size

    @staticmethod
    def get_impl_cls():
        return MultiQuantImpl

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
    from vllm.v1.attention.backend import AttentionCGSupport
    # TQ: CUDA kernel is graph-safe → full decode graphs
    # RQ: Clifford rotation has Python control flow → PIECEWISE only
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        from vllm.v1.attention.backend import AttentionCGSupport
        kv_dtype = str(getattr(vllm_config.cache_config, 'cache_dtype', ''))
        if kv_dtype.startswith("rq"):
            # RQ with fused Clifford kernel → graph-safe
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                _load_clifford_kernel,
            )
            if _load_clifford_kernel() is not None:
                return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
            return AttentionCGSupport.NEVER  # Python fallback → PIECEWISE
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE  # TQ

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

class MultiQuantImpl:
    """Custom TQ attention — no FlashInfer dependency."""

    supports_quant_query_input: bool = False
    can_return_lse_for_decode: bool = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    @staticmethod
    def _recover_head_dim(head_size: int, kv_cache_dtype: str) -> int:
        """Recover real head_dim from head_size parameter.

        vLLM passes EITHER the real head_dim (256) OR the packed_size (100)
        depending on the code path. Check if head_size is already a valid D
        (i.e., its packed_size != head_size), else reverse-map from packed.
        """
        from vllm.multiquant.registry import get_kv_quantizer_config
        # Check if head_size IS already the real D
        try:
            cfg = get_kv_quantizer_config(kv_cache_dtype, head_size)
            if cfg.key_packed_size != head_size:
                # head_size is real D (packed would be different)
                return head_size
        except Exception:
            pass
        # head_size is packed_size — reverse map to real D
        for d in [64, 96, 128, 192, 256, 512]:
            try:
                cfg = get_kv_quantizer_config(kv_cache_dtype, d)
                if cfg.key_packed_size == head_size:
                    return d
            except Exception:
                continue
        # Fallback
        return head_size

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="tq3",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        self.num_heads = num_heads
        # head_size from Spec is packed_size (uint8 bytes), not real head_dim.
        # Recover real D: packed = ceil(D*mse_bits/8) + ceil(D/8) + 4
        from vllm.multiquant.registry import get_kv_quantizer_config
        real_head_dim = self._recover_head_dim(head_size, kv_cache_dtype)
        self.head_size = real_head_dim
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self._tq_config = get_kv_quantizer_config(kv_cache_dtype, real_head_dim)
        self._packed_size = self._tq_config.key_packed_size
        self._mse_bits = self._tq_config.mse_bits
        self._mse_bytes = (real_head_dim * self._mse_bits + 7) // 8
        self._qjl_bytes = (real_head_dim + 7) // 8

        self._mask = (1 << self._mse_bits) - 1
        self._correction = math.sqrt(math.pi / 2) / real_head_dim
        self._is_rq = kv_cache_dtype.startswith("rq")

        # Pre-load decode kernel at init (not in forward — graph-safe)
        from vllm.v1.attention.ops.triton_mq_fused_decode import (
            mq_fused_decode_attention, _load_cuda_kernel,
        )
        cuda_kernel = _load_cuda_kernel()
        self._decode_fn = mq_fused_decode_attention

        # Pre-load RQ kernels at init (JIT compile before first forward)
        if self._is_rq:
            import vllm.multiquant.rotorquant.clifford  # noqa: F401
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                _load_rq_decode_kernel, _load_clifford_kernel,
            )
            _load_rq_decode_kernel()
            _load_clifford_kernel()

        logger.info(
            "MultiQuant attention: D=%d (from spec %d), %s KV, "
            "decode=%s, %s",
            self.head_size, head_size,
            self.kv_cache_dtype,
            "CUDA" if cuda_kernel else "Triton",
            "RQ Clifford" if self._is_rq else "TQ rotation",
        )

    def _rotate_forward(self, x, Pi):
        """Forward rotation: TQ = x @ Pi.T, RQ = rotor sandwich."""
        if self._is_rq:
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors, rotor_sandwich,
                extract_vectors_from_multivectors,
            )
            D = x.shape[-1]
            mv = embed_vectors_as_multivectors(x)
            mv_rot = rotor_sandwich(Pi, mv)
            return extract_vectors_from_multivectors(mv_rot, D)
        return x @ Pi.T

    def _rotate_inverse(self, x, Pi):
        """Inverse rotation: TQ = x @ Pi, RQ = reverse rotor sandwich."""
        if self._is_rq:
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors, rotor_sandwich,
                extract_vectors_from_multivectors, reverse,
            )
            D = x.shape[-1]
            mv = embed_vectors_as_multivectors(x)
            rotor_rev = reverse(Pi)
            mv_recon = rotor_sandwich(rotor_rev, mv)
            return extract_vectors_from_multivectors(mv_recon, D)
        return x @ Pi

    @torch.compiler.disable
    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Pack K+V into compressed uint8 cache.

        During CUDA Graph capture, this is a no-op — the actual packing
        happens at replay time with real data. The capture just sees
        the tensor shapes.
        """
        if self.kv_sharing_target_layer_name is not None:
            return

        # Skip during CUDA Graph capture — pack_vectors_batched has Python
        # loops that are not capture-safe. The real update happens on replay.
        if torch.cuda.is_current_stream_capturing():
            import os
            if os.environ.get("MQ_DEBUG"):
                logger.warning("[MQ_KV] SKIPPED — stream capturing!")
            return

        import os
        if os.environ.get("MQ_DEBUG"):
            logger.info("[MQ_KV] WRITE slots=%s key=%s val=%s",
                        slot_mapping.tolist()[:4], key.shape, value.shape)

        D = self.head_size
        device = key.device
        block_size = kv_cache.shape[2]

        Pi, S, centroids = self._get_matrices(layer, device)

        num_tokens, num_heads = key.shape[0], key.shape[1]

        # Vectorized pack: all tokens × all heads in one batch
        # key/value: (num_tokens, num_heads, D) → flat (N, D)
        k_flat = key.reshape(-1, D)  # (num_tokens * num_heads, D)
        v_flat = value.reshape(-1, D)
        k_packed = self._pack_batch(k_flat, Pi, S, centroids, D)  # (N, packed)
        v_packed = self._pack_batch(v_flat, Pi, S, centroids, D)
        k_packed = k_packed.reshape(num_tokens, num_heads, -1)
        v_packed = v_packed.reshape(num_tokens, num_heads, -1)

        # Write to cache via slot_mapping
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        bi = slots // block_size
        bo = slots % block_size
        kv_cache[bi, 0, bo, :, :self._packed_size] = k_packed[valid]
        kv_cache[bi, 1, bo, :, :self._packed_size] = v_packed[valid]

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        """Dispatch to decode (graph-safe) or prefill (compiler-disabled)."""
        D = self.head_size
        N = query.shape[0]

        # vLLM passes 3D tensors [N, heads, D] — flatten to 2D
        if query.dim() == 3:
            query = query.reshape(N, -1)
        if key is not None and key.dim() == 3:
            key = key.reshape(key.shape[0], -1)
        if value is not None and value.dim() == 3:
            value = value.reshape(value.shape[0], -1)

        output_3d = False
        if output is not None and output.dim() == 3:
            output_3d = True
            out_shape = output.shape
            output = output.view(N, -1)
        elif output is None:
            output = torch.empty(N, self.num_heads * D,
                                 device=query.device, dtype=query.dtype)

        if attn_metadata is None:
            if output_3d:
                return output.fill_(0).view(out_shape)
            return output.fill_(0)

        device = query.device
        Pi, S, centroids = self._get_matrices(layer, device)
        block_size = kv_cache.shape[2]

        num_prefill = attn_metadata.num_prefill_tokens
        num_decode = attn_metadata.num_decode_tokens

        import os
        if os.environ.get("MQ_DEBUG"):
            sl = attn_metadata.seq_lens[:max(1,num_decode)].tolist() if num_decode > 0 else []
            cache_nz = kv_cache.any(dim=-1).sum().item()
            logger.info("[MQ_FWD] pf=%d dc=%d sl=%s cache_nz=%d",
                        num_prefill, num_decode, sl[:4], cache_nz)

        if num_prefill > 0:
            self._forward_prefill(
                query, key, value, output, Pi, S, centroids,
                num_decode, num_prefill, D, device,
            )

        if num_decode > 0:
            self._forward_decode(
                query, output, kv_cache, Pi, S, centroids,
                attn_metadata, num_decode, block_size, D, device,
            )

        if output_3d:
            return output.view(out_shape)
        return output

    @torch.compiler.disable
    def _forward_prefill(self, query, key, value, output, Pi, S, centroids,
                         num_decode, num_prefill, D, device):
        """Prefill: naive causal bmm. Not graph-captured."""
        L = num_prefill
        pq = query[num_decode:num_decode + L].reshape(L, self.num_heads, D)
        pk = key[num_decode:num_decode + L].reshape(L, self.num_kv_heads, D)
        pv = value[num_decode:num_decode + L].reshape(L, self.num_kv_heads, D)

        if self.num_kv_groups > 1:
            pk = pk.repeat_interleave(self.num_kv_groups, dim=1)
            pv = pv.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.bmm(
            pq.transpose(0, 1).float(),
            pk.transpose(0, 1).float().transpose(-2, -1)
        ) * self.scale
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        prefill_out = torch.bmm(weights, pv.transpose(0, 1).float())
        output[num_decode:num_decode + L] = prefill_out.transpose(0, 1).reshape(
            L, -1).to(output.dtype)

    def _forward_decode(self, query, output, kv_cache, Pi, S, centroids,
                        attn_metadata, num_decode, block_size, D, device):
        """Decode: TQ → CUDA fused kernel, RQ → Python loop."""
        dq = query[:num_decode].reshape(num_decode, self.num_heads, D)
        if self._is_rq:
            # RQ: separate CUDA kernel with Clifford V decompression
            from vllm.v1.attention.ops.triton_mq_fused_decode import (
                _load_rq_decode_kernel, _rq_rotate_forward,
            )
            rq_kernel = _load_rq_decode_kernel()
            if rq_kernel is not None:
                q_flat = dq.reshape(num_decode * self.num_heads, D).float()
                q_rot = _rq_rotate_forward(q_flat, Pi)
                q_proj = torch.mm(q_flat, S.T)
                q_rot_3d = q_rot.reshape(num_decode, self.num_heads, D)
                q_proj_3d = q_proj.reshape(num_decode, self.num_heads, D)
                rq_out = torch.empty(
                    num_decode, self.num_heads, D,
                    device=device, dtype=torch.float32)
                s_block = kv_cache.stride(0)
                s_kv = kv_cache.stride(1)
                s_slot = kv_cache.stride(2)
                s_head = kv_cache.stride(3)
                rq_kernel.rq_fused_decode_attention(
                    q_rot_3d, q_proj_3d, kv_cache,
                    Pi.float().contiguous(),
                    S.float().contiguous(),
                    centroids,
                    attn_metadata.block_table[:num_decode].int(),
                    attn_metadata.seq_lens[:num_decode].int(),
                    rq_out,
                    D, self._mse_bits, centroids.shape[0], self.scale,
                    s_block, s_kv, s_slot, s_head,
                )
                output[:num_decode] = rq_out.reshape(
                    num_decode, -1).to(output.dtype)
            else:
                # Fallback: Python loop
                self._decode_python_loop(
                    dq, kv_cache, Pi, S, centroids,
                    attn_metadata.seq_lens[:num_decode],
                    attn_metadata.block_table[:num_decode],
                    block_size, D, device, output,
                )
        else:
            # TQ: CUDA fused kernel with full V decompression via Pi/S GEMV
            decode_out = self._decode_fn(
                q=dq, kv_cache=kv_cache, Pi=Pi, S=S,
                centroids=centroids,
                block_table=attn_metadata.block_table[:num_decode],
                seq_lens=attn_metadata.seq_lens[:num_decode],
                scale=self.scale, block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                mse_bits=self._mse_bits,
                correction=self._correction,
                is_rq=False,
            )
            output[:num_decode] = decode_out.reshape(num_decode, -1).to(output.dtype)

    # --- Helpers ---

    @torch.compiler.disable
    def _decode_python_loop(self, dq, kv_cache, Pi, S, centroids,
                            seq_lens, block_table, block_size, D,
                            device, output):
        """RQ fallback decode — Python loop over batch × heads × tokens."""
        num_decode = dq.shape[0]
        for qi in range(num_decode):
            sl = seq_lens[qi].item()
            if sl <= 0:
                continue
            positions = torch.arange(sl, device=device)
            bi_log = positions // block_size
            bo = positions % block_size
            bi_phys = block_table[qi, bi_log.long()]

            for kv_h in range(self.num_kv_heads):
                k_packed = kv_cache[bi_phys, 0, bo, kv_h]
                v_packed = kv_cache[bi_phys, 1, bo, kv_h]

                # CUDA unpack with .contiguous()
                try:
                    from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
                    k_result = cuda_unpack(k_packed.contiguous(), D, self._mse_bits)
                except Exception as e:
                    logger.debug("K cuda_unpack fallback: %s", e)
                    k_result = None

                if k_result is not None:
                    idx_all, signs_all, k_vn, k_rn = k_result
                    idx_all = idx_all.long()
                else:
                    no = self._mse_bytes + self._qjl_bytes
                    idx_all = torch.zeros(sl, D, dtype=torch.long, device=device)
                    for j in range(D):
                        boff = j * self._mse_bits
                        bi = boff // 8
                        bs = boff % 8
                        bv = k_packed[:, bi].long() >> bs
                        spill = bs + self._mse_bits - 8
                        if spill > 0 and bi + 1 < self._mse_bytes:
                            bv = bv | (k_packed[:, bi + 1].long() << (self._mse_bits - spill))
                        idx_all[:, j] = bv & self._mask
                    signs_all = torch.zeros(sl, D, dtype=torch.float32, device=device)
                    for b in range(self._qjl_bytes):
                        bv = k_packed[:, self._mse_bytes + b].long()
                        for k in range(8):
                            j = b * 8 + k
                            if j >= D:
                                break
                            signs_all[:, j] = torch.where(
                                ((bv >> k) & 1).bool(),
                                torch.ones(sl, device=device),
                                -torch.ones(sl, device=device))
                    vn_bytes = k_packed[:, no:no + 2].contiguous()
                    rn_bytes = k_packed[:, no + 2:no + 4].contiguous()
                    k_vn = vn_bytes.view(torch.float16).float().squeeze(-1)
                    k_rn = rn_bytes.view(torch.float16).float().squeeze(-1)

                no = self._mse_bytes + self._qjl_bytes
                c_idx = centroids[idx_all]

                for h in range(kv_h * self.num_kv_groups,
                               (kv_h + 1) * self.num_kv_groups):
                    q_rot_h = self._rotate_forward(
                        dq[qi, h].float().unsqueeze(0), Pi
                    ).squeeze(0)
                    q_proj_h = dq[qi, h].float() @ S.T
                    term1 = (q_rot_h.unsqueeze(0) * c_idx).sum(-1)
                    term2 = (q_proj_h.unsqueeze(0) * signs_all).sum(-1)
                    scores = k_vn * (
                        term1 + self._correction * k_rn * term2
                    ) * self.scale

                    # V unpack
                    try:
                        v_result = cuda_unpack(v_packed.contiguous(), D, self._mse_bits)
                    except Exception as e:
                        logger.debug("V cuda_unpack fallback: %s", e)
                        v_result = None

                    if v_result is not None:
                        v_idx, v_signs, v_vn, v_rn = v_result
                        v_idx = v_idx.long()
                    else:
                        v_idx = torch.zeros(sl, D, dtype=torch.long, device=device)
                        for j in range(D):
                            boff = j * self._mse_bits
                            bi = boff // 8
                            bs = boff % 8
                            bv = v_packed[:, bi].long() >> bs
                            spill = bs + self._mse_bits - 8
                            if spill > 0 and bi + 1 < self._mse_bytes:
                                bv = bv | (v_packed[:, bi + 1].long() << (self._mse_bits - spill))
                            v_idx[:, j] = bv & self._mask
                        v_signs = torch.zeros(sl, D, dtype=torch.float32, device=device)
                        for b in range(self._qjl_bytes):
                            bv = v_packed[:, self._mse_bytes + b].long()
                            for bk in range(8):
                                j = b * 8 + bk
                                if j >= D:
                                    break
                                v_signs[:, j] = torch.where(
                                    ((bv >> bk) & 1).bool(),
                                    torch.ones(sl, device=device),
                                    -torch.ones(sl, device=device))
                        v_vn_bytes = v_packed[:, no:no + 2].contiguous()
                        v_rn_bytes = v_packed[:, no + 2:no + 4].contiguous()
                        v_vn = v_vn_bytes.view(torch.float16).float().squeeze(-1)
                        v_rn = v_rn_bytes.view(torch.float16).float().squeeze(-1)
                    v_c = centroids[v_idx]
                    v_xm = self._rotate_inverse(v_c, Pi)
                    v_xq = self._correction * v_rn.unsqueeze(-1) * (v_signs @ S)
                    v_recon = v_vn.unsqueeze(-1) * (v_xm + v_xq)

                    weights = F.softmax(scores, dim=-1)
                    out_h = (weights.unsqueeze(-1) * v_recon).sum(0)
                    output[qi, h * D:(h + 1) * D] = out_h.to(output.dtype)

    def _get_matrices(self, layer, device):
        if not hasattr(layer, '_tq_Pi_f32'):
            layer._tq_Pi_f32 = layer._tq_Pi.to(device).float().contiguous()
            layer._tq_S_f32 = layer._tq_S.to(device).float().contiguous()
            layer._tq_c_f32 = layer._tq_centroids.to(device).float().contiguous()
        return layer._tq_Pi_f32, layer._tq_S_f32, layer._tq_c_f32

    def _pack_single(self, vec, Pi, S, centroids, D):
        """Pack a single vector — slow, per-vector. Use _pack_batch instead."""
        return self._pack_batch(vec.unsqueeze(0), Pi, S, centroids, D).squeeze(0)

    def _pack_batch(self, vecs, Pi, S, centroids, D):
        """Pack a batch of vectors — vectorized, no .item() calls."""
        x = vecs.float()
        vn = x.norm(dim=-1)
        xh = x / (vn.unsqueeze(-1) + 1e-8)

        rot = self._rotate_forward(xh, Pi)
        idx = (rot.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        xm = self._rotate_inverse(centroids[idx], Pi)
        r = xh - xm
        rn = r.norm(dim=-1)
        signs = (r @ S.T >= 0).float()
        signs[signs == 0] = -1.0

        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        return pack_vectors_batched(idx, signs, vn, rn, D, self._mse_bits)

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
