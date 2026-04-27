# SPDX-License-Identifier: Apache-2.0
"""XFP online MoE method — BF16 → per-expert learned codebook at load time.

v3: fused CUDA MoE kernel. Single kernel launch per GEMM handles all experts
via sorted_token_ids / expert_ids (Marlin pattern). No Python expert loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


# ─── Custom op for torch.compile / CUDA Graph compatibility ─────────
#
# The MoE forward contains moe_align_block_size (C++ op), dynamic
# tensor allocations, and Python control flow — all incompatible with
# CUDA Graph capture. Wrapping as a custom op makes torch.compile see
# an opaque operator with known output shape.

def _xfp_moe_forward_impl(
    x: torch.Tensor,              # [B, K]
    topk_weights: torch.Tensor,   # [B, topk]
    topk_ids: torch.Tensor,       # [B, topk]
    w13_packed: torch.Tensor,     # [E * fpe13] int32
    w13_codebook: torch.Tensor,   # [E * N13 * lut] fp16
    w2_packed: torch.Tensor,      # [E * fpe2] int32
    w2_codebook: torch.Tensor,    # [E * N2 * lut] fp16
    bits: int,
    K13: int, N13: int,
    K2: int, N2: int,
    E: int, fpe13: int, fpe2: int,
) -> torch.Tensor:
    """Real impl: full MoE forward (gate_up → SiLU → down → reduce)."""
    from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm
    moe_kernel = _load_xfp_moe_gemm()
    if moe_kernel is None:
        raise RuntimeError("XFP MoE kernel not loaded")

    B = x.shape[0]
    topk = topk_ids.shape[1]
    half_n = N13 // 2
    BT = B * topk

    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    no_w = torch.empty(0, dtype=torch.float32, device=x.device)

    # Token sorting — pure torch ops (CUDA Graph safe, no C++ custom op)
    flat_topk = topk_ids.reshape(-1)  # [B*topk]
    sort_indices = flat_topk.argsort(stable=True)
    sorted_token_ids = sort_indices.to(torch.int32)
    sorted_expert_ids = flat_topk[sort_indices].to(torch.int32)
    num_valid = sorted_token_ids.shape[0]

    # H1 diagnostic: surface global-vs-local expert mismatch before kernel launch
    import os as _dbg_os
    if _dbg_os.environ.get("XFP_MOE_DEBUG", "0") == "1":
        import torch.distributed as _dist
        _r = _dist.get_rank() if _dist.is_initialized() else 0
        _emax = int(topk_ids.max().item()) if topk_ids.numel() > 0 else -1
        print(
            f"[xfp_moe rank={_r}] B={B} topk={topk} BT={BT} "
            f"K13={K13} N13={N13} K2={K2} N2={N2} "
            f"E_global_max={_emax} E_local_w13={w13_packed.shape[0]} "
            f"E_local_w2={w2_packed.shape[0]} num_valid={num_valid}",
            flush=True,
        )

    # Gate+Up
    gate_up = torch.zeros(BT, N13, dtype=torch.bfloat16, device=x.device)
    moe_kernel.xfp_moe_gemm(
        x_bf16, w13_packed, w13_codebook,
        gate_up, sorted_token_ids, sorted_expert_ids,
        no_w, int(bits), int(K13), int(N13), int(topk),
        int(fpe13), num_valid)

    # SiLU — fused silu_and_mul: 1 kernel (was 3: silu, slice-mul, slice)
    activated = torch.empty(BT, half_n, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(activated, gate_up)

    # Down — pass topk_weights into kernel so epilogue multiplies by weight
    #   (was: down in bf16 → .float() → *weight → .to(bf16) → scatter_add_;
    #    now: weighted values written directly; scatter_add on bf16 no conversion.)
    down = torch.zeros(BT, N2, dtype=torch.bfloat16, device=x.device)
    down_expert_ids = topk_ids.reshape(-1).to(torch.int32)
    down_sorted = torch.arange(BT, dtype=torch.int32, device=x.device)
    tw_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()
    moe_kernel.xfp_moe_gemm(
        activated, w2_packed, w2_codebook,
        down, down_sorted, down_expert_ids,
        tw_flat, int(bits), int(K2), int(N2), 1,
        int(fpe2), BT)

    # Scatter-reduce (weights already applied in kernel epilogue)
    orig = torch.arange(BT, device=x.device, dtype=torch.int64) // topk
    output = torch.zeros(B, N2, dtype=torch.bfloat16, device=x.device)
    output.scatter_add_(0, orig.unsqueeze(1).expand_as(down), down)
    return output


def _xfp_moe_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_codebook: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_codebook: torch.Tensor,
    bits: int,
    K13: int, N13: int,
    K2: int, N2: int,
    E: int, fpe13: int, fpe2: int,
) -> torch.Tensor:
    """Fake impl: output shape [B, N2] bf16."""
    return torch.empty(x.shape[0], N2, dtype=torch.bfloat16, device=x.device)


try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="xfp_moe_forward",
        op_func=_xfp_moe_forward_impl,
        fake_impl=_xfp_moe_forward_fake,
    )
    _xfp_moe_op = torch.ops.vllm.xfp_moe_forward
    logger.info("XFP MoE custom op registered (torch.compile safe)")
except Exception as e:
    logger.warning("XFP MoE custom op registration failed: %s", e)
    _xfp_moe_op = _xfp_moe_forward_impl


class XFPMoEMethod(FusedMoEMethodBase):
    """Learned-codebook quant-on-load for FusedMoE layers.

    Per-expert Lloyd codebook + word-aligned packed indices.
    Apply uses fused CUDA kernel — one launch per GEMM, all experts.
    """

    def __init__(
        self,
        quant_config: "QuantizationConfig",
        dtype: str = "xfp4",
        moe_config: "FusedMoEConfig | None" = None,
    ):
        if dtype not in ("xfp", "xfp2", "xfp3", "xfp4"):
            raise ValueError(
                f"XFPMoEMethod: unsupported dtype '{dtype}', "
                f"supported: xfp (auto), xfp2, xfp3, xfp4"
            )
        if moe_config is not None:
            super().__init__(moe_config)
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS[dtype]

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        self._unquant = UnquantizedFusedMoEMethod(self.moe)
        # When the base_loader has flagged streaming quant-on-load, allocate
        # the huge expert tensors on meta. initialize_streaming_quantload()
        # will materialize them on CUDA right before the loader touches them.
        from vllm.model_executor.model_loader.utils import _moe_meta_active
        if _moe_meta_active():
            with torch.device("meta"):
                self._unquant.create_weights(
                    layer, num_experts, hidden_size,
                    intermediate_size_per_partition, params_dtype,
                    **extra_weight_attrs,
                )
        else:
            self._unquant.create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs,
            )
        layer._xfp_moe_hidden = hidden_size
        layer._xfp_moe_intermediate = intermediate_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_xfp_moe_packed", False):
            return

        from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
        from vllm.multiquant.xfp.xfp_moe_kernel import _load_xfp_moe_gemm
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        device = layer.w13_weight.device

        # ─── Cache check (skip 256-expert Lloyd if we have it on disk) ──
        cache = MultiQuantWeightCache.get_active()
        layer_prefix = getattr(layer, "layer_name", "") or \
            getattr(layer, "prefix", "") or ""
        if cache is not None and layer_prefix and xfp_cache.load_moe(
                cache, layer_prefix, layer, device):
            # Kernel JIT-load still needed so forward path is graph-safe.
            _load_xfp_gemm(int(layer._xfp_moe_bits))
            _load_xfp_moe_gemm()
            logger.info(
                "XFP MoE %s ← cache (skip Lloyd) bits=%d E=%d "
                "K13=%d N13=%d K2=%d N2=%d",
                layer_prefix, layer._xfp_moe_bits, layer._xfp_moe_E,
                layer._xfp_moe_K13, layer._xfp_moe_N13,
                layer._xfp_moe_K2, layer._xfp_moe_N2,
            )
            # Shrink the BF16 storage that streaming_loader materialized. A
            # plain `del layer.wXX_weight` removes the attribute but the
            # nn.Parameter keeps living in layer._parameters, so its CUDA
            # storage (~5 GB per MoE layer on Qwen 122B) stays allocated
            # until much later — over 48 MoE layers this accumulates ~240 GB
            # of ghost BF16 in UMA and OOMs before profile_run. Reassigning
            # .data to a zero-size tensor frees the original storage now.
            for attr in ("w13_weight", "w2_weight"):
                p = layer._parameters.get(attr)
                if p is not None:
                    p.data = torch.empty(0, device=p.data.device,
                                         dtype=p.data.dtype)
            try:
                del layer.w13_weight, layer.w2_weight
            except AttributeError:
                pass
            return

        bits = self.bits
        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]

        # Diagnostic: under TP>1 Ray, w2 has been observed with <3 dims
        # (IndexError at shape[2] below). Log the offenders with their
        # ranks + shapes so we can figure out the sharding pattern.
        if w13.ndim != 3 or w2.ndim != 3:
            try:
                import torch.distributed as _dist
                _rank = _dist.get_rank() if _dist.is_initialized() else -1
            except Exception:
                _rank = -1
            # PR1: include numel, dtype + cache-presence to disambiguate
            # "stub never overwritten" vs "vllm wrote partial" vs
            # "cache-hit branch should have fired but didn't".
            try:
                from vllm.multiquant.weight_cache import (
                    MultiQuantWeightCache,
                )
                _cache = MultiQuantWeightCache.get_active()
                _layer_prefix = getattr(layer, "layer_name", "") or \
                    getattr(layer, "prefix", "") or ""
                _cache_dir = (_cache.cache_dir / _layer_prefix
                              if _cache and _layer_prefix else None)
                _cache_present = _cache_dir.exists() if _cache_dir else False
            except Exception:
                _cache_present = "?"
            logger.warning(
                "XFP MoE: unexpected weight rank at %s "
                "(dist_rank=%d): w13.shape=%s numel=%d dtype=%s device=%s | "
                "w2.shape=%s numel=%d dtype=%s device=%s | "
                "cache_dir_for_layer_present=%s",
                getattr(layer, "layer_name", "?") or
                getattr(layer, "prefix", "?"),
                _rank, tuple(w13.shape), int(w13.numel()), w13.dtype,
                w13.device,
                tuple(w2.shape), int(w2.numel()), w2.dtype, w2.device,
                _cache_present,
            )
        E = int(w13.shape[0])

        # MoE Lloyd iters: defined BEFORE auto-select so both use the same.
        import os
        moe_lloyd_iters = int(os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))

        # Auto bit-width: sample a few experts, run auto_select with the
        # SAME lloyd_iters as the actual packing to avoid quality mismatch.
        if bits == 0:
            from vllm.multiquant.xfp.xfp_pack import xfp_auto_select
            # XFP_MOE_SAMPLE_EXPERTS=0 → alle Experten; default 4.
            se_env = int(os.environ.get("XFP_MOE_SAMPLE_EXPERTS", "4"))
            sample_experts = E if se_env == 0 else min(se_env, E)
            sample = w13[:sample_experts].reshape(-1, w13.shape[2]).float()
            bits = xfp_auto_select(
                sample,
                candidates=(2, 3, 4),
                min_cos=self.quant_config.auto_min_cos
                    if hasattr(self.quant_config, 'auto_min_cos') else 0.98,
                lloyd_iters=moe_lloyd_iters,
            )
            logger.info("XFP MoE auto-select: bits=%d (from %d/%d expert sample, "
                        "lloyd=%d)", bits, sample_experts, E, moe_lloyd_iters)
            # Sample kann bei sample_experts=E mehrere GB belegen → explizit
            # freigeben bevor die teure Packing-Stufe den HBM braucht.
            del sample
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        _load_xfp_gemm(bits)
        _load_xfp_moe_gemm()

        # Save shape metadata before freeing weights
        K13 = int(w13.shape[2])
        N13 = int(w13.shape[1])
        K2 = int(w2.shape[2])
        N2 = int(w2.shape[1])

        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        reg = MultiQuantPolicyRegistry.get_active()

        def _expertwise_pack(W_stack: torch.Tensor):
            """W_stack: [E, N, K] -> packed [E, K_packed, N], codebook [E, N, lut].

            Schema v2: keep tensors **pre-repack** (3D for MoE, no
            warp-interleave) so the cache can be TP-sliced with simple
            ``narrow()`` calls. The forward kernel's warp-interleaved
            repack happens on-device in ``load_moe`` after slicing.

            Packs one expert at a time: float32 transient = N×K×4 bytes
            (~25 MiB) instead of E×N×K×4 (~9 GiB). Critical for 122B+
            models on unified memory.
            """
            E_ = int(W_stack.shape[0])
            packed_list = []
            codebook_list = []
            last_stats = None

            for e in range(E_):
                W_e = W_stack[e].float()          # [N, K] float32
                packed_e, cb_e, _, _, stats_e = xfp_pack(
                    W_e, bits=bits, outlier_sigma=None,
                    lloyd_iters=moe_lloyd_iters,
                )
                del W_e
                # packed_e: [K_packed, N] int32 — keep as-is (no repack).
                packed_list.append(packed_e)
                codebook_list.append(cb_e)        # [N, lut_size]
                last_stats = stats_e

            del W_stack
            # Stack into [E, K_packed, N] / [E, N, lut].
            all_packed = torch.stack(packed_list, dim=0)
            del packed_list
            all_codebook = torch.stack(codebook_list, dim=0)
            del codebook_list
            return all_packed, all_codebook, last_stats

        # Pack w13, then free BF16 w13 before packing w2
        p13, cb13, stats13 = _expertwise_pack(w13)
        layer.w13_weight.data = torch.empty(0)  # free BF16

        p2, cb2, stats2 = _expertwise_pack(w2)
        layer.w2_weight.data = torch.empty(0)  # free BF16

        if reg is not None:
            reg.record_stats("routed_expert", stats13)
            reg.record_stats("routed_expert", stats2)

        # Attach stats to the layer so save_moe can embed them in the
        # per-layer _manifest.json metadata for offline analysis.
        layer._xfp_moe_stats13 = stats13
        layer._xfp_moe_stats2 = stats2

        # Persist to disk cache *before* any decision about whether to
        # keep packed tensors on the Layer — quant-only runs exit after
        # this, so the cache write must have happened first.
        _quant_only = os.environ.get("MULTIQUANT_QUANT_ONLY", "").lower() in (
            "1", "true", "yes", "on",
        )

        # (cache persistence happens below in the existing `if cache is
        # not None` block — we preserve that order; MULTIQUANT_QUANT_ONLY
        # only changes what we do AFTER cache.save)

        # Attach packed tensors unconditionally — save_moe reads them
        # from layer.w13_xfp_packed/.w2_xfp_packed/etc. In quant-only
        # mode we strip them AFTER save_moe (see below).
        # Schema v2: store pre-repack 3D tensors. The on-device
        # warp-interleaved repack happens in load_moe (post TP-slice).
        layer.w13_xfp_packed = nn.Parameter(p13.to(device), requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(cb13.to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(p2.to(device), requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(cb2.to(device), requires_grad=False)

        layer._xfp_moe_bits = bits
        layer._xfp_moe_dtype = self.dtype
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13
        layer._xfp_moe_K2 = K2
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = E
        # fpe = K_groups * N * WS, computed for the kernel's repacked shape.
        WS = 32
        K13_packed = (K13 * bits) // 32
        K_g13 = (K13_packed + WS - 1) // WS
        K2_packed = (K2 * bits) // 32
        K_g2 = (K2_packed + WS - 1) // WS
        layer._xfp_moe_fpe13 = K_g13 * N13 * WS
        layer._xfp_moe_fpe2 = K_g2 * N2 * WS
        layer._xfp_moe_packed = True

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

        # Persist to disk cache (if enabled) so future loads skip Lloyd.
        if cache is not None and layer_prefix:
            xfp_cache.save_moe(cache, layer_prefix, layer)

        # If we're going to keep this layer in VRAM (not quant-only) and
        # we hit the forward kernel during this run, we must on-device
        # repack the 3D pre-repack tensors into the warp-interleaved
        # flat form the kernel reads. quant-only exits before forward,
        # so we skip the repack to save time.
        if not _quant_only:
            from vllm.multiquant.xfp._repack import repack_3d_moe_to_flat
            w13_flat = repack_3d_moe_to_flat(layer.w13_xfp_packed.data)
            w2_flat = repack_3d_moe_to_flat(layer.w2_xfp_packed.data)
            layer.w13_xfp_packed.data = w13_flat
            layer.w2_xfp_packed.data = w2_flat

        # MULTIQUANT_QUANT_ONLY: now that cache.save has persisted the
        # packed artefacts, strip the attached Parameters' storage so
        # this layer's VRAM footprint returns to ~zero. Critical for
        # 397B XFP on 128 GB UMA (quantized ≈ 200 GB, won't fit
        # steady-state; only during pack-then-free per layer).
        if _quant_only:
            for _attr in ("w13_xfp_packed", "w13_xfp_codebook",
                          "w2_xfp_packed", "w2_xfp_codebook"):
                p = getattr(layer, _attr, None)
                if p is None:
                    continue
                p.data = torch.empty(0, dtype=p.data.dtype, device=p.data.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "XFP MoE %s quant-only: cache saved, VRAM stripped",
                layer_prefix or "?",
            )

        logger.info(
            "XFP MoE: %d experts w13[%dx%d] + w2[%dx%d] -> %s "
            "(fused, fpe=%d/%d, lloyd=%d)",
            E, layer._xfp_moe_N13, layer._xfp_moe_K13,
            layer._xfp_moe_N2, layer._xfp_moe_K2, self.dtype,
            layer._xfp_moe_fpe13, layer._xfp_moe_fpe2, moe_lloyd_iters,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not getattr(layer, "_xfp_moe_packed", False):
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                x, layer.w13_weight, layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids,
            )

        return _xfp_moe_op(
            x, topk_weights, topk_ids,
            layer.w13_xfp_packed, layer.w13_xfp_codebook,
            layer.w2_xfp_packed, layer.w2_xfp_codebook,
            int(layer._xfp_moe_bits),
            int(layer._xfp_moe_K13), int(layer._xfp_moe_N13),
            int(layer._xfp_moe_K2), int(layer._xfp_moe_N2),
            int(layer._xfp_moe_E),
            int(layer._xfp_moe_fpe13), int(layer._xfp_moe_fpe2),
        )
