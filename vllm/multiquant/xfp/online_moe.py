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

    # Empty-batch guard: a profile-run dummy with B=0 would launch the
    # downstream xfp_moe_gemm with grid (0,) and trigger cudaErrorInvalidValue.
    if BT == 0:
        return torch.zeros(B, N2, dtype=torch.bfloat16, device=x.device)

    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    no_w = torch.empty(0, dtype=torch.float32, device=x.device)

    # Token sorting — pure torch ops (CUDA Graph safe, no C++ custom op)
    flat_topk = topk_ids.reshape(-1)  # [B*topk]
    sort_indices = flat_topk.argsort(stable=True)
    sorted_token_ids = sort_indices.to(torch.int32)
    sorted_expert_ids = flat_topk[sort_indices].to(torch.int32)
    num_valid = sorted_token_ids.shape[0]

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


def _xfp_moe_v2_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_packed: torch.Tensor,        # [E*fpe13] int32, warp-interleaved
    w13_library: torch.Tensor,       # [L, 16] fp16
    w13_lib_id: torch.Tensor,        # [E, N13, G13] int32
    w13_scale: torch.Tensor,         # [E, N13, G13] fp16
    w13_mid: torch.Tensor,           # [E, N13, G13] fp16
    w2_packed: torch.Tensor,         # [E*fpe2] int32
    w2_library: torch.Tensor,
    w2_lib_id: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_mid: torch.Tensor,
    bits: int,
    K13: int, N13: int,
    K2: int, N2: int,
    group_size: int,
    fpe13: int, fpe2: int,
) -> torch.Tensor:
    """V2 MoE forward — direct kernel path, no dequant."""
    from vllm.multiquant.xfp.xfp_kernel import _load_xfp_v2_kernels
    _ = _load_xfp_v2_kernels()  # ensure loaded (3-tuple in HEAD)
    # MoE V2 kernel is loaded separately
    import os as _os
    from torch.utils.cpp_extension import load as _cpp_load
    global _xfp_moe_v17_module
    try:
        _xfp_moe_v17_module
    except NameError:
        kernel_dir = "/opt/mq_kernels"
        if not _os.path.exists(kernel_dir):
            kernel_dir = _os.path.normpath(
                _os.path.join(_os.path.dirname(__file__),
                              "..", "..", "..", "kernels", "multiquant"))
        _xfp_moe_v17_module = _cpp_load(
            name="xfp_moe_gemm_v17_lib",
            sources=[_os.path.join(kernel_dir, "xfp_moe_gemm_v17_lib.cu")],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-diag-suppress=177,3288",
            ],
            verbose=False,
        )
        logger.info("XFP-V2 MoE: v17_lib MoE kernel compiled")
    moe_kernel = _xfp_moe_v17_module

    B = int(x.shape[0])
    topk = int(topk_ids.shape[1])
    half_n = N13 // 2
    BT = B * topk

    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    no_w = torch.empty(0, dtype=torch.float32, device=x.device)

    # Token sorting (Marlin pattern) — torch ops, CUDA Graph safe
    flat_topk = topk_ids.reshape(-1)
    sort_indices = flat_topk.argsort(stable=True)
    sorted_token_ids = sort_indices.to(torch.int32)
    sorted_expert_ids = flat_topk[sort_indices].to(torch.int32)
    num_valid = sorted_token_ids.shape[0]

    # Gate+Up (no topk weighting yet)
    if not getattr(_xfp_moe_v2_forward, "_logged", False):
        logger.info(
            "XFP-V2 MoE forward shapes: B=%d topk=%d BT=%d K13=%d N13=%d "
            "K2=%d N2=%d num_valid=%d fpe13=%d fpe2=%d w13_packed=%s "
            "w13_lib_id=%s w13_scale=%s",
            B, topk, BT, K13, N13, K2, N2, num_valid, fpe13, fpe2,
            tuple(w13_packed.shape), tuple(w13_lib_id.shape),
            tuple(w13_scale.shape),
        )
        _xfp_moe_v2_forward._logged = True
    gate_up = torch.zeros(BT, N13, dtype=torch.bfloat16, device=x.device)
    moe_kernel.xfp_moe_gemm_v17_lib(
        x_bf16, w13_packed, w13_library, w13_lib_id, w13_scale, w13_mid,
        gate_up, sorted_token_ids, sorted_expert_ids, no_w,
        int(bits), int(K13), int(N13), int(group_size),
        int(topk), int(fpe13), int(num_valid),
    )

    # SiLU+mul
    activated = torch.empty(BT, half_n, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(activated, gate_up)

    # Down — applies topk weight in epilogue
    down = torch.zeros(BT, N2, dtype=torch.bfloat16, device=x.device)
    down_expert_ids = topk_ids.reshape(-1).to(torch.int32)
    down_sorted = torch.arange(BT, dtype=torch.int32, device=x.device)
    tw_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()
    moe_kernel.xfp_moe_gemm_v17_lib(
        activated, w2_packed, w2_library, w2_lib_id, w2_scale, w2_mid,
        down, down_sorted, down_expert_ids, tw_flat,
        int(bits), int(K2), int(N2), int(group_size),
        1, int(fpe2), int(BT),
    )

    # Scatter-reduce into final output (weights already applied)
    orig = torch.arange(BT, device=x.device, dtype=torch.int64) // topk
    output = torch.zeros(B, N2, dtype=torch.bfloat16, device=x.device)
    output.scatter_add_(0, orig.unsqueeze(1).expand_as(down), down)
    return output


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

        # ─── XFP-V2 MoE branch — env-gated, shared library across experts ──
        # Mirrors online_linear's _process_v2 path.
        import os as _v2_os
        if int(_v2_os.environ.get("XFP_V2", "0") or 0) >= 1:
            return self._process_moe_v2(layer)

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
            logger.warning(
                "XFP MoE: unexpected weight rank at %s "
                "(dist_rank=%d): w13.shape=%s device=%s | w2.shape=%s device=%s",
                getattr(layer, "layer_name", "?") or
                getattr(layer, "prefix", "?"),
                _rank, tuple(w13.shape), w13.device,
                tuple(w2.shape), w2.device,
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
            # MoE routed experts = "lazy" class: tolerant of more
            # aggressive compression than attention. XFP_MIN_COS_LAZY
            # overrides the linear-path strict cos (linear uses
            # XFP_MIN_COS_STRICT). Legacy XFP_MIN_COS works for both.
            _lazy_min_cos = float(
                os.environ.get("XFP_MIN_COS_LAZY",
                               os.environ.get("XFP_MIN_COS",
                                              str(self.quant_config.auto_min_cos
                                                  if hasattr(self.quant_config,
                                                             'auto_min_cos')
                                                  else 0.98)))
            )
            # V1 MoE kernel supports BITS=2/3/4 natively.
            bits = xfp_auto_select(
                sample,
                candidates=(2, 3, 4),
                min_cos=_lazy_min_cos,
                lloyd_iters=moe_lloyd_iters,
            )
            logger.info("XFP MoE auto-select: bits=%d (from %d/%d expert sample, "
                        "lloyd=%d, min_cos=%.4f)", bits, sample_experts, E,
                        moe_lloyd_iters, _lazy_min_cos)
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

        def _expertwise_pack_and_repack(W_stack: torch.Tensor):
            """W_stack: [E, N, K] -> flat packed [E*fpe], flat codebook [E*N*lut].

            Packs one expert at a time: float32 transient = N×K×4 bytes
            (~25 MiB) instead of E×N×K×4 (~9 GiB). Critical for 122B+
            models on unified memory.
            """
            E_ = int(W_stack.shape[0])
            N_ = int(W_stack.shape[1])
            K_ = int(W_stack.shape[2])
            lut_size = 1 << bits

            repacked_list = []
            codebook_list = []
            last_stats = None

            for e in range(E_):
                W_e = W_stack[e].float()          # [N, K] float32, ~25 MiB
                packed_e, cb_e, _, _, stats_e = xfp_pack(
                    W_e, bits=bits, outlier_sigma=None,
                    lloyd_iters=moe_lloyd_iters,
                )
                del W_e
                repacked_list.append(xfp_repack(packed_e))
                codebook_list.append(cb_e)        # [N, lut_size]
                del packed_e
                last_stats = stats_e

            del W_stack
            flat_per_expert = repacked_list[0].numel()
            all_repacked = torch.cat(repacked_list, dim=0)
            del repacked_list
            all_codebook = torch.cat(codebook_list, dim=0)
            del codebook_list

            return all_repacked, all_codebook, flat_per_expert, last_stats

        # Pack w13, then free BF16 w13 before packing w2
        p13, cb13, fpe13, stats13 = _expertwise_pack_and_repack(w13)
        layer.w13_weight.data = torch.empty(0)  # free BF16

        p2, cb2, fpe2, stats2 = _expertwise_pack_and_repack(w2)
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
        layer._xfp_moe_fpe13 = fpe13
        layer._xfp_moe_fpe2 = fpe2
        layer._xfp_moe_packed = True

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

        # Persist to disk cache (if enabled) so future loads skip Lloyd.
        if cache is not None and layer_prefix:
            xfp_cache.save_moe(cache, layer_prefix, layer)

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
            fpe13, fpe2, moe_lloyd_iters,
        )

    def _process_moe_v2(self, layer: nn.Module) -> None:
        """V2 MoE pack with shared library across experts per stack.

        Reuses xfp_moe_pack_v2 (which itself reuses _lloyd_per_channel +
        _pack_indices). Saves via xfp_cache.save_moe_v2.

        Apply path uses a Python reference: dequant per stack to BF16
        once, then call vllm's fused_experts. This is correct but
        slow — Phase 3 replaces with the v17_lib MoE kernel.
        """
        import os as _os
        from vllm.multiquant.xfp.xfp_pack import xfp_moe_pack_v2
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        cache = MultiQuantWeightCache.get_active()
        layer_prefix = (getattr(layer, "layer_name", None)
                        or getattr(layer, "prefix", None) or "")
        device = layer.w13_weight.device

        # Cache check first
        if cache is not None and layer_prefix and xfp_cache.load_moe_v2(
                cache, layer_prefix, layer, device):
            logger.info(
                "XFP-V2 MoE %s ← cache (skip Lloyd) bits=%d E=%d "
                "K13=%d N13=%d K2=%d N2=%d g=%d L=%d",
                layer_prefix, layer._xfp_moe_bits, layer._xfp_moe_E,
                layer._xfp_moe_K13, layer._xfp_moe_N13,
                layer._xfp_moe_K2, layer._xfp_moe_N2,
                layer._xfp_moe_group_size, layer._xfp_moe_library_size,
            )
            for attr in ("w13_weight", "w2_weight"):
                p = layer._parameters.get(attr)
                if p is not None:
                    p.data = torch.empty(0, device=p.data.device, dtype=p.data.dtype)
            try:
                del layer.w13_weight, layer.w2_weight
            except AttributeError:
                pass
            return

        bits = self.bits
        group_size = int(_os.environ.get("XFP_GROUP_SIZE", "128"))
        library_size = int(_os.environ.get("XFP_LIBRARY_SIZE", "32"))

        w13 = layer.w13_weight.data  # [E, N_gate_up, K]
        w2 = layer.w2_weight.data    # [E, N_down, K_down]

        K13 = int(w13.shape[2])
        N13 = int(w13.shape[1])
        K2 = int(w2.shape[2])
        N2 = int(w2.shape[1])
        E = int(w13.shape[0])

        # Auto-bits in V2 MoE: MoE = lazy class → XFP_MIN_COS_LAZY (fallback
        # XFP_MIN_COS, then quant_config.auto_min_cos). Same pattern as V1.
        if bits == 0:
            from vllm.multiquant.xfp.xfp_pack import xfp_auto_select
            se_env = int(_os.environ.get("XFP_MOE_SAMPLE_EXPERTS", "4"))
            sample_experts = E if se_env == 0 else min(se_env, E)
            sample = w13[:sample_experts].reshape(-1, w13.shape[2]).float()
            _lazy_min_cos = float(
                _os.environ.get("XFP_MIN_COS_LAZY",
                               _os.environ.get("XFP_MIN_COS",
                                              str(self.quant_config.auto_min_cos
                                                  if hasattr(self.quant_config,
                                                             'auto_min_cos')
                                                  else 0.98)))
            )
            moe_lloyd_iters = int(_os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))
            # V2 MoE kernel supports BITS=2/4. V3 kernel (XFP_V2>=3) adds
            # BITS=3 via lane-padding + SMEM-direct codebook lookup
            # ("V1+V2 symbiosis"). Per-group structure preserved.
            _xfp_v2_level = int(_os.environ.get("XFP_V2", "0") or 0)
            _v2_candidates_moe = (2, 3, 4) if _xfp_v2_level >= 3 else (2, 4)
            bits = xfp_auto_select(
                sample,
                candidates=_v2_candidates_moe,
                min_cos=_lazy_min_cos,
                lloyd_iters=moe_lloyd_iters,
            )
            logger.info("XFP-V2 MoE %s auto-select: bits=%d "
                        "(from %d/%d expert sample, lloyd=%d, min_cos=%.4f)",
                        layer_prefix, bits, sample_experts, E,
                        moe_lloyd_iters, _lazy_min_cos)
            del sample
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        # Pack w13
        p13, lib13, lid13, sc13, mid13, st13 = xfp_moe_pack_v2(
            w13.float(), bits=bits, group_size=group_size,
            library_size=library_size,
        )
        layer.w13_weight.data = torch.empty(0)  # free BF16 immediately

        # Pack w2
        p2, lib2, lid2, sc2, mid2, st2 = xfp_moe_pack_v2(
            w2.float(), bits=bits, group_size=group_size,
            library_size=library_size,
        )
        layer.w2_weight.data = torch.empty(0)

        layer.w13_xfp_packed = nn.Parameter(p13.contiguous(), requires_grad=False)
        layer.w13_xfp_library = nn.Parameter(lib13.contiguous(), requires_grad=False)
        layer.w13_xfp_group_lib_id = nn.Parameter(lid13.contiguous(), requires_grad=False)
        layer.w13_xfp_group_scale = nn.Parameter(sc13.contiguous(), requires_grad=False)
        layer.w13_xfp_group_mid = nn.Parameter(mid13.contiguous(), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(p2.contiguous(), requires_grad=False)
        layer.w2_xfp_library = nn.Parameter(lib2.contiguous(), requires_grad=False)
        layer.w2_xfp_group_lib_id = nn.Parameter(lid2.contiguous(), requires_grad=False)
        layer.w2_xfp_group_scale = nn.Parameter(sc2.contiguous(), requires_grad=False)
        layer.w2_xfp_group_mid = nn.Parameter(mid2.contiguous(), requires_grad=False)

        layer._xfp_moe_bits = bits
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13
        layer._xfp_moe_K2 = K2
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = E
        # fpe stored for kernel compatibility (unused by V2 reference path)
        layer._xfp_moe_fpe13 = int(p13[0].numel())
        layer._xfp_moe_fpe2 = int(p2[0].numel())
        layer._xfp_moe_group_size = group_size
        layer._xfp_moe_library_size = library_size
        layer._xfp_moe_packed = True
        layer._xfp_v2 = True

        logger.info(
            "XFP-V2 MoE %s: %d experts %s + %s -> xfp%d (g=%d L=%d) | "
            "w13 cos=%.4f, w2 cos=%.4f, lib_p5(w13)=%.4f lib_p5(w2)=%.4f",
            layer_prefix, E, list(w13.shape[1:]), list(w2.shape[1:]),
            bits, group_size, library_size,
            st13.cos_sim, st2.cos_sim,
            st13.library_p5_cos, st2.library_p5_cos,
        )

        if cache is not None and layer_prefix:
            xfp_cache.save_moe_v2(cache, layer_prefix, layer)

        try:
            del layer.w13_weight, layer.w2_weight
        except AttributeError:
            pass

    def _v2_lazy_repack(self, layer: nn.Module) -> None:
        """One-time per-expert xfp_repack of V2 packed tensors + int32 cast
        of lib_id. Cached on layer as ``_xfp_v2_w{13,2}_packed_repacked``
        and ``_xfp_v2_w{13,2}_lib_id_int32``.
        """
        from vllm.multiquant.xfp.xfp_pack import xfp_repack

        if hasattr(layer, "_xfp_v2_w13_packed_repacked"):
            return

        E = int(layer._xfp_moe_E)
        # layer.w*_xfp_packed.data is [E, K_packed, N] raw int32
        rp13 = [xfp_repack(layer.w13_xfp_packed.data[e]) for e in range(E)]
        rp2  = [xfp_repack(layer.w2_xfp_packed.data[e])  for e in range(E)]
        layer._xfp_v2_w13_packed_repacked = torch.cat(
            [t.reshape(-1) for t in rp13], dim=0).contiguous()
        layer._xfp_v2_w2_packed_repacked = torch.cat(
            [t.reshape(-1) for t in rp2], dim=0).contiguous()
        # fpe = int32 words per expert (post-repack, same as pre-repack)
        layer._xfp_v2_fpe13 = int(rp13[0].numel())
        layer._xfp_v2_fpe2  = int(rp2[0].numel())

        lid13 = layer.w13_xfp_group_lib_id.data
        lid2  = layer.w2_xfp_group_lib_id.data
        layer._xfp_v2_w13_lib_id_int32 = (
            lid13 if lid13.dtype == torch.int32
            else lid13.to(torch.int32).contiguous()
        )
        layer._xfp_v2_w2_lib_id_int32 = (
            lid2 if lid2.dtype == torch.int32
            else lid2.to(torch.int32).contiguous()
        )

        # Free original packed tensors (we have the repacked ones now)
        layer.w13_xfp_packed.data = torch.empty(
            0, dtype=torch.int32, device=layer.w13_xfp_packed.data.device)
        layer.w2_xfp_packed.data = torch.empty(
            0, dtype=torch.int32, device=layer.w2_xfp_packed.data.device)

    def _moe_v2_dequant_to_bf16(self, layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """[DEPRECATED] V2 reference: per-stack dequant ALL experts to BF16.
        Replaced by direct V2 kernel call. Kept for numeric verification.
        """
        from vllm.multiquant.xfp.xfp_pack import dequant_xfp_v2_packed

        bits = int(layer._xfp_moe_bits)
        group_size = int(layer._xfp_moe_group_size)
        E = int(layer._xfp_moe_E)
        K13 = int(layer._xfp_moe_K13)
        K2 = int(layer._xfp_moe_K2)
        N13 = int(layer._xfp_moe_N13)
        N2 = int(layer._xfp_moe_N2)

        w13_dense = torch.empty(
            E, N13, K13, dtype=torch.bfloat16,
            device=layer.w13_xfp_packed.device,
        )
        w2_dense = torch.empty(
            E, N2, K2, dtype=torch.bfloat16,
            device=layer.w2_xfp_packed.device,
        )
        for e in range(E):
            w13_dense[e] = dequant_xfp_v2_packed(
                layer.w13_xfp_packed.data[e],
                layer.w13_xfp_library.data,
                layer.w13_xfp_group_lib_id.data[e],
                layer.w13_xfp_group_scale.data[e],
                layer.w13_xfp_group_mid.data[e],
                K=K13, bits=bits, group_size=group_size,
            ).to(torch.bfloat16)
            w2_dense[e] = dequant_xfp_v2_packed(
                layer.w2_xfp_packed.data[e],
                layer.w2_xfp_library.data,
                layer.w2_xfp_group_lib_id.data[e],
                layer.w2_xfp_group_scale.data[e],
                layer.w2_xfp_group_mid.data[e],
                K=K2, bits=bits, group_size=group_size,
            ).to(torch.bfloat16)
        return w13_dense, w2_dense

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

        # XFP-V2 apply: direct kernel call on packed indices — no dequant.
        if getattr(layer, "_xfp_v2", False):
            self._v2_lazy_repack(layer)
            return _xfp_moe_v2_forward(
                x, topk_weights, topk_ids,
                layer._xfp_v2_w13_packed_repacked,
                layer.w13_xfp_library.data,
                layer._xfp_v2_w13_lib_id_int32,
                layer.w13_xfp_group_scale.data,
                layer.w13_xfp_group_mid.data,
                layer._xfp_v2_w2_packed_repacked,
                layer.w2_xfp_library.data,
                layer._xfp_v2_w2_lib_id_int32,
                layer.w2_xfp_group_scale.data,
                layer.w2_xfp_group_mid.data,
                int(layer._xfp_moe_bits),
                int(layer._xfp_moe_K13), int(layer._xfp_moe_N13),
                int(layer._xfp_moe_K2), int(layer._xfp_moe_N2),
                int(layer._xfp_moe_group_size),
                int(layer._xfp_v2_fpe13), int(layer._xfp_v2_fpe2),
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
