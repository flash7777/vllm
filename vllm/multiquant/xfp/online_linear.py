# SPDX-License-Identifier: Apache-2.0
"""XFP online linear method — BF16 → learned codebook + packed indices.

Quant-on-load pipeline for Linear layers:
    create_weights                — allocate BF16 ModelWeightParameter
    process_weights_after_loading — xfp_pack(W, bits), register packed +
                                    codebook, record stats, free BF16
    apply                         — fused depack + LUT + GEMM via
                                    _load_xfp_gemm(bits).xfp_gemm

Self-contained — no Archer imports. Stats are recorded in the central
MultiQuantPolicyRegistry for end-of-load summary and future auto-size.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.multiquant.policy import DTYPE_BITS

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


# ─── Custom op registration for torch.compile compatibility ────────────
#
# Dynamo cannot trace pybind11 C/C++ extensions (including torch's cpp_extension
# JIT-loaded modules). We wrap the kernel call in a torch custom op with a
# fake implementation so torch.compile's graph capture sees a well-defined
# operator with known output shape, while eager execution calls the real
# CUDA kernel. Pattern mirrored from ArcherOnlineLinearMethod's custom op
# registration in weight_quant/online_linear.py:143-155.

def _xfp_apply_impl(
    x: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    K: int,
    N_out: int,
) -> torch.Tensor:
    """Real impl: eager fused xfp_gemm. Native bf16 — no dtype conversion.

    packed is 1D (repacked via xfp_repack for coalesced warp reads).
    codebook is bf16 [N_out, 2^bits]. A and C are bf16.
    """
    from vllm.multiquant.xfp import xfp_kernel as _xk
    kernel = _xk._xfp_gemm_kernel
    if kernel is None:
        raise RuntimeError(
            "xfp custom op called before xfp_gemm kernel was loaded. "
            "process_weights_after_loading should have triggered JIT."
        )
    # A and C are bf16 (vLLM native), codebook stays fp16 (more precision).
    # No dtype conversion on the hot path — kernel handles mixed types.
    x_bf16 = x.to(torch.bfloat16).contiguous() if x.dtype != torch.bfloat16 else x.contiguous()
    cb_fp16 = codebook.to(torch.float16) if codebook.dtype != torch.float16 else codebook
    C = torch.zeros(x_bf16.shape[0], N_out,
                    dtype=torch.bfloat16, device=x.device)
    # v9 kernel has fused outlier scatter — detected by method name
    if hasattr(kernel, "xfp_gemm_v9"):
        # Will be called with outliers from _xfp_apply_fused_impl
        kernel.xfp_gemm_v9(
            x_bf16, packed.reshape(-1), cb_fp16, C, int(bits), int(K),
            torch.empty(0, dtype=torch.int64, device=x.device),
            torch.empty(0, dtype=torch.int64, device=x.device),
            torch.empty(0, dtype=torch.bfloat16, device=x.device),
        )
    else:
        # v12 primary + v11 fallback for K > K_SMEM_MAX_LINEAR.
        _xk.dispatch_linear_gemm(
            x_bf16, packed.reshape(-1), cb_fp16, C, int(bits), int(K))
    return C


def _xfp_apply_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    K: int,
    N_out: int,
) -> torch.Tensor:
    """Fake impl for torch.compile graph tracing. Matches real dtype."""
    return torch.empty(
        x.shape[0], N_out, dtype=torch.bfloat16, device=x.device
    )


try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="xfp_apply",
        op_func=_xfp_apply_impl,
        fake_impl=_xfp_apply_fake,
    )
    _xfp_op = torch.ops.vllm.xfp_apply
    logger.info("XFP custom op registered (torch.compile safe)")
except Exception as e:
    logger.warning(
        "XFP custom op registration failed: %s — "
        "torch.compile with --weight-dtype-*=xfp* may not work", e
    )
    _xfp_op = _xfp_apply_impl


# Outlier scatter-add op — registered separately so torch.compile can
# trace the full forward as a single graph with opaque operator boundary.
# Without this wrapper, dynamo specializes on the dynamic outlier count
# per layer and re-compiles for every weight matrix, causing a multi-minute
# compile-time explosion.

def _xfp_outlier_scatter_impl(
    base_out: torch.Tensor,      # [M, N_out] output tensor to accumulate into
    x: torch.Tensor,             # [M, K] input activations
    outlier_row: torch.Tensor,   # [n_outliers] int64 output-channel indices
    outlier_col: torch.Tensor,   # [n_outliers] int64 input-channel indices
    outlier_val: torch.Tensor,   # [n_outliers] fp16 weight values
) -> torch.Tensor:
    """Real impl: base_out[:, row] += x[:, col] * val."""
    # Layers without outliers pass zero-size index tensors. CUDA's
    # index_select rejects those with "invalid argument" on some kernels,
    # and scatter_add is a no-op anyway.
    if outlier_col.numel() == 0:
        return base_out
    x_cast = x.to(base_out.dtype)
    x_cols = x_cast.index_select(1, outlier_col)   # [M, n_outliers]
    contrib = x_cols * outlier_val.to(base_out.dtype)
    rows = outlier_row.unsqueeze(0).expand(x.shape[0], -1)
    return base_out.scatter_add(1, rows, contrib)


def _xfp_outlier_scatter_fake(
    base_out: torch.Tensor,
    x: torch.Tensor,
    outlier_row: torch.Tensor,
    outlier_col: torch.Tensor,
    outlier_val: torch.Tensor,
) -> torch.Tensor:
    """Fake impl: same shape as base_out."""
    return torch.empty_like(base_out)


try:
    direct_register_custom_op(
        op_name="xfp_outlier_scatter",
        op_func=_xfp_outlier_scatter_impl,
        fake_impl=_xfp_outlier_scatter_fake,
    )
    _xfp_outlier_op = torch.ops.vllm.xfp_outlier_scatter
    logger.info("XFP outlier-scatter custom op registered")
except Exception as e:
    logger.warning(
        "XFP outlier-scatter custom op registration failed: %s", e,
    )
    _xfp_outlier_op = _xfp_outlier_scatter_impl


class XFPLinearMethod(QuantizeMethodBase):
    """Learned-codebook quantization (XFP2/XFP3/XFP4) at model load time."""

    uses_meta_device: bool = False

    # Default outlier extraction settings (v3). Based on weight-distribution
    # inspection of GLM-4.7-Flash (tests/xfp/inspect_distribution.py), k=4
    # catches the 40σ attention outliers (kv_b_proj, q_b_proj) while
    # marking only 0.01–0.8 % of weights per layer.
    outlier_sigma: float = 4.0
    outlier_max_fraction: float = 0.02

    # Auto bit-width selection: minimum per-channel cosine similarity
    # for a candidate bit width to be accepted. Configurable via
    # XFP_MIN_COS environment variable. Default 0.98 — calibrated on
    # GLM-4.7-Flash where it separates routed experts (xfp3 OK at
    # cos 0.982) from attention (needs xfp4 at cos 0.994).
    auto_min_cos: float = float(os.environ.get("XFP_MIN_COS", "0.98"))

    def __init__(
        self,
        quant_config: "QuantizationConfig",
        dtype: str = "xfp4",
    ):
        if dtype not in ("xfp", "xfp2", "xfp3", "xfp4"):
            raise ValueError(
                f"XFPLinearMethod: unsupported dtype '{dtype}', "
                f"supported: xfp (auto), xfp2, xfp3, xfp4"
            )
        self.quant_config = quant_config
        self.dtype = dtype
        self.bits = DTYPE_BITS[dtype]  # 0 = auto, 2/3/4 = explicit

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.parameter import ModelWeightParameter

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
        layer._xfp_input_size = input_size
        layer._xfp_output_size = output_size
        layer._xfp_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """BF16 weight → xfp_pack → (packed uint32, codebook fp16).

        Records XFPPackStats on layer and into the central registry.
        Drops the BF16 weight after packing. Also eagerly compiles the
        xfp_gemm kernel so the forward-pass `apply()` is torch.compile
        friendly (no lazy JIT inside a graphed region).
        """
        if getattr(layer, "_xfp_packed_done", False):
            return

        from vllm.multiquant.xfp.xfp_pack import xfp_pack
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        W = layer.weight.data  # [N_out, K]
        device = W.device

        # ─── XFP-V2 branch — per-group + shared codebook library ──
        # Gated on env XFP_V2=1. Pack/cache uses the existing reuse map
        # (see doc/xfp/v2_codebook_library.md). Apply path falls back to
        # a Python reference until the v17_lib kernel ships (Phase 3).
        import os as _v2_os
        _xfp_v2 = _v2_os.environ.get("XFP_V2", "0") == "1"
        if _xfp_v2:
            return self._process_v2(layer, W, device)

        # ─── Cache check (skip Lloyd + pack if we've done this before) ──
        cache = MultiQuantWeightCache.get_active()
        layer_prefix = getattr(layer, "layer_name", "") or \
            getattr(layer, "prefix", "") or ""
        if cache is not None and layer_prefix and xfp_cache.load_linear(
                cache, layer_prefix, layer, device):
            # Eagerly JIT kernel so the forward path is torch.compile safe.
            _load_xfp_gemm(int(layer._xfp_bits))
            logger.info(
                "XFP %s ← cache (skip Lloyd) bits=%d K=%d N=%d",
                layer_prefix, layer._xfp_bits,
                layer._xfp_K, layer._xfp_N,
            )
            # Free the BF16 storage before dropping the attr (same reason as
            # online_moe.py cache-hit path — plain del leaves the storage
            # alive via layer._parameters until much later in the pipeline).
            p = layer._parameters.get("weight")
            if p is not None:
                p.data = torch.empty(0, device=p.data.device,
                                     dtype=p.data.dtype)
            try:
                del layer.weight
            except AttributeError:
                pass
            return

        # Auto bit-width selection: when dtype="xfp" (bits=0), run Lloyd
        # at candidates 2/3/4 and pick the lowest that passes the cos gate.
        bits = self.bits
        if bits == 0:
            from vllm.multiquant.xfp.xfp_pack import xfp_auto_select
            bits = xfp_auto_select(
                W.float(),
                candidates=(2, 3, 4),
                min_cos=self.auto_min_cos,
                outlier_sigma=self.outlier_sigma,
                outlier_max_fraction=self.outlier_max_fraction,
            )

        # Eagerly JIT-compile the kernel once, outside any torch.compile
        # graph. Subsequent .apply() calls will find the cached module.
        _load_xfp_gemm(bits)

        packed, codebook, o_idx, o_val, stats = xfp_pack(
            W.float(),
            bits=bits,
            also_score_widths=(),
            outlier_sigma=self.outlier_sigma,
            outlier_max_fraction=self.outlier_max_fraction,
        )

        # Repack for coalesced warp reads (v4opt kernel expects 1D repacked)
        from vllm.multiquant.xfp.xfp_pack import xfp_repack
        repacked = xfp_repack(packed)

        K = int(W.shape[1])
        N_out = int(W.shape[0])

        # MULTIQUANT_QUANT_ONLY: cache only, no VRAM retention. Used for
        # models that don't fit quantized in single-node UMA (e.g. 397B
        # XFP ~200 GB on 128 GB GB10). Serving happens in a later
        # --load-format multiquant run on potentially multiple nodes.
        import os as _os
        _quant_only = _os.environ.get(
            "MULTIQUANT_QUANT_ONLY", "").lower() in ("1", "true", "yes", "on")

        # Attach packed tensors to the layer. save_linear() (below)
        # reads them from layer.xfp_packed/.xfp_codebook/.xfp_outlier_*,
        # so the attach must happen regardless of mode. In quant-only
        # mode we strip them AFTER save, leaving nn.Parameters with
        # size-0 stubs so nothing lingers in VRAM.
        layer.xfp_packed = nn.Parameter(
            repacked.to(device), requires_grad=False
        )
        layer.xfp_codebook = nn.Parameter(
            codebook.to(device), requires_grad=False
        )

        if o_idx is not None and o_val is not None and o_idx.numel() > 0:
            o_idx_dev = o_idx.to(device)
            layer.xfp_outlier_row = nn.Parameter(
                (o_idx_dev // K).to(torch.int64), requires_grad=False
            )
            layer.xfp_outlier_col = nn.Parameter(
                (o_idx_dev % K).to(torch.int64), requires_grad=False
            )
            layer.xfp_outlier_val = nn.Parameter(
                o_val.to(device), requires_grad=False
            )
            layer._xfp_has_outliers = True
        else:
            layer._xfp_has_outliers = False

        layer._xfp_bits = bits
        layer._xfp_K = K
        layer._xfp_N = N_out
        layer._xfp_stats = stats
        layer._xfp_packed_done = True

        # Accumulate into the central registry for end-of-load summary
        from vllm.multiquant.policy import (
            MultiQuantPolicyRegistry,
            classify_layer,
        )
        reg = MultiQuantPolicyRegistry.get_active()
        if reg is not None:
            layer_prefix = getattr(layer, "layer_name", "") or \
                getattr(layer, "prefix", "") or ""
            layer_type = classify_layer(layer_prefix) or "other"
            reg.record_stats(layer_type, stats)

        auto_tag = f"(auto)" if self.bits == 0 else ""
        # Log a richer name (prefix or layer_name fallback) and, when the
        # pack stats are pathological (nan/inf), dump source-W stats so we
        # can tell input-pathology from pack-bug.
        _name = (getattr(layer, "layer_name", None)
                 or getattr(layer, "prefix", None)
                 or f"<{type(layer).__name__}>")
        import math as _math
        _bad = (
            _math.isnan(stats.mse) or _math.isnan(stats.cos_sim)
            or _math.isinf(stats.mse)
        )
        logger.info(
            "XFP %s [%dx%d] -> xfp%d%s | mse=%.3g cos=%.3f | "
            "3sigma=%.1f%% | outliers=%.3f%% (k=%.1f)",
            _name, stats.shape[0], stats.shape[1],
            bits, auto_tag, stats.mse, stats.cos_sim,
            100.0 * stats.outlier_ratio_k3,
            100.0 * stats.outlier_fraction,
            stats.outlier_sigma,
        )
        if _bad:
            _Wf = W.float()
            _norm = _Wf.norm().item()
            _has_nan = bool(torch.isnan(_Wf).any().item())
            _has_inf = bool(torch.isinf(_Wf).any().item())
            _all_zero = bool((_Wf == 0).all().item())
            _row_norms = _Wf.norm(dim=1)
            _zero_rows = int((_row_norms == 0).sum().item())
            _row_min = _row_norms.min().item()
            _row_max = _row_norms.max().item()
            logger.warning(
                "[xfp-nan] %s shape=%s | W.norm=%.3g all_zero=%s nan=%s inf=%s "
                "| row_norms: zero_rows=%d/%d min=%.3g max=%.3g | "
                "W mean=%.3g std=%.3g abs_max=%.3g",
                _name, tuple(W.shape), _norm, _all_zero, _has_nan, _has_inf,
                _zero_rows, _Wf.shape[0], _row_min, _row_max,
                _Wf.mean().item(), _Wf.std().item(), _Wf.abs().max().item(),
            )

        # Persist to disk cache (if enabled) so future loads skip Lloyd.
        if cache is not None and layer_prefix:
            xfp_cache.save_linear(cache, layer_prefix, layer)

        # Free the BF16 weight
        try:
            del layer.weight
        except AttributeError:
            pass

        # MULTIQUANT_QUANT_ONLY: after cache.save has persisted the
        # packed artefacts, strip the attached Parameters' storage so
        # this layer's VRAM footprint returns to ~zero. Critical for
        # models that are too big to keep quantized in UMA (397B).
        if _quant_only:
            for _attr in ("xfp_packed", "xfp_codebook",
                          "xfp_outlier_row", "xfp_outlier_col",
                          "xfp_outlier_val"):
                p = getattr(layer, _attr, None)
                if p is None:
                    continue
                p.data = torch.empty(0, dtype=p.data.dtype, device=p.data.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _process_v2(
        self,
        layer: nn.Module,
        W: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Phase 1+2 V2 path: pack via library + per-group, save_linear_v2.

        Apply will use a Python reference (slow); v17_lib kernel ships
        in Phase 3.
        """
        import os as _os
        from vllm.multiquant.xfp.xfp_pack import xfp_pack_v2
        from vllm.multiquant.xfp.xfp_kernel import _load_xfp_gemm
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        cache = MultiQuantWeightCache.get_active()
        layer_prefix = (getattr(layer, "layer_name", None)
                        or getattr(layer, "prefix", None) or "")

        # Cache check first — V2 shards live under a different cache_key
        # (XFP_V2 + group_size + library_size in the hash) so V1/V2 don't
        # collide on disk.
        if cache is not None and layer_prefix and xfp_cache.load_linear_v2(
                cache, layer_prefix, layer, device):
            logger.info(
                "XFP-V2 %s ← cache (skip Lloyd) bits=%d K=%d N=%d "
                "group=%d L=%d",
                layer_prefix, layer._xfp_bits, layer._xfp_K, layer._xfp_N,
                layer._xfp_group_size, layer._xfp_library_size,
            )
            p = layer._parameters.get("weight")
            if p is not None:
                p.data = torch.empty(0, device=p.data.device, dtype=p.data.dtype)
            try:
                del layer.weight
            except AttributeError:
                pass
            return

        # Fresh pack
        bits = self.bits if self.bits != 0 else 4  # V2 fixed bits=4 in phase 1
        group_size = int(_os.environ.get("XFP_GROUP_SIZE", "128"))
        library_size = int(_os.environ.get("XFP_LIBRARY_SIZE", "32"))

        # K must be divisible by group_size; otherwise the pack raises.
        Wf = W.float()
        K = Wf.shape[1]
        if K % group_size != 0:
            # Fall back to V1 for layers whose K is not divisible by
            # group_size (e.g. attention proj with odd intermediate size).
            logger.warning(
                "XFP-V2 %s: K=%d not divisible by group_size=%d → V1 fallback",
                layer_prefix, K, group_size,
            )
            return self._process_v1_fallback(layer, W, device)

        packed, library, lib_id, scale, mid, stats = xfp_pack_v2(
            Wf, bits=bits, group_size=group_size, library_size=library_size,
        )

        layer.xfp_packed = nn.Parameter(packed.contiguous(), requires_grad=False)
        layer.xfp_library = nn.Parameter(library.contiguous(), requires_grad=False)
        layer.xfp_group_lib_id = nn.Parameter(lib_id.contiguous(), requires_grad=False)
        layer.xfp_group_scale = nn.Parameter(scale.contiguous(), requires_grad=False)
        layer.xfp_group_mid = nn.Parameter(mid.contiguous(), requires_grad=False)
        layer._xfp_bits = bits
        layer._xfp_K = K
        layer._xfp_N = int(Wf.shape[0])
        layer._xfp_group_size = group_size
        layer._xfp_library_size = library_size
        layer._xfp_has_outliers = False  # outliers wired in later
        layer._xfp_v2 = True
        layer._xfp_packed_done = True
        layer._xfp_stats = stats

        # JIT-load the V1 kernel still, so cache-hit fallback paths work.
        # The actual V2 forward uses Python reference until v17_lib ships.
        try:
            _load_xfp_gemm(bits)
        except Exception:
            pass

        logger.info(
            "XFP-V2 %s [%dx%d] -> xfp%d (g=%d L=%d) | mse=%.3g cos=%.4f "
            "lib_p5=%.4f overhead=%.2f bits/param",
            layer_prefix, stats.shape[0], stats.shape[1], bits,
            group_size, library_size, stats.mse, stats.cos_sim,
            stats.library_p5_cos, stats.overhead_bits_per_param,
        )

        if cache is not None and layer_prefix:
            xfp_cache.save_linear_v2(cache, layer_prefix, layer)

        # Free BF16
        p = layer._parameters.get("weight")
        if p is not None:
            p.data = torch.empty(0, device=p.data.device, dtype=p.data.dtype)
        try:
            del layer.weight
        except AttributeError:
            pass

    def _process_v1_fallback(self, layer, W, device):
        """V1 path used when V2 prerequisites (K % group_size == 0) fail."""
        # Re-enter normal processing without the v2 flag. We just call
        # the rest of the V1 pipeline by toggling the env flag locally.
        import os as _os
        prev = _os.environ.get("XFP_V2", "0")
        _os.environ["XFP_V2"] = "0"
        try:
            return self.process_weights_after_loading(layer)
        finally:
            _os.environ["XFP_V2"] = prev

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # XFP-V2 reference apply (Python). Replace with v17_lib kernel
        # call once Phase 3 ships. Slow, but mathematically equivalent
        # to what the kernel must compute.
        if getattr(layer, "_xfp_v2", False):
            from vllm.multiquant.xfp.xfp_pack import dequant_xfp_v2_packed
            W_rec = dequant_xfp_v2_packed(
                layer.xfp_packed.data,
                layer.xfp_library.data,
                layer.xfp_group_lib_id.data,
                layer.xfp_group_scale.data,
                layer.xfp_group_mid.data,
                K=int(layer._xfp_K),
                bits=int(layer._xfp_bits),
                group_size=int(layer._xfp_group_size),
            ).to(x.dtype)
            return torch.nn.functional.linear(x, W_rec, bias)

        out_shape = x.shape[:-1] + (layer._xfp_N,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # v9 kernel: GEMM + outlier scatter in a single kernel launch.
        # A_row is loaded into SMEM once (shared across warps), outlier
        # accumulation happens inline — no separate kernel or x re-read.
        # Set XFP_USE_V9=1 to enable (disabled by default until validated).
        import os as _os
        from vllm.multiquant.xfp import xfp_kernel as _xk
        kernel = _xk._xfp_gemm_kernel
        use_v9 = _os.environ.get("XFP_USE_V9", "0") == "1"
        if use_v9 and kernel is not None and hasattr(kernel, "xfp_gemm_v9"):
            x_bf16 = reshaped_x.to(torch.bfloat16).contiguous() \
                if reshaped_x.dtype != torch.bfloat16 else reshaped_x.contiguous()
            cb_fp16 = layer.xfp_codebook.to(torch.float16) \
                if layer.xfp_codebook.dtype != torch.float16 \
                else layer.xfp_codebook
            C = torch.zeros(x_bf16.shape[0], layer._xfp_N,
                            dtype=torch.bfloat16, device=x.device)
            has_out = getattr(layer, "_xfp_has_outliers", False)
            kernel.xfp_gemm_v9(
                x_bf16,
                layer.xfp_packed.reshape(-1),
                cb_fp16,
                C,
                int(layer._xfp_bits),
                int(layer._xfp_K),
                layer.xfp_outlier_row if has_out else torch.empty(
                    0, dtype=torch.int64, device=x.device),
                layer.xfp_outlier_col if has_out else torch.empty(
                    0, dtype=torch.int64, device=x.device),
                layer.xfp_outlier_val if has_out else torch.empty(
                    0, dtype=torch.bfloat16, device=x.device),
            )
        else:
            # v8 fallback: separate GEMM + outlier scatter
            C = _xfp_op(
                reshaped_x,
                layer.xfp_packed,
                layer.xfp_codebook,
                int(layer._xfp_bits),
                int(layer._xfp_K),
                int(layer._xfp_N),
            )
            if getattr(layer, "_xfp_has_outliers", False):
                C = _xfp_outlier_op(
                    C,
                    reshaped_x,
                    layer.xfp_outlier_row,
                    layer.xfp_outlier_col,
                    layer.xfp_outlier_val,
                )

        if bias is not None:
            C = C + bias.to(C.dtype)
        return C.reshape(out_shape)
