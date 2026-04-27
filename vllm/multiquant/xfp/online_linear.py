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

def _infer_linear_tp_role(layer: nn.Module) -> str:
    """Map a vllm Linear-base subclass to its TP-role keyword.

    Used by save_linear to record the slicing axis the generic loader
    will need at load time. Returns one of:
        ``column`` | ``row`` | ``qkv`` | ``merged_column`` | ``replicated``

    A MergedColumnParallelLinear with a *single* element in
    ``output_sizes`` is effectively a plain ColumnParallelLinear
    (Qwen3-Next / GatedDeltaNet uses this pattern for in_proj_qkvz and
    in_proj_ba) — downgrade to ``column`` so the slicer doesn't expect
    multi-shard ``shard_offsets`` we'd then have to populate.
    """
    try:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
            QKVParallelLinear,
            MergedColumnParallelLinear,
        )
    except ImportError:
        return "replicated"
    if isinstance(layer, QKVParallelLinear):
        return "qkv"
    if isinstance(layer, MergedColumnParallelLinear):
        out_sizes = getattr(layer, "output_sizes", None)
        if out_sizes is not None and len(out_sizes) == 1:
            return "column"
        return "merged_column"
    if isinstance(layer, RowParallelLinear):
        return "row"
    if isinstance(layer, ColumnParallelLinear):
        return "column"
    return "replicated"


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
        # Use the worker's current CUDA device, not W.device. Under
        # streaming-quant init layer.weight.data was reset to a size-0
        # stub on the constructor's default device (often cuda:0
        # regardless of the worker's tp_rank), so W.device would route
        # the per-rank tensors of an XFP cache-hit (packed, codebook,
        # outliers) onto cuda:0 even on rank 1 — mismatch at forward
        # time.
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = W.device

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

        # Schema v2: store pre-repack 2D ``[K_packed, N]`` so the cache is
        # TP-slicable. The on-device warp-interleaved repack happens
        # below for non-quant-only runs (so the forward kernel sees the
        # familiar flat layout) and inside ``load_linear`` for cache
        # warmstarts.
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
            packed.to(device), requires_grad=False
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
        logger.info(
            "XFP %s [%dx%d] -> xfp%d%s | mse=%.3g cos=%.3f | "
            "3sigma=%.1f%% | outliers=%.3f%% (k=%.1f)",
            getattr(layer, "layer_name", "?"),
            stats.shape[0], stats.shape[1],
            bits, auto_tag, stats.mse, stats.cos_sim,
            100.0 * stats.outlier_ratio_k3,
            100.0 * stats.outlier_fraction,
            stats.outlier_sigma,
        )

        # Persist to disk cache (if enabled) so future loads skip Lloyd.
        # Determine the TP role from the linear-layer's class so the
        # generic load-time slicer can recover this rank's slice.
        if cache is not None and layer_prefix:
            tp_role = _infer_linear_tp_role(layer)
            xfp_cache.save_linear(
                cache, layer_prefix, layer,
                tp_role=tp_role,
                input_dim_packed=0,  # packed is [K_packed, N]
                output_dim=1,
            )

        # Free the BF16 weight
        try:
            del layer.weight
        except AttributeError:
            pass

        # On-device warp-interleave repack so the forward kernel sees
        # the familiar flat layout. Skip in quant-only (no forward).
        if not _quant_only:
            from vllm.multiquant.xfp._repack import repack_2d_to_warp_flat
            layer.xfp_packed.data = repack_2d_to_warp_flat(
                layer.xfp_packed.data)

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

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Defensive lazy device-migration: under TP > 1 the streaming-quant
        # init path can leave xfp_* params on the wrong CUDA index (e.g.
        # cuda:0 on rank 1) because torch.cuda.set_device(local_rank) hasn't
        # taken effect at PWAL time. Migrate on first forward when the
        # activation lands on the worker's actual device, then cache so
        # subsequent forwards skip the check.
        if not getattr(layer, "_xfp_devmoved", False):
            x_dev = x.device
            for _attr in ("xfp_packed", "xfp_codebook",
                          "xfp_outlier_row", "xfp_outlier_col",
                          "xfp_outlier_val"):
                p = getattr(layer, _attr, None)
                if p is None:
                    continue
                if p.device != x_dev:
                    p.data = p.data.to(x_dev)
            layer._xfp_devmoved = True

        # Defensive lazy outlier TP-remap: cache writer marks
        # xfp_outlier_{row,col,val} as "replicated" (xfp_weight_cache.py:125),
        # so under TP > 1 every rank loads the *full-N* indices. For
        # column-parallel layers, output-channel indices in
        # [rank * N_per_rank, (rank+1) * N_per_rank) belong to this rank;
        # outside that range they OOB scatter_add. Detect via
        # outlier_row.max() vs N_per_rank, then filter+shift in place.
        # Symmetric for row-parallel via outlier_col vs K_per_rank.
        if (
            not getattr(layer, "_xfp_outliers_tpfixed", False)
            and getattr(layer, "_xfp_has_outliers", False)
        ):
            try:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )
                tp_rank = get_tensor_model_parallel_rank()
                tp_world = get_tensor_model_parallel_world_size()
            except Exception:
                tp_rank, tp_world = 0, 1

            row = layer.xfp_outlier_row.data
            col = layer.xfp_outlier_col.data
            val = layer.xfp_outlier_val.data
            N_per = int(layer._xfp_N)
            K_per = int(layer._xfp_K)

            if tp_world > 1 and row.numel() > 0:
                # Column-parallel detection: row indices reach into the
                # full-N range above N_per_rank.
                if int(row.max().item()) >= N_per:
                    lo = tp_rank * N_per
                    hi = (tp_rank + 1) * N_per
                    mask = (row >= lo) & (row < hi)
                    layer.xfp_outlier_row.data = (row[mask] - lo).contiguous()
                    layer.xfp_outlier_col.data = col[mask].contiguous()
                    layer.xfp_outlier_val.data = val[mask].contiguous()
                # Row-parallel detection: col indices reach into the
                # full-K range above K_per_rank.
                else:
                    col_after = layer.xfp_outlier_col.data
                    if col_after.numel() > 0 and \
                            int(col_after.max().item()) >= K_per:
                        lo = tp_rank * K_per
                        hi = (tp_rank + 1) * K_per
                        mask = (col_after >= lo) & (col_after < hi)
                        layer.xfp_outlier_row.data = \
                            layer.xfp_outlier_row.data[mask].contiguous()
                        layer.xfp_outlier_col.data = \
                            (col_after[mask] - lo).contiguous()
                        layer.xfp_outlier_val.data = \
                            layer.xfp_outlier_val.data[mask].contiguous()

            if layer.xfp_outlier_row.data.numel() == 0:
                layer._xfp_has_outliers = False
            layer._xfp_outliers_tpfixed = True

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
