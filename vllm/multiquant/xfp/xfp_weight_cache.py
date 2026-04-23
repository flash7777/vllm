# SPDX-License-Identifier: Apache-2.0
"""XFP adapter for the generic MultiQuant weight cache.

The actual plumbing (key, manifest, safetensors I/O, stats) lives in
``vllm.multiquant.weight_cache``. This module only knows how to pack
an XFP-quantized layer's attributes into a tensor dict and reconstruct
them on load.

Other quant methods (Archer / TurboQuant / RotorQuant / AutoRound RTN)
follow the same pattern: a per-method adapter module next to the method's
online_linear/online_moe, using the generic cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.multiquant.weight_cache import MultiQuantWeightCache

logger = init_logger(__name__)

# ─── XFP Linear ───────────────────────────────────────────────────────

_XFP_LINEAR_METHOD = "xfp_linear"
_XFP_MOE_METHOD = "xfp_moe"


def save_linear(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    tensors: dict[str, torch.Tensor] = {
        "xfp_packed": layer.xfp_packed.data,
        "xfp_codebook": layer.xfp_codebook.data,
    }
    has_outliers = bool(getattr(layer, "_xfp_has_outliers", False))
    if has_outliers:
        tensors["xfp_outlier_row"] = layer.xfp_outlier_row.data
        tensors["xfp_outlier_col"] = layer.xfp_outlier_col.data
        tensors["xfp_outlier_val"] = layer.xfp_outlier_val.data
    metadata: dict = {
        "bits": int(layer._xfp_bits),
        "K": int(layer._xfp_K),
        "N": int(layer._xfp_N),
        "has_outliers": 1 if has_outliers else 0,
    }
    # Attach full XFPPackStats (cos_sim, mse, outlier_fraction, cos_hist,
    # outlier_hist, recommended_bits, mse_per_bits, ...) for paper-time
    # offline analysis via tools/pack_report.py. Negligible size.
    stats = getattr(layer, "_xfp_stats", None)
    if stats is not None and hasattr(stats, "to_dict"):
        try:
            metadata["stats"] = stats.to_dict()
        except Exception as e:
            logger.warning(
                "XFP cache: stats.to_dict failed for %s (%s) — saving without",
                layer_prefix, e,
            )
    return cache.save(layer_prefix, _XFP_LINEAR_METHOD, tensors, metadata)


def load_linear(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    res = cache.load(layer_prefix, _XFP_LINEAR_METHOD, device)
    if res is None:
        return False
    tensors, meta = res
    try:
        layer.xfp_packed = nn.Parameter(
            tensors["xfp_packed"], requires_grad=False)
        layer.xfp_codebook = nn.Parameter(
            tensors["xfp_codebook"], requires_grad=False)
        has_outliers = meta.get("has_outliers", "0") == "1"
        if has_outliers:
            layer.xfp_outlier_row = nn.Parameter(
                tensors["xfp_outlier_row"], requires_grad=False)
            layer.xfp_outlier_col = nn.Parameter(
                tensors["xfp_outlier_col"], requires_grad=False)
            layer.xfp_outlier_val = nn.Parameter(
                tensors["xfp_outlier_val"], requires_grad=False)
        layer._xfp_has_outliers = has_outliers
        layer._xfp_bits = int(meta["bits"])
        layer._xfp_K = int(meta["K"])
        layer._xfp_N = int(meta["N"])
        layer._xfp_stats = None
        layer._xfp_packed_done = True
        return True
    except KeyError as e:
        logger.warning(
            "XFP cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e,
        )
        return False


# ─── XFP MoE ──────────────────────────────────────────────────────────

def save_moe(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    tensors: dict[str, torch.Tensor] = {
        "w13_xfp_packed":   layer.w13_xfp_packed.data,
        "w13_xfp_codebook": layer.w13_xfp_codebook.data,
        "w2_xfp_packed":    layer.w2_xfp_packed.data,
        "w2_xfp_codebook":  layer.w2_xfp_codebook.data,
    }
    metadata: dict = {
        "bits":  int(layer._xfp_moe_bits),
        "K13":   int(layer._xfp_moe_K13),
        "N13":   int(layer._xfp_moe_N13),
        "K2":    int(layer._xfp_moe_K2),
        "N2":    int(layer._xfp_moe_N2),
        "E":     int(layer._xfp_moe_E),
        "fpe13": int(layer._xfp_moe_fpe13),
        "fpe2":  int(layer._xfp_moe_fpe2),
    }
    # Attach per-projection stats (w13 = gate_up, w2 = down) for offline
    # analysis. They are the mean-over-experts stats of the pack.
    stats13 = getattr(layer, "_xfp_moe_stats13", None)
    stats2 = getattr(layer, "_xfp_moe_stats2", None)
    if stats13 is not None and hasattr(stats13, "to_dict"):
        try:
            metadata["w13_stats"] = stats13.to_dict()
        except Exception as e:
            logger.warning(
                "XFP MoE cache: stats13.to_dict failed for %s (%s)",
                layer_prefix, e,
            )
    if stats2 is not None and hasattr(stats2, "to_dict"):
        try:
            metadata["w2_stats"] = stats2.to_dict()
        except Exception as e:
            logger.warning(
                "XFP MoE cache: stats2.to_dict failed for %s (%s)",
                layer_prefix, e,
            )
    return cache.save(layer_prefix, _XFP_MOE_METHOD, tensors, metadata)


_WARP_SIZE = 32


def load_moe(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    """Load XFP-packed MoE weights for a single layer.

    Applies two load-time filters that the classic vLLM weight_loader does
    inline but that this cache-only path needs to do explicitly:

    1. **RIY expert-skip** — if ``layer._expert_map`` marks a subset as
       pruned (``local_num_experts < global_num_experts``), only kept
       experts are materialized in VRAM.
    2. **TP slice** — the cache stores TP=1 full-width packed tensors. For
       TP>1 serves, each rank reads its own slice:
         * w13 is ColumnParallel (output N-dim split): narrow N on dim=1
           in the repacked view ``[E, K_groups, N, WS]``.
         * w2 is RowParallel (input K-dim split): un-repack into
           ``[E, K_packed_padded, N]``, narrow K_packed on dim=1, re-repack.
           Un-repack/re-repack is zero-copy + one contiguous() call, plus
           F.pad when K_packed_tp is not a multiple of warp_size (affects
           e.g. bits=3 at TP=2: K_packed_tp=48 → padded to 64).

    The cache-save side is unchanged; a single shard serves any TP world
    size as long as ``K * bits`` is divisible by ``32 * tp_world`` (holds
    for the usual (K, bits, TP) combinations with K = intermediate_size).
    """
    _expert_map = getattr(layer, "_expert_map", None)
    _local_E = getattr(layer, "local_num_experts", None)
    _global_E = getattr(layer, "global_num_experts", None)
    has_riy_filter = (
        _expert_map is not None
        and _local_E is not None
        and _global_E is not None
        and _local_E < _global_E
    )

    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_world, tp_rank = 1, 0

    # Stage on CPU if either RIY or TP-slice changes the final shape.
    needs_stage = has_riy_filter or tp_world > 1
    stage_device = torch.device("cpu") if needs_stage else device
    res = cache.load(layer_prefix, _XFP_MOE_METHOD, stage_device)
    if res is None:
        return False
    tensors, meta = res
    try:
        cached_E = int(meta["E"])
        fpe13 = int(meta["fpe13"])
        fpe2 = int(meta["fpe2"])
        N13 = int(meta["N13"])
        N2 = int(meta["N2"])
        K13 = int(meta["K13"])
        K2 = int(meta["K2"])
        bits = int(meta["bits"])
        lut = 1 << bits
        WS = _WARP_SIZE

        if has_riy_filter and cached_E < _global_E:
            logger.warning(
                "XFP MoE %s: cached E=%d < model global_E=%d — cache was "
                "packed with a different RIY profile; skipping filter",
                layer_prefix, cached_E, _global_E,
            )
            has_riy_filter = False

        if tp_world > 1:
            if N13 % tp_world != 0 or K2 % tp_world != 0:
                raise ValueError(
                    f"TP slice: N13={N13}, K2={K2} must be divisible by "
                    f"tp_world={tp_world}")
            # For w2 bit-slice: K2*bits must align at 32*tp_world boundary.
            if (K2 * bits) % (32 * tp_world) != 0:
                raise ValueError(
                    f"TP slice w2: K2*bits={K2*bits} not divisible by "
                    f"32*tp_world={32*tp_world}; re-pack at target TP")

        # Packed shapes (post-repack): per expert w13 flat of size
        # K_g13 * N13 * WS, w2 flat of size K_g2 * N2 * WS.
        K13_packed = (K13 * bits) // 32
        K_g13 = (K13_packed + WS - 1) // WS
        K2_packed = (K2 * bits) // 32
        K_g2 = (K2_packed + WS - 1) // WS

        # Step 1: RIY expert filter (if active) on E-dim.
        if has_riy_filter:
            emap_cpu = _expert_map.detach().to("cpu")
            kept_mask = emap_cpu >= 0  # [cached_E] bool
            n_kept = int(kept_mask.sum().item())
            if n_kept != _local_E:
                raise ValueError(
                    f"RIY filter: _expert_map keeps {n_kept} but "
                    f"local_num_experts={_local_E}")
        else:
            kept_mask = None
            n_kept = cached_E

        def _select_experts(t: torch.Tensor) -> torch.Tensor:
            """Slice the E-dim if RIY active, else pass through."""
            if kept_mask is None:
                return t
            return t[kept_mask]

        # Step 2: per-tensor reshape → optional RIY filter → optional TP slice → flatten.
        # w13 packed [E, K_g13, N13, WS] — slice N on dim 2 for TP
        w13_view = tensors["w13_xfp_packed"].view(cached_E, K_g13, N13, WS)
        w13_view = _select_experts(w13_view)
        if tp_world > 1:
            N13_tp = N13 // tp_world
            w13_view = w13_view.narrow(2, tp_rank * N13_tp, N13_tp)
        else:
            N13_tp = N13
        w13p = w13_view.contiguous().reshape(-1)
        fpe13_tp = K_g13 * N13_tp * WS

        # w13 codebook [E, N13, lut]
        cb13_view = tensors["w13_xfp_codebook"].view(cached_E, N13, lut)
        cb13_view = _select_experts(cb13_view)
        if tp_world > 1:
            cb13_view = cb13_view.narrow(1, tp_rank * N13_tp, N13_tp)
        w13c = cb13_view.contiguous().reshape(-1)

        # w2 packed [E, K_g2, N2, WS] — un-repack → slice K → re-repack
        w2_view = tensors["w2_xfp_packed"].view(cached_E, K_g2, N2, WS)
        w2_view = _select_experts(w2_view)
        # un-repack: [E, K_g2, N2, WS] → [E, K_g2, WS, N2] → reshape [E, K_g2*WS, N2]
        w2_2d = (
            w2_view.permute(0, 1, 3, 2)
            .contiguous()
            .reshape(n_kept, K_g2 * WS, N2)
        )
        # Slice K_packed along dim 1
        if tp_world > 1:
            K2_packed_tp = K2_packed // tp_world
            w2_2d = w2_2d.narrow(1, tp_rank * K2_packed_tp, K2_packed_tp)
            K2_tp = K2 // tp_world
            K_g2_tp = (K2_packed_tp + WS - 1) // WS
            pad_needed = K_g2_tp * WS - K2_packed_tp
            if pad_needed > 0:
                import torch.nn.functional as F
                w2_2d = F.pad(w2_2d, (0, 0, 0, pad_needed), value=0)
        else:
            K2_tp = K2
            K_g2_tp = K_g2
        # Re-repack: [E, K_g2_tp*WS, N2] → [E, K_g2_tp, WS, N2] → permute → [E, K_g2_tp, N2, WS] → flat
        w2p = (
            w2_2d.view(n_kept, K_g2_tp, WS, N2)
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(-1)
        )
        fpe2_tp = K_g2_tp * N2 * WS

        # w2 codebook [E, N2, lut] — N2 is NOT TP-split (output dim of w2)
        cb2_view = tensors["w2_xfp_codebook"].view(cached_E, N2, lut)
        cb2_view = _select_experts(cb2_view)
        w2c = cb2_view.contiguous().reshape(-1)

        # Move to device + attach.
        logger.info(
            "[xfp_tp] load_moe %s writing to device=%s (tp_rank=%d, "
            "current_device=%d, w13p on %s)",
            layer_prefix, device, tp_rank,
            torch.cuda.current_device() if torch.cuda.is_available() else -1,
            w13p.device,
        )
        layer.w13_xfp_packed = nn.Parameter(w13p.to(device), requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(w13c.to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(w2p.to(device), requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(w2c.to(device), requires_grad=False)

        effective_E = n_kept
        layer._xfp_moe_bits = bits
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13_tp
        layer._xfp_moe_K2 = K2_tp
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = effective_E
        layer._xfp_moe_fpe13 = fpe13_tp
        layer._xfp_moe_fpe2 = fpe2_tp
        layer._xfp_moe_packed = True

        if has_riy_filter or tp_world > 1:
            riy_tag = (
                f"+RIY({_local_E}/{cached_E})" if has_riy_filter else ""
            )
            tp_tag = (
                f"+TP{tp_world}[rank{tp_rank}] "
                f"N13 {N13}->{N13_tp} K2 {K2}->{K2_tp}"
                if tp_world > 1 else ""
            )
            # VRAM ratio vs full-cached, per-rank.
            saved_pct = 100.0 * (
                1.0 - (n_kept * (fpe13_tp + fpe2_tp))
                / (cached_E * (fpe13 + fpe2))
            )
            logger.info(
                "XFP MoE %s ← cache %s %s (−%.1f%% VRAM/rank)",
                layer_prefix, riy_tag, tp_tag, saved_pct,
            )
        return True
    except (KeyError, ValueError) as e:
        logger.warning(
            "XFP MoE cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e,
        )
        return False
