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
# V2: per-group + shared codebook library. Distinct method strings so the
# generic cache (and pack_report.py) can tell V1/V2 shards apart.
_XFP_LINEAR_METHOD_V2 = "xfp_linear_v2"
_XFP_MOE_METHOD_V2 = "xfp_moe_v2"


def _slice_stacked(
    t: torch.Tensor, dim: int, sub_sizes: list[int],
    tp_world: int, tp_rank: int,
) -> torch.Tensor:
    """Sub-stack-aware TP narrow.

    For tensors stacked along ``dim`` as ``[A, B, ...]`` (e.g. QKV or
    gate+up), slice EACH sub-block independently by tp_world and concat.
    This matches what vLLM's ``QKVParallelLinear``/``MergedColumnParallelLinear``
    weight_loader does — naive ``narrow(dim, rank·N/TP, N/TP)`` would
    cut across sub-block boundaries (e.g. taking all gate, no up).
    """
    if tp_world == 1:
        return t
    parts = []
    offset = 0
    for sub in sub_sizes:
        if sub % tp_world != 0:
            raise ValueError(
                f"sub_size={sub} not divisible by tp_world={tp_world} "
                f"in stacked sizes {sub_sizes}")
        sub_local = sub // tp_world
        parts.append(t.narrow(dim, offset + tp_rank * sub_local, sub_local))
        offset += sub
    return torch.cat(parts, dim=dim)


def _detect_linear_sub_sizes(layer) -> tuple[str, list[int] | None]:
    """Identify TP-mode and sub-stack sizes from layer class.

    Returns:
        (mode, sub_sizes) where mode ∈ {"col", "row", "replicated"}
        and sub_sizes is None for non-stacked, else a list of sub-output
        sizes along the column dim.
    """
    try:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, RowParallelLinear,
            MergedColumnParallelLinear, QKVParallelLinear,
            ReplicatedLinear,
        )
    except Exception:
        # Defensive: degrade to simple narrow.
        return "col", None

    if isinstance(layer, RowParallelLinear):
        return "row", None
    if isinstance(layer, QKVParallelLinear):
        # Q/K/V with potentially asymmetric kv heads.
        try:
            q = int(layer.q_size)
            kv = int(layer.kv_size)
            return "col", [q, kv, kv]
        except Exception:
            return "col", None
    if isinstance(layer, MergedColumnParallelLinear):
        # gate_up_proj, in_proj_qkvz (4-stack), in_proj_ba (2-stack), …
        try:
            return "col", list(layer.output_sizes)
        except Exception:
            return "col", None
    if isinstance(layer, ColumnParallelLinear):
        return "col", None  # single output, naive narrow OK
    if isinstance(layer, ReplicatedLinear):
        return "replicated", None
    # Fallback for unknown — assume column (safer than skipping)
    return "col", None


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
    # cache.load returns (tensors, meta, tensor_meta) since refactor.
    tensors, meta, _ = res
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


# ─── XFP-V2 Linear: per-group + shared codebook library ───────────────

def save_linear_v2(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    """Save XFP-V2 packed linear: packed indices + library + group params.

    Tensors written:
      - xfp_packed       [K_packed, N] int32   (same layout as V1)
      - xfp_library      [L, n_centroids] fp16 (shared prototype codebooks)
      - xfp_group_lib_id [N, G] uint8/int32    (per-group library index)
      - xfp_group_scale  [N, G] fp16            (per-group magnitude)
      - xfp_group_mid    [N, G] fp16            (per-group midpoint)

    Outliers re-use the V1 fields (`xfp_outlier_*`) when present.
    """
    tensors: dict[str, torch.Tensor] = {
        "xfp_packed":       layer.xfp_packed.data,
        "xfp_library":      layer.xfp_library.data,
        "xfp_group_lib_id": layer.xfp_group_lib_id.data,
        "xfp_group_scale":  layer.xfp_group_scale.data,
        "xfp_group_mid":    layer.xfp_group_mid.data,
    }
    has_outliers = bool(getattr(layer, "_xfp_has_outliers", False))
    if has_outliers:
        tensors["xfp_outlier_row"] = layer.xfp_outlier_row.data
        tensors["xfp_outlier_col"] = layer.xfp_outlier_col.data
        tensors["xfp_outlier_val"] = layer.xfp_outlier_val.data
    metadata: dict = {
        "bits":         int(layer._xfp_bits),
        "K":            int(layer._xfp_K),
        "N":            int(layer._xfp_N),
        "group_size":   int(layer._xfp_group_size),
        "library_size": int(layer._xfp_library_size),
        "has_outliers": 1 if has_outliers else 0,
    }
    stats = getattr(layer, "_xfp_stats", None)
    if stats is not None and hasattr(stats, "to_dict"):
        try:
            metadata["stats"] = stats.to_dict()
        except Exception as e:
            logger.warning(
                "XFP-V2 cache: stats.to_dict failed for %s (%s) — saving without",
                layer_prefix, e,
            )
    return cache.save(layer_prefix, _XFP_LINEAR_METHOD_V2, tensors, metadata)


def load_linear_v2(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    """Load XFP-V2 packed linear shard with TP-aware narrow + device move.

    The cache stores TP=1 full-width tensors. For TP>1 each rank reads its
    own slice based on layer's ColumnParallel/RowParallel role:
      * ColumnParallel (qkv, gate_up, ...): narrow N on dim 1 of xfp_packed
        and on dim 0 of (group_lib_id, group_scale, group_mid).
      * RowParallel (o_proj, down, ...): narrow K_packed on dim 0 of
        xfp_packed and G on dim 1 of metadata. Library is shared, never
        narrowed.
    """
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_world, tp_rank = 1, 0

    # vLLM TP-rank-affinity quirk: layer.weight.device may be cuda:0 on
    # rank 1's worker. Force device to the worker's rank-mapped GPU.
    if tp_world > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{tp_rank}")
    elif device is not None and device.type == "cuda" and torch.cuda.is_available():
        cur = torch.cuda.current_device()
        if device.index is not None and device.index != cur:
            device = torch.device(f"cuda:{cur}")

    # Stage on CPU when narrowing — avoids loading full-width tensors on
    # the wrong GPU before slicing.
    needs_stage = tp_world > 1
    stage_device = torch.device("cpu") if needs_stage else device
    res = cache.load(layer_prefix, _XFP_LINEAR_METHOD_V2, stage_device)
    if res is None:
        return False
    tensors, meta, _ = res

    # Determine TP mode + sub-stack sizes from layer class.
    tp_dim = None
    sub_sizes_n: list[int] | None = None
    if tp_world > 1:
        mode, sub_sizes_n = _detect_linear_sub_sizes(layer)
        tp_dim = mode  # "row" | "col" | "replicated"
        logger.debug(
            "[xfp_v2_tp] load_linear %s tp_rank=%d cls=%s tp_dim=%s "
            "sub_sizes_n=%s", layer_prefix, tp_rank, type(layer).__name__,
            tp_dim, sub_sizes_n,
        )

    try:
        K_full = int(meta["K"])
        N_full = int(meta["N"])
        bits = int(meta["bits"])
        group_size = int(meta["group_size"])
        K_packed_full = (K_full + (8 if bits == 4 else 16) - 1) // (8 if bits == 4 else 16)
        G_full = K_full // group_size

        xfp_packed = tensors["xfp_packed"]
        lib_id = tensors["xfp_group_lib_id"]
        scale = tensors["xfp_group_scale"]
        mid = tensors["xfp_group_mid"]

        if tp_dim == "col":
            assert N_full % tp_world == 0, (
                f"N={N_full} not divisible by tp_world={tp_world}")
            N_tp = N_full // tp_world
            if sub_sizes_n and len(sub_sizes_n) > 1:
                # Stacked column-parallel (QKV / MergedColumn): slice each
                # sub-block independently, otherwise sub-block boundaries
                # are crossed and the per-rank tensor mixes Q/K/V wrongly.
                xfp_packed = _slice_stacked(
                    xfp_packed, 1, sub_sizes_n, tp_world, tp_rank)
                lib_id = _slice_stacked(
                    lib_id, 0, sub_sizes_n, tp_world, tp_rank)
                scale = _slice_stacked(
                    scale, 0, sub_sizes_n, tp_world, tp_rank)
                mid = _slice_stacked(
                    mid, 0, sub_sizes_n, tp_world, tp_rank)
            else:
                xfp_packed = xfp_packed.narrow(1, tp_rank * N_tp, N_tp)
                lib_id = lib_id.narrow(0, tp_rank * N_tp, N_tp)
                scale = scale.narrow(0, tp_rank * N_tp, N_tp)
                mid = mid.narrow(0, tp_rank * N_tp, N_tp)
            N_local = N_tp
            K_local = K_full
        elif tp_dim == "row":
            assert K_packed_full % tp_world == 0, (
                f"K_packed={K_packed_full} not divisible by tp_world={tp_world}")
            assert G_full % tp_world == 0, (
                f"G={G_full} not divisible by tp_world={tp_world}")
            K_packed_tp = K_packed_full // tp_world
            G_tp = G_full // tp_world
            xfp_packed = xfp_packed.narrow(0, tp_rank * K_packed_tp, K_packed_tp)
            lib_id = lib_id.narrow(1, tp_rank * G_tp, G_tp)
            scale = scale.narrow(1, tp_rank * G_tp, G_tp)
            mid = mid.narrow(1, tp_rank * G_tp, G_tp)
            N_local = N_full
            K_local = K_full // tp_world
        else:
            N_local = N_full
            K_local = K_full

        layer.xfp_packed = nn.Parameter(
            xfp_packed.contiguous().to(device), requires_grad=False)
        layer.xfp_library = nn.Parameter(
            tensors["xfp_library"].to(device), requires_grad=False)
        layer.xfp_group_lib_id = nn.Parameter(
            lib_id.contiguous().to(device), requires_grad=False)
        layer.xfp_group_scale = nn.Parameter(
            scale.contiguous().to(device), requires_grad=False)
        layer.xfp_group_mid = nn.Parameter(
            mid.contiguous().to(device), requires_grad=False)
        has_outliers = meta.get("has_outliers", "0") == "1"
        if has_outliers:
            # Outliers store per-coord (row, col, val); column-parallel
            # narrows row, row-parallel narrows col. Skip for now if TP>1
            # — correctness matters; will surface as missing outliers.
            if tp_world > 1:
                logger.warning(
                    "XFP-V2 %s: outlier-tensor TP-narrow not implemented; "
                    "skipping outliers for this layer at TP>1", layer_prefix,
                )
                has_outliers = False
            else:
                layer.xfp_outlier_row = nn.Parameter(
                    tensors["xfp_outlier_row"].to(device), requires_grad=False)
                layer.xfp_outlier_col = nn.Parameter(
                    tensors["xfp_outlier_col"].to(device), requires_grad=False)
                layer.xfp_outlier_val = nn.Parameter(
                    tensors["xfp_outlier_val"].to(device), requires_grad=False)
        layer._xfp_has_outliers = has_outliers
        layer._xfp_bits = bits
        layer._xfp_K = K_local
        layer._xfp_N = N_local
        layer._xfp_group_size = group_size
        layer._xfp_library_size = int(meta["library_size"])
        layer._xfp_stats = None
        layer._xfp_packed_done = True
        layer._xfp_v2 = True
        return True
    except (KeyError, AssertionError) as e:
        logger.warning(
            "XFP-V2 cache: %s is incomplete or TP-incompatible (%s) — miss",
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
    # cache.load returns (tensors, meta, tensor_meta); load_moe computes
    # its TP-slice plan inline so tensor_meta is not needed here.
    tensors, meta, _ = res
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


# ─── XFP-V2 MoE: per-group + shared codebook library ──────────────────

def save_moe_v2(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    """Save XFP-V2 MoE: w13/w2 packed indices + library + group params per stack.

    Each stack (w13 = gate+up merged, w2 = down) has its own library.
    """
    tensors: dict[str, torch.Tensor] = {
        "w13_xfp_packed":       layer.w13_xfp_packed.data,
        "w13_xfp_library":      layer.w13_xfp_library.data,
        "w13_xfp_group_lib_id": layer.w13_xfp_group_lib_id.data,
        "w13_xfp_group_scale":  layer.w13_xfp_group_scale.data,
        "w13_xfp_group_mid":    layer.w13_xfp_group_mid.data,
        "w2_xfp_packed":        layer.w2_xfp_packed.data,
        "w2_xfp_library":       layer.w2_xfp_library.data,
        "w2_xfp_group_lib_id":  layer.w2_xfp_group_lib_id.data,
        "w2_xfp_group_scale":   layer.w2_xfp_group_scale.data,
        "w2_xfp_group_mid":     layer.w2_xfp_group_mid.data,
    }
    metadata: dict = {
        "bits":         int(layer._xfp_moe_bits),
        "K13":          int(layer._xfp_moe_K13),
        "N13":          int(layer._xfp_moe_N13),
        "K2":           int(layer._xfp_moe_K2),
        "N2":           int(layer._xfp_moe_N2),
        "E":            int(layer._xfp_moe_E),
        "fpe13":        int(layer._xfp_moe_fpe13),
        "fpe2":         int(layer._xfp_moe_fpe2),
        "group_size":   int(layer._xfp_moe_group_size),
        "library_size": int(layer._xfp_moe_library_size),
    }
    return cache.save(layer_prefix, _XFP_MOE_METHOD_V2, tensors, metadata)


def load_moe_v2(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    """Load XFP-V2 MoE shard with TP-aware narrow + device move.

    Cache stores TP=1 full-width tensors. Per-rank slicing here, mirroring
    V1 ``load_moe``:
      * w13 (gate+up, ColumnParallel): narrow N13 on dim 2 of packed
        [E, K13_packed, N13] and on dim 1 of metadata [E, N13, G13].
      * w2 (down, RowParallel): narrow K2_packed on dim 1 of packed
        [E, K2_packed, N2] and G2 on dim 2 of metadata [E, N2, G2].
      * Library [L, lut] is shared across experts and ranks — never narrowed.
    """
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_world, tp_rank = 1, 0

    # vLLM TP-rank-affinity quirk: layer.weight.device may be cuda:0 on
    # rank 1's worker. Force to the worker's rank-mapped GPU.
    if tp_world > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{tp_rank}")
    elif device is not None and device.type == "cuda" and torch.cuda.is_available():
        cur = torch.cuda.current_device()
        if device.index is not None and device.index != cur:
            device = torch.device(f"cuda:{cur}")

    needs_stage = tp_world > 1
    stage_device = torch.device("cpu") if needs_stage else device
    res = cache.load(layer_prefix, _XFP_MOE_METHOD_V2, stage_device)
    if res is None:
        return False
    tensors, meta, _ = res

    try:
        bits = int(meta["bits"])
        K13 = int(meta["K13"])
        N13 = int(meta["N13"])
        K2 = int(meta["K2"])
        N2 = int(meta["N2"])
        group_size = int(meta["group_size"])
        vals_per_word = 8 if bits == 4 else (16 if bits == 2 else 10)
        K13_packed = (K13 + vals_per_word - 1) // vals_per_word
        K2_packed = (K2 + vals_per_word - 1) // vals_per_word
        G13 = K13 // group_size
        G2 = K2 // group_size

        w13p = tensors["w13_xfp_packed"]
        w13_lib_id = tensors["w13_xfp_group_lib_id"]
        w13_scale = tensors["w13_xfp_group_scale"]
        w13_mid = tensors["w13_xfp_group_mid"]
        w2p = tensors["w2_xfp_packed"]
        w2_lib_id = tensors["w2_xfp_group_lib_id"]
        w2_scale = tensors["w2_xfp_group_scale"]
        w2_mid = tensors["w2_xfp_group_mid"]

        N13_local, K13_local = N13, K13
        N2_local, K2_local = N2, K2
        if tp_world > 1:
            # w13 = FusedMoE gate_up stack: [gate, up] each = N13/2.
            # vLLM splits each sub-block separately — narrow on dim=2 must
            # be sub-stack-aware so rank gets [gate_local, up_local] not
            # ALL gate or ALL up.
            assert N13 % 2 == 0, (
                f"N13={N13} must be 2-stacked (gate+up) for FusedMoE")
            half = N13 // 2
            assert half % tp_world == 0, (
                f"N13/2={half} not divisible by tp_world={tp_world}")
            sub_n13 = [half, half]  # [gate, up]
            w13p = _slice_stacked(w13p, 2, sub_n13, tp_world, tp_rank)
            w13_lib_id = _slice_stacked(w13_lib_id, 1, sub_n13, tp_world, tp_rank)
            w13_scale  = _slice_stacked(w13_scale,  1, sub_n13, tp_world, tp_rank)
            w13_mid    = _slice_stacked(w13_mid,    1, sub_n13, tp_world, tp_rank)
            N13_local = N13 // tp_world
            logger.debug(
                "[xfp_v2_tp] load_moe %s tp_rank=%d N13_full=%d N13_local=%d "
                "(stacked gate+up sliced per rank)",
                layer_prefix, tp_rank, N13, N13_local,
            )

            # w2 = RowParallel on K2: narrow K2_packed (dim 1) and G2 (dim 2).
            assert K2_packed % tp_world == 0, (
                f"K2_packed={K2_packed} not divisible by tp_world={tp_world}")
            assert G2 % tp_world == 0, (
                f"G2={G2} not divisible by tp_world={tp_world}")
            K2_packed_tp = K2_packed // tp_world
            G2_tp = G2 // tp_world
            w2p = w2p.narrow(1, tp_rank * K2_packed_tp, K2_packed_tp)
            w2_lib_id = w2_lib_id.narrow(2, tp_rank * G2_tp, G2_tp)
            w2_scale  = w2_scale.narrow(2, tp_rank * G2_tp, G2_tp)
            w2_mid    = w2_mid.narrow(2, tp_rank * G2_tp, G2_tp)
            K2_local = K2 // tp_world

        layer.w13_xfp_packed = nn.Parameter(
            w13p.contiguous().to(device), requires_grad=False)
        layer.w13_xfp_library = nn.Parameter(
            tensors["w13_xfp_library"].to(device), requires_grad=False)
        layer.w13_xfp_group_lib_id = nn.Parameter(
            w13_lib_id.contiguous().to(device), requires_grad=False)
        layer.w13_xfp_group_scale = nn.Parameter(
            w13_scale.contiguous().to(device), requires_grad=False)
        layer.w13_xfp_group_mid = nn.Parameter(
            w13_mid.contiguous().to(device), requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(
            w2p.contiguous().to(device), requires_grad=False)
        layer.w2_xfp_library = nn.Parameter(
            tensors["w2_xfp_library"].to(device), requires_grad=False)
        layer.w2_xfp_group_lib_id = nn.Parameter(
            w2_lib_id.contiguous().to(device), requires_grad=False)
        layer.w2_xfp_group_scale = nn.Parameter(
            w2_scale.contiguous().to(device), requires_grad=False)
        layer.w2_xfp_group_mid = nn.Parameter(
            w2_mid.contiguous().to(device), requires_grad=False)

        layer._xfp_moe_bits         = bits
        layer._xfp_moe_K13          = K13_local
        layer._xfp_moe_N13          = N13_local
        layer._xfp_moe_K2           = K2_local
        layer._xfp_moe_N2           = N2_local
        layer._xfp_moe_E            = int(meta["E"])
        # fpe = numel of one expert's packed slice (post-narrow).
        layer._xfp_moe_fpe13        = int(layer.w13_xfp_packed.data[0].numel())
        layer._xfp_moe_fpe2         = int(layer.w2_xfp_packed.data[0].numel())
        layer._xfp_moe_group_size   = group_size
        layer._xfp_moe_library_size = int(meta["library_size"])
        layer._xfp_moe_packed       = True
        layer._xfp_v2               = True
        return True
    except (KeyError, ValueError, AssertionError) as e:
        logger.warning(
            "XFP-V2 MoE cache: %s is incomplete or TP-incompatible (%s) — miss",
            layer_prefix, e,
        )
        return False
