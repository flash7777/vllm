# SPDX-License-Identifier: Apache-2.0
"""XFP adapter for the generic MultiQuant weight cache (schema v2).

Storage layout (schema v2):
    Linear:  ``xfp_packed: [K_packed, N]`` int32 + ``xfp_codebook: [N, lut]``
    MoE:     ``w13_xfp_packed: [E, K_packed, N]`` int32 +
             ``w13_xfp_codebook: [E, N, lut]`` (analog for w2)

Why pre-repack? The forward kernel still wants the warp-interleaved 1D
layout, but persisting that form makes K-dim slicing a multi-step
un-repack/slice/re-repack dance because ``xfp_repack`` interleaves K
inside warp_size. By keeping the cache 2D/3D pre-repack, TP slicing
becomes a single ``narrow()`` per dim and the load path runs
``repack_2d_to_warp_flat()`` / ``repack_3d_moe_to_flat()`` once on-device
after the slice.

The per-shard manifest carries ``tensor_meta`` (tp_role + slicing axes)
that ``slice_for_tp`` consumes generically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.multiquant.weight_cache import MultiQuantWeightCache
from vllm.multiquant._tp_slicer import slice_for_tp
from vllm.multiquant.xfp._repack import (
    repack_2d_to_warp_flat,
    repack_3d_moe_to_flat,
)

logger = init_logger(__name__)

_XFP_LINEAR_METHOD = "xfp_linear"
_XFP_MOE_METHOD = "xfp_moe"


# ─── XFP Linear ───────────────────────────────────────────────────────


def save_linear(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
    *, tp_role: str = "column", input_dim_packed: int = 0, output_dim: int = 1,
) -> bool:
    """Save XFP-packed Linear weights in v2 layout (pre-repack 2D).

    Args:
        tp_role: "column" for ColumnParallelLinear, "row" for
            RowParallelLinear, "qkv"/"merged_column" for fused projections,
            "replicated" for non-TP layers.
        input_dim_packed / output_dim: storage-tensor dims that map to K_packed
            (input) and N (output) respectively. For ``[K_packed, N]`` the
            defaults (0, 1) are correct.
    """
    # ``layer.xfp_packed`` is the pre-repack [K_packed, N] tensor that
    # online_linear.py now retains (it used to repack before save).
    packed_2d = layer.xfp_packed.data
    if packed_2d.ndim != 2:
        # Backward-compatible shim if a caller still hands in a
        # warp-interleaved flat tensor — refuse to save it as v2.
        raise ValueError(
            f"save_linear expects pre-repack 2D xfp_packed, got shape "
            f"{tuple(packed_2d.shape)}. Caller must skip xfp_repack before save.")
    tensors: dict[str, torch.Tensor] = {
        "xfp_packed": packed_2d,
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
        "K_orig": int(layer._xfp_K),
        "N_orig": int(layer._xfp_N),
        "has_outliers": 1 if has_outliers else 0,
    }
    stats = getattr(layer, "_xfp_stats", None)
    if stats is not None and hasattr(stats, "to_dict"):
        try:
            metadata["stats"] = stats.to_dict()
        except Exception as e:
            logger.warning(
                "XFP cache: stats.to_dict failed for %s (%s)",
                layer_prefix, e)
    # tensor_meta — what the generic slicer needs at load time.
    # The codebook for column-parallel is split on N (dim 0), for row-parallel
    # it's replicated (output channels = input dim of original W = K, which is
    # NOT TP-split; codebook is per-output-channel of W which after pack
    # corresponds to N rows, equal to hidden when row-parallel).
    if tp_role in ("column", "merged_column"):
        # merged_column with single output_sizes element behaves like
        # plain column for slicing purposes; the slicer handles missing
        # shard_offsets by falling back to a straight narrow.
        tensor_meta = {
            "xfp_packed": {"tp_role": tp_role, "output_dim": output_dim},
            "xfp_codebook": {"tp_role": "column", "output_dim": 0},
        }
    elif tp_role == "row":
        tensor_meta = {
            "xfp_packed": {
                "tp_role": "row", "input_dim_packed": input_dim_packed},
            "xfp_codebook": {"tp_role": "replicated"},
        }
    else:
        # qkv / replicated → callers should provide role-specific kwargs;
        # default to replicated as safest fallback.
        tensor_meta = {
            "xfp_packed": {"tp_role": tp_role or "replicated"},
            "xfp_codebook": {"tp_role": "replicated"},
        }
    if has_outliers:
        for k in ("xfp_outlier_row", "xfp_outlier_col", "xfp_outlier_val"):
            tensor_meta[k] = {"tp_role": "replicated"}
    return cache.save(
        layer_prefix, _XFP_LINEAR_METHOD, tensors, metadata,
        tensor_meta=tensor_meta)


def load_linear(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    """Load + slice + repack-on-device an XFP linear layer."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_world, tp_rank = 1, 0

    # Stage on CPU when TP > 1 to slice before transferring.
    stage = torch.device("cpu") if tp_world > 1 else device
    res = cache.load(layer_prefix, _XFP_LINEAR_METHOD, stage)
    if res is None:
        return False
    tensors, meta, tensor_meta = res
    try:
        bits = int(meta["bits"])
        K_orig = int(meta.get("K_orig", meta["K"]))
        N_orig = int(meta.get("N_orig", meta["N"]))
        has_outliers = meta.get("has_outliers", "0") == "1"
        slice_meta = {"bits": bits, "K_orig": K_orig, "N_orig": N_orig}

        # Apply per-tensor slicer
        sliced: dict[str, torch.Tensor] = {}
        for key, t in tensors.items():
            entry = tensor_meta.get(key, {"tp_role": "replicated"})
            sliced[key] = slice_for_tp(t, entry, tp_rank, tp_world, slice_meta)

        # Move to device + repack the packed tensor
        packed_2d = sliced["xfp_packed"].to(device).contiguous()
        codebook = sliced["xfp_codebook"].to(device).contiguous()

        # On-device repack to warp-interleaved 1D form expected by kernel.
        packed_flat = repack_2d_to_warp_flat(packed_2d)

        layer.xfp_packed = nn.Parameter(packed_flat, requires_grad=False)
        layer.xfp_codebook = nn.Parameter(codebook, requires_grad=False)
        if has_outliers:
            layer.xfp_outlier_row = nn.Parameter(
                sliced["xfp_outlier_row"].to(device), requires_grad=False)
            layer.xfp_outlier_col = nn.Parameter(
                sliced["xfp_outlier_col"].to(device), requires_grad=False)
            layer.xfp_outlier_val = nn.Parameter(
                sliced["xfp_outlier_val"].to(device), requires_grad=False)
        layer._xfp_has_outliers = has_outliers
        layer._xfp_bits = bits
        # K/N reflect the per-rank shape after slicing: N is reduced for
        # column-parallel, K_orig is reduced for row-parallel.
        N_eff = packed_2d.shape[1]
        # For row-parallel (K-slice), K_orig becomes K_orig // tp_world.
        role = tensor_meta.get("xfp_packed", {}).get("tp_role", "replicated")
        if role == "row" and tp_world > 1:
            K_eff = K_orig // tp_world
        else:
            K_eff = K_orig
        layer._xfp_K = K_eff
        layer._xfp_N = N_eff
        layer._xfp_stats = None
        layer._xfp_packed_done = True
        return True
    except (KeyError, ValueError) as e:
        logger.warning(
            "XFP cache: %s is incomplete or shape-mismatched (%s) — miss",
            layer_prefix, e)
        return False


# ─── XFP MoE ──────────────────────────────────────────────────────────


def save_moe(
    cache: MultiQuantWeightCache, layer_prefix: str, layer: nn.Module,
) -> bool:
    """Save XFP-packed MoE weights in v2 layout (pre-repack 3D)."""
    w13_packed_3d = layer.w13_xfp_packed.data
    w2_packed_3d = layer.w2_xfp_packed.data
    if w13_packed_3d.ndim != 3 or w2_packed_3d.ndim != 3:
        raise ValueError(
            f"save_moe expects 3D pre-repack [E, K_packed, N], got "
            f"w13={tuple(w13_packed_3d.shape)} w2={tuple(w2_packed_3d.shape)}")
    tensors: dict[str, torch.Tensor] = {
        "w13_xfp_packed":   w13_packed_3d,
        "w13_xfp_codebook": layer.w13_xfp_codebook.data,
        "w2_xfp_packed":    w2_packed_3d,
        "w2_xfp_codebook":  layer.w2_xfp_codebook.data,
    }
    metadata: dict = {
        "bits":  int(layer._xfp_moe_bits),
        "K13":   int(layer._xfp_moe_K13),
        "N13":   int(layer._xfp_moe_N13),
        "K2":    int(layer._xfp_moe_K2),
        "N2":    int(layer._xfp_moe_N2),
        "E":     int(layer._xfp_moe_E),
        "K_orig":    int(layer._xfp_moe_K13),
        "K_orig_w2": int(layer._xfp_moe_K2),
    }
    stats13 = getattr(layer, "_xfp_moe_stats13", None)
    stats2 = getattr(layer, "_xfp_moe_stats2", None)
    if stats13 is not None and hasattr(stats13, "to_dict"):
        try:
            metadata["w13_stats"] = stats13.to_dict()
        except Exception as e:
            logger.warning(
                "XFP MoE cache: stats13.to_dict failed for %s (%s)",
                layer_prefix, e)
    if stats2 is not None and hasattr(stats2, "to_dict"):
        try:
            metadata["w2_stats"] = stats2.to_dict()
        except Exception as e:
            logger.warning(
                "XFP MoE cache: stats2.to_dict failed for %s (%s)",
                layer_prefix, e)
    tensor_meta = {
        "w13_xfp_packed":   {"tp_role": "fused_moe_w13"},
        "w13_xfp_codebook": {"tp_role": "fused_moe_codebook_w13"},
        "w2_xfp_packed":    {"tp_role": "fused_moe_w2"},
        "w2_xfp_codebook":  {"tp_role": "fused_moe_codebook_w2"},
    }
    return cache.save(
        layer_prefix, _XFP_MOE_METHOD, tensors, metadata,
        tensor_meta=tensor_meta)


def load_moe(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    """Load + RIY-filter + TP-slice + repack-on-device an XFP MoE layer."""
    # RIY: layer.local_num_experts < global → drop pruned experts.
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

    needs_stage = has_riy_filter or tp_world > 1
    stage = torch.device("cpu") if needs_stage else device
    res = cache.load(layer_prefix, _XFP_MOE_METHOD, stage)
    if res is None:
        return False
    tensors, meta, tensor_meta = res
    try:
        cached_E = int(meta["E"])
        bits = int(meta["bits"])
        K13 = int(meta["K13"])
        N13 = int(meta["N13"])
        K2 = int(meta["K2"])
        N2 = int(meta["N2"])
        slice_meta = {
            "bits": bits, "K_orig": K13, "K_orig_w2": K2, "N_orig": N13,
        }

        if has_riy_filter and cached_E < _global_E:
            logger.warning(
                "XFP MoE %s: cached E=%d < model global_E=%d — cache packed "
                "with a different RIY profile; skipping filter",
                layer_prefix, cached_E, _global_E)
            has_riy_filter = False

        # Step 1: RIY expert filter on dim 0 (E).
        if has_riy_filter:
            emap_cpu = _expert_map.detach().to("cpu")
            kept_mask = emap_cpu >= 0
            n_kept = int(kept_mask.sum().item())
            if n_kept != _local_E:
                raise ValueError(
                    f"RIY filter: _expert_map keeps {n_kept} but "
                    f"local_num_experts={_local_E}")
        else:
            kept_mask = None
            n_kept = cached_E

        def _select_experts(t: torch.Tensor) -> torch.Tensor:
            return t if kept_mask is None else t[kept_mask]

        # Step 2: per-tensor TP slice via generic slicer.
        sliced: dict[str, torch.Tensor] = {}
        for key, t in tensors.items():
            t_filtered = _select_experts(t)
            entry = tensor_meta.get(key, {"tp_role": "replicated"})
            sliced[key] = slice_for_tp(
                t_filtered, entry, tp_rank, tp_world, slice_meta)

        # Step 3: move to device + repack the packed tensors (3D → flat).
        w13_packed_3d = sliced["w13_xfp_packed"].to(device).contiguous()
        w13_codebook = sliced["w13_xfp_codebook"].to(device).contiguous()
        w2_packed_3d = sliced["w2_xfp_packed"].to(device).contiguous()
        w2_codebook = sliced["w2_xfp_codebook"].to(device).contiguous()

        w13_flat = repack_3d_moe_to_flat(w13_packed_3d)
        w2_flat = repack_3d_moe_to_flat(w2_packed_3d)

        layer.w13_xfp_packed = nn.Parameter(w13_flat, requires_grad=False)
        layer.w13_xfp_codebook = nn.Parameter(w13_codebook, requires_grad=False)
        layer.w2_xfp_packed = nn.Parameter(w2_flat, requires_grad=False)
        layer.w2_xfp_codebook = nn.Parameter(w2_codebook, requires_grad=False)

        # Effective shapes after RIY + TP slicing.
        N13_eff = w13_packed_3d.shape[2]
        K2_eff = (K2 // tp_world) if tp_world > 1 else K2
        # fpe = K_groups * N * WS for the kernel's index math.
        WS = 32
        K13_packed_eff = (K13 * bits) // 32  # column-parallel: K unchanged
        K_g13 = (K13_packed_eff + WS - 1) // WS
        K2_packed_eff = (K2_eff * bits) // 32  # row-parallel: K reduced
        K_g2 = (K2_packed_eff + WS - 1) // WS
        fpe13 = K_g13 * N13_eff * WS
        fpe2 = K_g2 * N2 * WS

        layer._xfp_moe_bits = bits
        layer._xfp_moe_K13 = K13
        layer._xfp_moe_N13 = N13_eff
        layer._xfp_moe_K2 = K2_eff
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = n_kept
        layer._xfp_moe_fpe13 = fpe13
        layer._xfp_moe_fpe2 = fpe2
        layer._xfp_moe_packed = True

        if has_riy_filter or tp_world > 1:
            riy_tag = f"+RIY({_local_E}/{cached_E})" if has_riy_filter else ""
            tp_tag = (
                f"+TP{tp_world}[rank{tp_rank}] N13 {N13}->{N13_eff} "
                f"K2 {K2}->{K2_eff}" if tp_world > 1 else ""
            )
            logger.info(
                "XFP MoE %s ← cache v2 %s %s",
                layer_prefix, riy_tag, tp_tag)
        return True
    except (KeyError, ValueError) as e:
        logger.warning(
            "XFP MoE cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e)
        return False
