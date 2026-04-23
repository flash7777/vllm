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


def load_moe(
    cache: MultiQuantWeightCache, layer_prefix: str,
    layer: nn.Module, device: torch.device,
) -> bool:
    # RIY-aware: if the layer has been set up with a pruned expert map
    # (local_num_experts < global_num_experts), stage the cache tensors on
    # CPU so we can drop masked experts *before* they consume VRAM. Without
    # this filter the cache-only loader would always allocate the full
    # cached expert set and defeat RIY's whole purpose.
    _expert_map = getattr(layer, "_expert_map", None)
    _local_E = getattr(layer, "local_num_experts", None)
    _global_E = getattr(layer, "global_num_experts", None)
    has_riy_filter = (
        _expert_map is not None
        and _local_E is not None
        and _global_E is not None
        and _local_E < _global_E
    )

    stage_device = torch.device("cpu") if has_riy_filter else device
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
        bits = int(meta["bits"])

        if has_riy_filter and cached_E < _global_E:
            logger.warning(
                "XFP MoE %s: cached E=%d < model global_E=%d — cache was "
                "packed with a different RIY profile; skipping filter",
                layer_prefix, cached_E, _global_E,
            )
            has_riy_filter = False

        if has_riy_filter:
            lut = 1 << bits
            emap_cpu = _expert_map.detach().to("cpu")
            kept_mask = emap_cpu >= 0  # [cached_E] bool
            n_kept = int(kept_mask.sum().item())
            if n_kept != _local_E:
                raise ValueError(
                    f"RIY filter: _expert_map keeps {n_kept} but "
                    f"local_num_experts={_local_E}")

            def _filter(t: torch.Tensor, per_expert: int) -> torch.Tensor:
                return t.view(cached_E, per_expert)[kept_mask].contiguous().view(-1)

            w13p = _filter(tensors["w13_xfp_packed"], fpe13).to(device)
            w13c = _filter(tensors["w13_xfp_codebook"], N13 * lut).to(device)
            w2p = _filter(tensors["w2_xfp_packed"], fpe2).to(device)
            w2c = _filter(tensors["w2_xfp_codebook"], N2 * lut).to(device)

            layer.w13_xfp_packed = nn.Parameter(w13p, requires_grad=False)
            layer.w13_xfp_codebook = nn.Parameter(w13c, requires_grad=False)
            layer.w2_xfp_packed = nn.Parameter(w2p, requires_grad=False)
            layer.w2_xfp_codebook = nn.Parameter(w2c, requires_grad=False)

            effective_E = _local_E
            logger.info(
                "XFP MoE %s ← cache +RIY: kept %d/%d experts (−%.1f%% VRAM)",
                layer_prefix, _local_E, cached_E,
                100.0 * (cached_E - _local_E) / cached_E,
            )
        else:
            layer.w13_xfp_packed = nn.Parameter(
                tensors["w13_xfp_packed"], requires_grad=False)
            layer.w13_xfp_codebook = nn.Parameter(
                tensors["w13_xfp_codebook"], requires_grad=False)
            layer.w2_xfp_packed = nn.Parameter(
                tensors["w2_xfp_packed"], requires_grad=False)
            layer.w2_xfp_codebook = nn.Parameter(
                tensors["w2_xfp_codebook"], requires_grad=False)
            effective_E = cached_E

        layer._xfp_moe_bits = bits
        layer._xfp_moe_K13 = int(meta["K13"])
        layer._xfp_moe_N13 = N13
        layer._xfp_moe_K2 = int(meta["K2"])
        layer._xfp_moe_N2 = N2
        layer._xfp_moe_E = effective_E
        layer._xfp_moe_fpe13 = fpe13
        layer._xfp_moe_fpe2 = fpe2
        layer._xfp_moe_packed = True
        return True
    except (KeyError, ValueError) as e:
        logger.warning(
            "XFP MoE cache: %s is incomplete (%s) — treating as miss",
            layer_prefix, e,
        )
        return False
