# SPDX-License-Identifier: Apache-2.0
"""MultiQuant cache-only model loader.

Loads a model *without* reading the original bf16 (or any other) safetensor
weights. Instead, it relies on a previously populated MultiQuant disk
cache:

    <MULTIQUANT_CACHE_DIR>/<model_basename>/<policy-hash>/
        manifest.json
        <layer_prefix>/ (one dir per quantized layer)
            w13_xfp_packed.safetensors
            w13_xfp_codebook.safetensors
            ... etc
        residuals.safetensors    ← embeddings, RMSNorms, non-fp8 lm_heads

The cache is populated on a prior run with bf16 source
(``--load-format auto`` + ``MULTIQUANT_CACHE_DIR`` env). Once the cache
exists, subsequent runs can use ``--load-format multiquant`` on any
machine that has the cache directory — no bf16 disk presence required.

This unlocks TP=2 on hardware where the bf16 source wouldn't fit on a
single node's disk (e.g. Qwen 3.5-397B on GB10's 916 GB NVMe: bf16 =
~800 GB, cache ~200 GB + residuals ~3 GB ≈ 203 GB → fits).

NOTE: only MultiQuant-backed quantization methods populate the cache
(currently XFP). Other quant methods (standard autoround_rtn without
``--weight-dtype xfp`` override etc.) won't have their weights in the
cache and will fail to load via this path.
"""

from __future__ import annotations

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


class MultiQuantCacheOnlyLoader(BaseModelLoader):
    """Model loader that skips bf16 source and loads from MultiQuant cache.

    Relies on the standard ``load_model`` flow in ``BaseModelLoader``:
      1. ``initialize_model`` constructs the nn.Module tree. MoE layers
         with streaming-quant enabled create their expert tensors on the
         meta device (no CUDA alloc).
      2. ``self.load_weights`` (us) does the minimum: load residuals
         into non-quant params, and tell streaming-quant's per-layer
         counter that "all bytes are loaded" so
         ``process_weights_after_loading`` fires and hits the cache.
      3. ``process_weights_after_loading`` runs — for each quant layer,
         the cache-hit path loads packed/codebook tensors directly.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        """Nothing to download — cache is always local."""
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm.multiquant.weight_cache import MultiQuantWeightCache

        cache = MultiQuantWeightCache.get_active()
        if cache is None:
            raise RuntimeError(
                "--load-format multiquant requires MULTIQUANT_CACHE_DIR to "
                "be set and the cache to have been populated by a prior run "
                "with bf16 source. Run once with --load-format auto + your "
                "MultiQuant quant flags to populate the cache, then rerun "
                "with --load-format multiquant."
            )

        # 1) Load non-quant residuals (embed, norms, bias, lm_head if bf16).
        #    This also materializes any meta-device Parameters for those.
        cache.load_residuals(model)

        # 2) Materialize meta-device expert tensors with size-0 stubs on a
        #    real device. The XFP MoE cache-hit path reads
        #    ``layer.w13_weight.device`` to pick the target device for the
        #    packed tensors it's about to assign, and it calls
        #    ``torch.empty(0, device=p.data.device, ...)`` to shrink the
        #    bf16 storage — neither works on a meta tensor.
        #
        #    We keep size=0 here (no bf16 materialization!) because the
        #    cache-hit code shortly after reassigns .data to the packed
        #    tensor directly anyway.
        # [xfp_tp] Stub on this worker's physical GPU.
        import torch as _torch
        if _torch.cuda.is_available():
            _tp_rank = -1
            _tp_source = "default"
            try:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                )
                _tp_rank = get_tensor_model_parallel_rank()
                _tp_source = "vllm.distributed"
            except Exception as _e:
                try:
                    import torch.distributed as _td
                    if _td.is_initialized():
                        _tp_rank = _td.get_rank()
                        _tp_source = "torch.distributed"
                except Exception:
                    pass
            if _tp_rank < 0:
                _tp_rank = 0
                _tp_source = "fallback=0"
            _cur = _torch.cuda.current_device()
            _count = _torch.cuda.device_count()
            # Pick: if CUDA_VISIBLE_DEVICES restricts to 1 GPU, our cuda:0
            # is already the right GPU. Otherwise, use cuda:{tp_rank}.
            if _count > 1:
                real_dev = _torch.device(f"cuda:{_tp_rank}")
            else:
                real_dev = _torch.device("cuda:0")
            logger.info(
                "[xfp_tp] stub device=%s (tp_rank=%d [%s], "
                "current_device=%d, device_count=%d)",
                real_dev, _tp_rank, _tp_source, _cur, _count,
            )
        else:
            real_dev = _torch.device("cpu")
        # Stub MoE expert tensors only — these are the ones online_moe's
        # cache-hit path replaces wholesale. Plain ``.weight`` of regular
        # Linear layers must keep its original meta-or-real tensor with
        # the per-rank shape so load_residuals (which uses target.shape
        # to materialize) gets the correct shape; stubbing it to size-0
        # would degrade target.shape to (0,) and break the materialize.
        n_meta_fixed = 0
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            FusedMoE = None
        for _, module in model.named_modules():
            if FusedMoE is not None and not isinstance(module, FusedMoE):
                continue
            for attr in ("w13_weight", "w2_weight"):
                p = getattr(module, attr, None)
                if p is None:
                    continue
                if p.data.device.type == "meta":
                    p.data = _torch.empty(0, dtype=p.data.dtype,
                                          device=real_dev)
                    n_meta_fixed += 1

        logger.info(
            "MultiQuant cache-only load: residuals loaded, %d meta-params "
            "stubbed — quant layers will load from cache on "
            "process_weights_after_loading",
            n_meta_fixed,
        )

        # Diagnostic: surface any Linear-style param that's still 1-D / size-0
        # after residual load. This catches cases where a non-MoE Linear's
        # weight failed to materialize (vision QKV, language attn, etc.) and
        # would later crash the GEMM with "mat2 must be a matrix, got 1-D".
        suspicious = []
        for name, p in model.named_parameters():
            if not name.endswith(".weight"):
                continue
            if p.data.dim() >= 2:
                continue
            # Skip legitimately 1-D weights: RMSNorm / LayerNorm scalars,
            # rotary inv_freq, scale tensors, bias-like.
            if any(k in name for k in (
                    "norm", "rotary", "scale", "_bias",
                    "score_correction", "inv_freq", "patch_embed.proj",
            )):
                continue
            # MoE expert tensors are intentionally stubbed at this point —
            # XFP cache-hit replaces them on process_weights_after_loading.
            if name.endswith(".w13_weight") or name.endswith(".w2_weight"):
                continue
            suspicious.append((name, tuple(p.data.shape),
                               str(p.data.dtype),
                               p.data.device.type))
        if suspicious:
            logger.warning(
                "[xfp_tp] suspicious sub-2D weights post-residual-load "
                "(%d): %s%s",
                len(suspicious),
                suspicious[:8],
                f" ... +{len(suspicious) - 8} more" if len(suspicious) > 8
                else "",
            )
