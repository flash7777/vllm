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
        # [xfp_tp] Stub on this worker's physical GPU. Depending on how
        # the worker was launched, cuda:0 may refer to the shared GPU 0
        # for every rank (CUDA_VISIBLE_DEVICES unset) — that's the bug
        # that made TP>1 serves OOM on rank 0 while rank 1's GPU sat idle.
        # Fix: pick device = cuda:{tp_rank} when >1 GPUs are visible, else
        # cuda:0 (CUDA_VISIBLE_DEVICES already restricted our view).
        import torch as _torch
        if _torch.cuda.is_available():
            try:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                )
                _rank = get_tensor_model_parallel_rank()
            except Exception:
                _rank = 0
            if _torch.cuda.device_count() > 1:
                real_dev = _torch.device(f"cuda:{_rank}")
            else:
                real_dev = _torch.device("cuda:0")
        else:
            real_dev = _torch.device("cpu")
        n_meta_fixed = 0
        for _, module in model.named_modules():
            for attr in ("w13_weight", "w2_weight", "weight"):
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
