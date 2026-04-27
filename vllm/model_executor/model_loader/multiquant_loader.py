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
machine that has the cache directory.

The loader delegates non-quant weight loading to vllm's own
``model.load_weights(iterator)`` pipeline — same path that
``DefaultModelLoader`` uses for HF safetensors. This means TP-slicing,
QKV-merge, gate-up-merge, vocab-parallel-padding and disable_tp
detection all come from vllm's per-Parameter ``weight_loader`` hooks
(``_ColumnvLLMParameter.load_column_parallel_weight``,
``QKVParameter.load_qkv_weight``, ``RowvLLMParameter.load_row_parallel_weight``,
etc.) — no ad-hoc reimplementation here.

The MQ-packed expert tensors (w13/w2 of FusedMoE layers) are filtered
out of the iterator: they come via XFP cache-hit in
``process_weights_after_loading``, not via weight-loader streaming.

Iterator source priority:
  1. ``<cache_dir>/residuals.safetensors`` — small (≤3 GB) local file
     written during the PACK run. Preferred when present.
  2. HF safetensors at the model_path. Used when residuals is missing
     (e.g. cache from an aborted PACK or pre-residuals-fix codebase).
     Adds streaming-quant grouping so per-layer cache-hit triggers
     correctly.
  3. Hard error otherwise — re-pack instructions emitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


class MultiQuantCacheOnlyLoader(BaseModelLoader):
    """Model loader that delegates to vllm's load_weights pipeline.

    Constructs an HF-keyed weight iterator from the cache (or from the
    model's HF safetensors when the cache lacks residuals) and feeds it
    to ``model.load_weights(iterator)``. vllm's per-Parameter
    ``weight_loader`` hooks handle all TP semantics — we never do
    ``narrow``/``copy_`` on tensors ourselves.

    LOAD path — never writes the cache. ``is_pack_loader = False`` keeps
    ``_finalize_multiquant_cache`` from overwriting on-disk residuals
    with per-rank tensors, and we additionally force the cache itself
    into read_only mode as a belt-and-suspenders guard.
    """

    is_pack_loader = False

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        """Nothing to download — cache is always local."""
        pass

    def load_weights(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        from vllm.multiquant.weight_cache import MultiQuantWeightCache

        cache = MultiQuantWeightCache.get_active()
        if cache is None:
            raise RuntimeError(
                "--load-format multiquant requires MULTIQUANT_CACHE_DIR "
                "to be set. Run once with --load-format auto + your "
                "MultiQuant quant flags to populate the cache, then "
                "rerun with --load-format multiquant.")

        # Belt-and-suspenders: any save_residuals call from anywhere
        # downstream becomes a no-op on this loader.
        if not cache.read_only:
            cache.read_only = True
            logger.info(
                "MultiQuant cache: cache-only loader forces read_only=True")

        # PR2: don't stub MoE expert tensors. The stub overwrote
        # FusedMoEMethodBase.create_weights' correctly-sized
        # w13_weight/w2_weight params with size-0 tensors, expecting
        # vllm's stock weight_loader to repopulate them. This worked
        # for w13 (MergedColumnParallel reallocates on merged load) but
        # silently failed for w2 (down_proj is single-shard, expects
        # pre-allocated param.data[expert_id] slot — indexing into
        # size-0 silently no-ops). The fix is to leave the params at
        # their normal allocated size and let vllm load BF16; the
        # XFP-pack PWAL branch then frees BF16 storage explicitly.
        import os as _dbg_os

        # Pick the iterator source.
        residuals_path = cache.cache_dir / "residuals.safetensors"
        if residuals_path.exists():
            logger.info(
                "MultiQuant cache: loading residuals from %s",
                residuals_path)
            weights_iter = self._residuals_iterator(residuals_path, model)
        elif cache.model_path and Path(cache.model_path).exists():
            hf_path = Path(cache.model_path)
            shards = sorted(hf_path.glob("*.safetensors"))
            if not shards:
                raise RuntimeError(
                    f"MultiQuant cache at {cache.cache_dir} has no "
                    f"residuals.safetensors and no HF shards in "
                    f"{hf_path}. Re-pack with MULTIQUANT_QUANT_ONLY=1 "
                    f"to populate residuals.")
            logger.info(
                "MultiQuant cache: residuals missing — streaming "
                "non-MQ-packed weights from HF source %s (%d shards)",
                hf_path, len(shards))
            self._streaming_quant_active = True
            weights_iter = self._hf_source_iterator(shards)
        else:
            raise RuntimeError(
                f"MultiQuant cache at {cache.cache_dir} has neither "
                f"residuals.safetensors nor an accessible HF source "
                f"(model_path={cache.model_path!r}). Re-pack with "
                f"MULTIQUANT_QUANT_ONLY=1.")

        # PR2: don't filter MQ-packed expert names from the iterator.
        # vllm's stock weight_loader populates layer.w13_weight /
        # layer.w2_weight at full BF16 size; the XFP PWAL cache-hit
        # branch then replaces them with packed tensors and frees the
        # BF16 storage explicitly via `p.data = torch.empty(0, ...)`.
        loaded = model.load_weights(weights_iter)
        n_loaded = len(loaded) if loaded is not None else -1
        logger.info(
            "MultiQuant cache-only load: %d weights loaded via vllm "
            "model.load_weights — XFP PWAL will pack/cache-hit per layer",
            n_loaded)

        # PR2 diagnostic — verify vllm populated w13/w2 to full 3D shape.
        # Stays gated on XFP_LOAD_DEBUG to avoid log spam on normal runs.
        if _dbg_os.environ.get("XFP_LOAD_DEBUG", "0") == "1":
            self._dbg_dump_moe_shapes(model, "AFTER_LOAD_WEIGHTS")

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _dbg_dump_moe_shapes(self, model: nn.Module, tag: str) -> None:
        """PR1 diagnostic — log w13/w2 shape per FusedMoE layer + rank."""
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            return
        try:
            import torch.distributed as _td
            r = _td.get_rank() if _td.is_initialized() else -1
        except Exception:
            r = -1
        first_logged = 0
        for name, module in model.named_modules():
            if not isinstance(module, FusedMoE):
                continue
            w13 = getattr(module, "w13_weight", None)
            w2 = getattr(module, "w2_weight", None)
            w13_shape = tuple(w13.data.shape) if w13 is not None else None
            w2_shape = tuple(w2.data.shape) if w2 is not None else None
            w13_dev = str(w13.data.device) if w13 is not None else "?"
            w2_dev = str(w2.data.device) if w2 is not None else "?"
            # Log only first 3 + last 1 layers per rank to bound noise
            if first_logged < 3 or "59" in name or "47" in name:
                logger.info(
                    "[XFP_LOAD_DEBUG %s rank=%d] %s: "
                    "w13.shape=%s dev=%s | w2.shape=%s dev=%s",
                    tag, r, name, w13_shape, w13_dev, w2_shape, w2_dev,
                )
                first_logged += 1

    def _residuals_iterator(
        self, residuals_path: Path, model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield (name, tensor) from residuals.safetensors.

        Buffers (rotary inv_freq, e_score_correction_bias) live under
        the ``__buffer__/`` prefix; ``model.load_weights`` doesn't
        understand them (AutoWeightsLoader's child-iteration only sees
        named_parameters + named_children), so we copy them in-place
        here and don't yield them downstream.
        """
        from safetensors import safe_open
        name_to_buffer = dict(model.named_buffers())
        with safe_open(str(residuals_path), framework="pt",
                       device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith("__buffer__/"):
                    buf_name = key[len("__buffer__/"):]
                    buf = name_to_buffer.get(buf_name)
                    if buf is not None:
                        buf.data.copy_(tensor.to(
                            dtype=buf.dtype, device=buf.device))
                    continue
                yield key, tensor

    def _hf_source_iterator(
        self, shards: list[Path],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield (name, tensor) from the model's HF safetensors shards.

        Uses vllm's own iterator builders so streaming-quant per-layer
        grouping works identically to DefaultModelLoader.
        """
        from vllm.model_executor.model_loader.weight_utils import (
            layer_grouped_safetensors_weights_iterator,
            safetensors_weights_iterator,
        )
        shard_paths = [str(p) for p in shards]
        strategy = self.load_config.safetensors_load_strategy
        if getattr(self, "_streaming_quant_active", False):
            yield from layer_grouped_safetensors_weights_iterator(
                shard_paths,
                self.load_config.use_tqdm_on_load,
                strategy,
            )
        else:
            yield from safetensors_weights_iterator(
                shard_paths,
                self.load_config.use_tqdm_on_load,
                strategy,
            )

