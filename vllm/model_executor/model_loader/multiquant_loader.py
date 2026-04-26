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


# Names that look like a Linear weight but are intentionally MQ-packed
# (replaced wholesale by the cache-hit path in
# ``process_weights_after_loading``). Filtering them out of the iterator
# prevents vllm's streaming-quant wrapper from materializing BF16
# storage just to immediately discard it.
_MQ_PACKED_LEAF_NAMES = ("gate_up_proj", "down_proj")


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

        # Stub MoE expert tensors so the cache-hit path in
        # process_weights_after_loading sees a real cuda device on
        # ``layer.w13_weight`` instead of a meta tensor.
        n_stubbed = self._stub_moe_expert_tensors(model)

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

        # Filter MQ-packed expert tensors so they don't churn through
        # streaming-quant. Filter is idempotent for residuals (which
        # never contains MQ-packed keys) — only effective for HF source.
        weights_iter = self._filter_mq_packed(weights_iter, cache.cache_dir)

        # Delegate everything else: vllm's stacked_params_mapping,
        # expert_params_mapping, weight_loader hooks handle TP semantics.
        loaded = model.load_weights(weights_iter)
        n_loaded = len(loaded) if loaded is not None else -1
        logger.info(
            "MultiQuant cache-only load: %d weights loaded via vllm's "
            "model.load_weights, %d MoE expert params stubbed — quant "
            "layers will fill from cache in process_weights_after_loading",
            n_loaded, n_stubbed)

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _stub_moe_expert_tensors(self, model: nn.Module) -> int:
        """Replace meta-device w13/w2 of FusedMoE with size-0 cuda stubs.

        The XFP MoE cache-hit path in ``online_moe.process_weights_after_loading``
        reads ``layer.w13_weight.device`` to pick where to allocate the
        packed tensors and calls ``torch.empty(0, device=p.data.device, ...)``
        to free the BF16 storage — neither works on a meta tensor.

        We deliberately keep size=0 here (no BF16 materialization!)
        because the cache-hit code reassigns ``.data`` to the packed
        tensor immediately afterward.
        """
        if torch.cuda.is_available():
            tp_rank = -1
            try:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                )
                tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                try:
                    import torch.distributed as _td
                    if _td.is_initialized():
                        tp_rank = _td.get_rank()
                except Exception:
                    pass
            if tp_rank < 0:
                tp_rank = 0
            cur = torch.cuda.current_device()
            count = torch.cuda.device_count()
            # Pick: if CUDA_VISIBLE_DEVICES restricts to 1 GPU, our
            # cuda:0 is already the right GPU. Otherwise use cuda:tp_rank.
            real_dev = (torch.device(f"cuda:{tp_rank}") if count > 1
                        else torch.device("cuda:0"))
            logger.info(
                "[xfp_tp] MoE-stub device=%s (tp_rank=%d, "
                "current_device=%d, device_count=%d)",
                real_dev, tp_rank, cur, count)
        else:
            real_dev = torch.device("cpu")

        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            FusedMoE = None  # type: ignore

        n = 0
        for _, module in model.named_modules():
            if FusedMoE is None or not isinstance(module, FusedMoE):
                continue
            for attr in ("w13_weight", "w2_weight"):
                p = getattr(module, attr, None)
                if p is None:
                    continue
                if p.data.device.type == "meta":
                    p.data = torch.empty(
                        0, dtype=p.data.dtype, device=real_dev)
                    n += 1
        return n

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

    def _filter_mq_packed(
        self,
        weights_iter: Generator[tuple[str, torch.Tensor], None, None],
        cache_dir: Path,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Drop MoE-expert keys that are filled by XFP cache-hit.

        Each FusedMoE layer with cached XFP shards has a sibling cache
        directory like ``<layer>.mlp.experts/``. For those layers we
        skip the BF16 expert-projection keys
        (``...experts.gate_up_proj.weight``,
        ``...experts.down_proj.weight``) — they would otherwise force
        the streaming-quant wrapper into a BF16 stub→materialize
        round-trip per expert (~5 GB transient per layer) before being
        replaced anyway by the cache-hit path.

        The check is conservative and only fires for layers that
        actually have a cache directory; layers without cache fall
        through to vllm's normal loader (which would expose any
        coverage gap in the PACK pipeline).
        """
        moe_prefixes = sorted({
            p.name for p in cache_dir.glob("*.experts")
            if p.is_dir()
        })
        if not moe_prefixes:
            yield from weights_iter
            return
        for name, tensor in weights_iter:
            if any(name.startswith(prefix + ".")
                   for prefix in moe_prefixes):
                if any(leaf in name for leaf in _MQ_PACKED_LEAF_NAMES):
                    continue
            yield name, tensor
