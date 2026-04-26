# SPDX-License-Identifier: Apache-2.0
"""Generic disk cache for MultiQuant quantized weights.

Any quant method in the MultiQuant ecosystem (XFP, TurboQuant, RotorQuant,
…) can persist its per-layer artefacts here and skip the expensive
quant pipeline on subsequent starts. The cache is keyed by a hash that
includes the MultiQuantPolicyRegistry state — so changing *what* is
quantized *how* automatically invalidates the cache.

Design:
  - This module owns the plumbing only: paths, safetensors I/O, manifest,
    hash over policy + source code, stats, singleton.
  - Per-method serialization (which layer attributes to save and load)
    lives in the method's own module (e.g.
    ``vllm.multiquant.xfp.xfp_weight_cache``). Those modules are thin
    adapters over ``MultiQuantWeightCache.save`` / ``.load``.

On-disk layout::

    $MULTIQUANT_CACHE_DIR/<model_basename>/<hash16>/
        manifest.json
        <layer_prefix>.safetensors      (metadata['method'] == "xfp_linear", ...)
        …

Enable via env ``MULTIQUANT_CACHE_DIR`` (``XFP_CACHE_DIR`` kept as an
alias for backwards compat). Set ``MULTIQUANT_CACHE_READ_ONLY=1`` for
eval runs that must never write.

Every fallback path emits a log line — silent misses defeat the purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# Constants that affect the bytes written to cache. Keep in sync with the
# defaults in the quant methods (xfp_pack etc.). Anything listed here is
# part of the cache key.
_DEFAULT_OUTLIER_SIGMA = 4.0
_DEFAULT_OUTLIER_MAX_FRACTION = 0.02
_LLOYD_ITERS_LINEAR = 20
# Schema 2: per-shard ``_manifest.json`` carries ``format_version`` plus
# a ``tensor_meta`` dict that names each cached tensor's tp_role +
# slicing axes. The XFP packed tensors are stored pre-repack (2D for
# linear, 3D for MoE) so they slice with simple ``narrow()`` calls; the
# warp-interleaved repack happens on-device at load time. Schema 1
# baked repack into storage and had no per-tensor TP metadata — caches
# from schema 1 are not load-compatible and must be regenerated.
#
# Schema 3: identical wire format to schema 2, but adds the PACK/LOAD
# separation in base_loader: cache-only LOAD never invokes
# save_residuals, so the on-disk full-shape tensors stay intact across
# arbitrarily many TP>1 LOADs. Caches packed under schema 2 (where
# save_residuals fired on every LOAD) accumulated per-rank shapes for
# vision QKV bias, GatedDeltaNet conv1d.weight, and the MoE router
# gate.weight — those are unrecoverable and rejected here so the
# operator is forced into a clean PACK with the new code.
_MANIFEST_SCHEMA_VERSION = 3


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _file_sha(path: str | Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        logger.warning(
            "MultiQuant cache: cannot hash %s (%s) — treating as dirty",
            path, e,
        )
        return "unreadable"


def _tensor_cpu_contig(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "cpu":
        t = t.detach().cpu()
    return t.contiguous()


def _guess_device(model) -> torch.device:
    """Pick a sensible device for materializing a meta-Parameter from
    residuals. Looks for any already-materialized param on the model and
    copies its device; falls back to current CUDA device (per-worker in
    TP), then cpu.

    [xfp_tp] Must NOT hard-code cuda:0 — a TP>1 serve has each rank using
    its own physical GPU and hard-coding cuda:0 would make rank N>0 write
    residuals to rank 0's GPU, doubling its memory footprint.
    """
    for p in model.parameters():
        if p is not None and p.data.numel() > 0 and p.data.device.type != "meta":
            return p.data.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class MultiQuantWeightCache:
    """Per-process disk cache. Singleton via ``get_active()``."""

    _active: Optional["MultiQuantWeightCache"] = None

    def __init__(
        self,
        cache_root: str,
        model_basename: str,
        cache_key: str,
        read_only: bool = False,
        model_path: str | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.model_basename = model_basename
        self.cache_key = cache_key
        self.read_only = read_only
        self.cache_dir = self.cache_root / model_basename / cache_key
        # Source HF safetensors path — fallback when residuals.safetensors
        # is missing some entries (e.g. a v1-cache packed before vision
        # tower weights were captured by save_residuals). Empty when the
        # caller didn't pass it.
        self.model_path = model_path
        # Stats for the end-of-load summary.
        self._hits: dict[str, int] = {}     # by method name
        self._misses: dict[str, int] = {}
        self._saves = 0
        self._save_fail = 0
        self._load_fail = 0
        self._t_loaded_s = 0.0
        self._t_saved_s = 0.0

    # ─── Singleton ───────────────────────────────────────────────────

    @classmethod
    def get_active(cls) -> Optional["MultiQuantWeightCache"]:
        return cls._active

    @classmethod
    def set_active(cls, cache: Optional["MultiQuantWeightCache"]) -> None:
        cls._active = cache

    @classmethod
    def from_env(
        cls,
        model_path: str,
        registry,
        hf_config,
        tp_size: int = 1,
        ep_size: int = 1,
    ) -> Optional["MultiQuantWeightCache"]:
        """Construct from env. Returns None when cache disabled."""
        # Primary var + XFP-era alias.
        cache_root = (
            os.environ.get("MULTIQUANT_CACHE_DIR", "").strip()
            or os.environ.get("XFP_CACHE_DIR", "").strip()
        )
        if not cache_root:
            logger.info(
                "MultiQuant cache: disabled "
                "(set MULTIQUANT_CACHE_DIR to enable)"
            )
            return None
        read_only = (
            _env_truthy("MULTIQUANT_CACHE_READ_ONLY")
            or _env_truthy("XFP_CACHE_READ_ONLY")
        )
        model_basename = os.path.basename(
            os.path.normpath(model_path)
        ) or "model"
        cache_key = cls.compute_cache_key(
            model_path, registry, hf_config, tp_size, ep_size,
        )
        inst = cls(cache_root, model_basename, cache_key, read_only,
                   model_path=model_path)
        try:
            inst.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "MultiQuant cache: cannot mkdir %s (%s) — disabled",
                inst.cache_dir, e,
            )
            return None
        logger.info(
            "MultiQuant cache: root=%s key=%s%s (dir=%s)",
            cache_root, cache_key,
            " (read-only)" if read_only else "",
            inst.cache_dir,
        )
        return inst

    # ─── Hash / key ───────────────────────────────────────────────────

    @staticmethod
    def compute_cache_key(
        model_path: str,
        registry,
        hf_config,
        tp_size: int = 1,
        ep_size: int = 1,
    ) -> str:
        """Hash over everything that changes the quantized bytes."""
        h = hashlib.sha256()

        # Model identity
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_path):
            h.update(b"config_json|")
            h.update(_file_sha(cfg_path).encode())
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            h.update(b"|weights_index|")
            h.update(_file_sha(idx_path).encode())
        else:
            try:
                entries = sorted(
                    f"{f.name}:{f.stat().st_size}"
                    for f in Path(model_path).iterdir()
                    if f.is_file() and f.name.endswith(".safetensors")
                )
                h.update(b"|shard_listing|")
                h.update(json.dumps(entries).encode())
            except OSError as e:
                logger.warning(
                    "MultiQuant cache: cannot list shards in %s (%s)",
                    model_path, e,
                )

        # MultiQuant policy (WAS wird wie quantisiert)
        h.update(b"|policy|")
        try:
            pol = registry.to_dict() if registry is not None else {}
        except Exception as e:
            logger.warning(
                "MultiQuant cache: registry.to_dict failed (%s)", e
            )
            pol = {}
        h.update(json.dumps(pol, sort_keys=True).encode())

        # Quant pipeline source code — catches semantic changes to packers.
        for mod_name in (
            "vllm.multiquant.policy",
            "vllm.multiquant.xfp.xfp_pack",
        ):
            try:
                mod = __import__(mod_name, fromlist=["*"])
                path = getattr(mod, "__file__", None)
                if path:
                    h.update(f"|{mod_name}_sha|".encode())
                    h.update(_file_sha(path).encode())
            except ImportError:
                # Module absent → skip in hash; no effect on cache key
                continue

        # Quant-tuning knobs not in the registry (env-configurable).
        moe_lloyd = int(os.environ.get("XFP_MOE_LLOYD_ITERS", "5"))
        auto_min_cos = float(os.environ.get("XFP_MIN_COS", "0.98"))
        moe_sample = int(os.environ.get("XFP_MOE_SAMPLE_EXPERTS", "4"))
        # TP/EP are intentionally NOT in the cache key by default: the
        # packed bytes (codebook + indices) are per-layer complete
        # [E, N, K] tensors; slicing by rank happens at cache-load time.
        # That means one pack serves any TP/EP config. Set
        # MULTIQUANT_CACHE_TP_SPECIFIC=1 to opt back into TP-specific
        # shards if a quant method stores pre-sliced data.
        tp_in_hash = (
            f"|tp={tp_size}|ep={ep_size}"
            if _env_truthy("MULTIQUANT_CACHE_TP_SPECIFIC")
            else ""
        )
        h.update(
            f"|sigma={_DEFAULT_OUTLIER_SIGMA}"
            f"|maxf={_DEFAULT_OUTLIER_MAX_FRACTION}"
            f"|lloyd_lin={_LLOYD_ITERS_LINEAR}"
            f"|moe_lloyd={moe_lloyd}"
            f"|min_cos={auto_min_cos}"
            f"|moe_sample={moe_sample}"
            f"{tp_in_hash}"
            f"|schema={_MANIFEST_SCHEMA_VERSION}".encode()
        )
        return h.hexdigest()[:16]

    # ─── Path resolution ──────────────────────────────────────────────

    def layer_path(self, layer_prefix: str) -> Path:
        """Return the per-layer DIRECTORY. Each tensor lives in its own
        .safetensors file inside this dir — keeps save-transient memory
        bounded to one tensor at a time instead of N simultaneously.
        Critical on GB10 UMA where .cpu() copies share physical DRAM with
        the CUDA originals (doubles footprint during save)."""
        safe = layer_prefix.replace("/", "__").replace(":", "_")
        if not safe:
            safe = "__root__"
        return self.cache_dir / safe

    # ─── Generic save / load (used by per-method adapters) ───────────

    def save(
        self,
        layer_prefix: str,
        method: str,
        tensors: dict[str, torch.Tensor],
        metadata: Optional[dict[str, str]] = None,
        tensor_meta: Optional[dict[str, dict]] = None,
    ) -> bool:
        """Persist a per-layer artefact bundle. Returns True on success.

        Writes each tensor to its own .safetensors file inside a directory
        per layer. This bounds the save-transient memory to ONE tensor at
        a time — critical on GB10 unified memory, where the .cpu() copy
        needed for safetensors-write shares physical DRAM with the CUDA
        original. Writing all N tensors at once (the old layout) peaked at
        2× model size during the final save calls of a MoE layer.
        """
        if self.read_only or not layer_prefix:
            return False
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning(
                "MultiQuant cache: safetensors not available — "
                "cache save disabled",
            )
            return False
        import gc as _gc

        layer_dir = self.layer_path(layer_prefix)
        tmp_dir = layer_dir.with_name(layer_dir.name + ".tmp")
        # Clean up any stale tmp from prior crash
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, str] = {"method": method, "cache_key": self.cache_key}
        if metadata:
            for k, v in metadata.items():
                # Safetensors metadata values MUST be strings.
                meta[str(k)] = str(v)

        t0 = time.perf_counter()
        try:
            # Stream each tensor individually — one CPU copy alive at a time.
            # The old code did `{k: v.cpu() for k,v in tensors.items()}` which
            # held N copies simultaneously.
            for key, tensor in tensors.items():
                safe_key = key.replace("/", "__").replace(":", "_")
                tensor_cpu = _tensor_cpu_contig(tensor)
                save_file(
                    {key: tensor_cpu},
                    str(tmp_dir / f"{safe_key}.safetensors"),
                    metadata=meta,
                )
                # Drop the CPU copy before the next allocation.
                del tensor_cpu
                _gc.collect()
            # Manifest lists the keys so load() doesn't need to scandir.
            # Schema 2: also embed ``tensor_meta`` (per-tensor tp_role +
            # slicing axes) so the load path can reconstruct per-rank
            # shards from a TP-agnostic cache.
            (tmp_dir / "_manifest.json").write_text(
                json.dumps({
                    "format_version": _MANIFEST_SCHEMA_VERSION,
                    "method": method,
                    "cache_key": self.cache_key,
                    "keys": list(tensors.keys()),
                    "metadata": metadata or {},
                    "tensor_meta": tensor_meta or {},
                }, sort_keys=True)
            )
            # Atomic swap: rename tmp_dir -> layer_dir
            if layer_dir.exists():
                import shutil
                shutil.rmtree(layer_dir)
            tmp_dir.rename(layer_dir)
            self._saves += 1
            self._t_saved_s += time.perf_counter() - t0
            return True
        except (OSError, RuntimeError) as e:
            self._save_fail += 1
            logger.warning(
                "MultiQuant cache: save failed for %s/%s (%s)",
                method, layer_prefix, e,
            )
            # Best-effort cleanup of partial tmp
            try:
                import shutil
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            return False

    def load(
        self,
        layer_prefix: str,
        expected_method: str,
        device: torch.device,
    ) -> Optional[tuple[dict[str, torch.Tensor], dict[str, str], dict[str, dict]]]:
        """Return (tensors_on_device, metadata, tensor_meta) or None on miss/mismatch.

        Reads each per-key .safetensors file inside the layer directory
        and moves its tensor to device, freeing the CPU copy before the
        next read. Peak transient = one tensor at a time (same bound as
        the save path).

        Schema 2: rejects any shard whose ``_manifest.json`` does not
        carry ``format_version: 2`` — schema 1 caches stored a baked
        warp-interleaved repack that's not slicable for TP > 1.
        """
        if not layer_prefix:
            return None
        layer_dir = self.layer_path(layer_prefix)
        if not layer_dir.exists() or not layer_dir.is_dir():
            self._misses[expected_method] = \
                self._misses.get(expected_method, 0) + 1
            return None
        manifest_path = layer_dir / "_manifest.json"
        if not manifest_path.exists():
            # Treat as miss — incomplete or stale layout.
            self._misses[expected_method] = \
                self._misses.get(expected_method, 0) + 1
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "MultiQuant cache: manifest unreadable for %s (%s)",
                layer_prefix, e,
            )
            self._load_fail += 1
            return None
        on_disk_method = manifest.get("method")
        if on_disk_method != expected_method:
            logger.warning(
                "MultiQuant cache: %s method=%r != expected %r → miss",
                layer_dir, on_disk_method, expected_method,
            )
            self._load_fail += 1
            return None
        on_disk_format = manifest.get("format_version", 1)
        if on_disk_format != _MANIFEST_SCHEMA_VERSION:
            logger.warning(
                "MultiQuant cache: %s format_version=%r != expected %d "
                "(schema 1 had baked warp-interleaved repack and no tp_role "
                "metadata; re-pack required) → miss",
                layer_dir, on_disk_format, _MANIFEST_SCHEMA_VERSION,
            )
            self._load_fail += 1
            return None
        try:
            from safetensors import safe_open
            import gc as _gc
            t0 = time.perf_counter()
            tensors: dict[str, torch.Tensor] = {}
            meta_out: dict[str, str] = {"method": expected_method}
            for key in manifest.get("keys", []):
                safe_key = key.replace("/", "__").replace(":", "_")
                fpath = layer_dir / f"{safe_key}.safetensors"
                if not fpath.exists():
                    logger.warning(
                        "MultiQuant cache: missing %s for %s — miss",
                        fpath.name, layer_prefix,
                    )
                    self._load_fail += 1
                    return None
                with safe_open(str(fpath), framework="pt") as f:
                    # Move to device immediately; CPU temp frees at close
                    tensors[key] = f.get_tensor(key).to(device)
                    fmeta = f.metadata() or {}
                    # Merge string metadata; method fields stay consistent
                    for mk, mv in fmeta.items():
                        if mk not in meta_out:
                            meta_out[mk] = mv
                _gc.collect()
            # Promote plain-dict "metadata" block into the output meta
            for mk, mv in (manifest.get("metadata") or {}).items():
                meta_out[str(mk)] = str(mv)
            tensor_meta_out: dict[str, dict] = manifest.get("tensor_meta") or {}
            self._hits[expected_method] = \
                self._hits.get(expected_method, 0) + 1
            self._t_loaded_s += time.perf_counter() - t0
            return tensors, meta_out, tensor_meta_out
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            self._load_fail += 1
            logger.warning(
                "MultiQuant cache: load failed for %s/%s (%s)",
                expected_method, layer_prefix, e,
            )
            return None

    # ─── Manifest ─────────────────────────────────────────────────────

    def write_manifest(self, registry, inventory: list[str]) -> None:
        if self.read_only:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Capture all XFP_* and MULTIQUANT_* env vars so offline analysis
        # (tools/pack_report.py) can diff hyperparameter sweeps across
        # shards without having to parse stdout logs.
        env_snapshot = {
            k: v for k, v in os.environ.items()
            if k.startswith(("XFP_", "MULTIQUANT_"))
        }
        mf = {
            "schema": _MANIFEST_SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "model_basename": self.model_basename,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "policy": registry.to_dict() if registry is not None else {},
            "inventory": sorted(inventory),
            "vllm_version": _best_effort_vllm_version(),
            "env": env_snapshot,
        }
        path = self.cache_dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(mf, indent=2, sort_keys=True))
            tmp.replace(path)
        except OSError as e:
            logger.warning(
                "MultiQuant cache: manifest write failed (%s): %s",
                path, e,
            )

    def verify_manifest(self) -> bool:
        path = self.cache_dir / "manifest.json"
        if not path.exists():
            logger.info(
                "MultiQuant cache: no manifest at %s — cold start, "
                "cache will populate",
                path,
            )
            return False
        try:
            mf = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "MultiQuant cache: manifest unreadable (%s): %s → re-quant",
                path, e,
            )
            return False
        if mf.get("schema") != _MANIFEST_SCHEMA_VERSION:
            logger.warning(
                "MultiQuant cache: manifest schema=%s (want %d) → re-quant",
                mf.get("schema"), _MANIFEST_SCHEMA_VERSION,
            )
            return False
        if mf.get("cache_key") != self.cache_key:
            logger.warning(
                "MultiQuant cache: manifest cache_key=%s but "
                "computed=%s → re-quant",
                mf.get("cache_key"), self.cache_key,
            )
            return False
        return True

    # ─── Residuals (for cache-only load) ─────────────────────────────
    #
    # After MultiQuant-packed layers have replaced their bf16 storage with
    # packed/codebook/outlier tensors (and shrunk the bf16 Parameter to
    # numel=0), whatever non-zero Parameter data remains on the model are
    # the "residuals": embed_tokens, RMSNorms, e_score_correction_bias,
    # non-fp8 lm_heads. For 397B these are ~2–3 GB total.
    #
    # Dumping them to <cache_dir>/residuals.safetensors lets a later run
    # reconstruct the full model from just (cache + residuals) — no bf16
    # source on disk required. That's what enables TP=2 XFP on a 397B
    # model that won't fit on PGX's 916 GB disk as bf16 source.

    def save_residuals(self, model) -> bool:
        """Dump all non-zero-numel parameters to residuals.safetensors.

        Call once at end of load_model after all
        process_weights_after_loading ran — at that point, MultiQuant
        layers have zeroed their bf16 params and only non-quant params
        (embed, norms, etc) still carry data.
        """
        if self.read_only:
            return False
        # Hard guard: residuals are inherently a TP=1 (full-shape) artifact.
        # If we're called at TP>1, the model.named_parameters() iteration
        # would write per-rank shards over the full tensors and corrupt
        # the cache for any future LOAD. Bail out — the LOAD path should
        # never reach here (it sets read_only above), but defend against
        # accidental writes from any other code path.
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_world_size as _get_tp_world,
            )
            tp_world_now = _get_tp_world()
        except Exception:
            tp_world_now = 1
        if tp_world_now > 1:
            logger.warning(
                "MultiQuant cache: refusing save_residuals at TP=%d — "
                "params are per-rank shaped, would corrupt the on-disk "
                "TP=1 cache. Run a quant-only PACK at TP=1 first.",
                tp_world_now)
            return False
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning(
                "MultiQuant cache: safetensors unavailable — residuals skipped",
            )
            return False

        # MultiQuant quant methods store their per-layer packed/codebook/
        # outlier tensors as regular nn.Parameters on the layer (so vllm's
        # state_dict sees them). Those live in the per-layer cache shards
        # already, so they MUST be excluded from the residuals bundle or we
        # double-store the entire model (~70 GB bf16 → 19 GB residuals +
        # 17 GB cache for 35B before this filter).
        # Param names end with e.g. ".linear_attn.in_proj_ba.xfp_packed"
        # (Linear path) or ".mlp.experts.w13_xfp_packed" (MoE path). The
        # leading "." matters — plain "xfp_packed" would also match a
        # hypothetical ".foo_xfp_packed" which we'd want to treat as
        # residual. Explicit prefixes keep this conservative.
        _MQ_PARAM_SUFFIXES = (
            ".xfp_packed", ".xfp_codebook",
            ".xfp_outlier_row", ".xfp_outlier_col", ".xfp_outlier_val",
            ".w13_xfp_packed", ".w13_xfp_codebook",
            ".w2_xfp_packed", ".w2_xfp_codebook",
        )

        def _is_mq_packed_param(name: str) -> bool:
            return any(name.endswith(s) for s in _MQ_PARAM_SUFFIXES)

        from vllm.multiquant._tp_slicer import infer_tp_role_from_param
        tensors: dict[str, torch.Tensor] = {}
        # tp_meta: per-param tp_role + axis hints, written next to
        # residuals.safetensors so the load-time slicer can recover the
        # rank-specific shard at TP > 1.
        tp_meta: dict[str, dict] = {}
        seen_ptrs: set[int] = set()
        skipped_mq = 0
        skipped_empty: list[str] = []
        for name, p in model.named_parameters():
            if p is None:
                continue
            if p.data.numel() == 0:
                skipped_empty.append(name)
                continue
            if _is_mq_packed_param(name):
                skipped_mq += 1
                continue
            ptr = p.data.data_ptr() if p.data.is_contiguous() else id(p.data)
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors[name] = _tensor_cpu_contig(p.data)
            tp_meta[name] = infer_tp_role_from_param(name, p)
        if skipped_empty:
            logger.warning(
                "MultiQuant cache: save_residuals skipped %d empty params "
                "(first 5: %s) — these will be missing from the cache and "
                "the cache-only load path will leave them stubbed.",
                len(skipped_empty), skipped_empty[:5],
            )

        # Also capture registered buffers (e.g. rotary inv_freq,
        # e_score_correction_bias if stored as buffer not param)
        for name, b in model.named_buffers():
            if b is None or b.numel() == 0:
                continue
            ptr = b.data_ptr() if b.is_contiguous() else id(b)
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            buf_name = f"__buffer__/{name}"
            tensors[buf_name] = _tensor_cpu_contig(b)
            # Buffers are usually replicated (e.g. inv_freq, layer_idx).
            tp_meta[buf_name] = {"tp_role": "replicated"}

        if not tensors:
            logger.info("MultiQuant cache: no residuals to dump (empty)")
            return True

        path = self.cache_dir / "residuals.safetensors"
        tp_meta_path = self.cache_dir / "residuals_tp_meta.json"
        tmp = path.with_suffix(".safetensors.tmp")
        tmp_meta = tp_meta_path.with_suffix(".json.tmp")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                tensors,
                str(tmp),
                metadata={
                    "cache_key": self.cache_key,
                    "schema": str(_MANIFEST_SCHEMA_VERSION),
                    "count": str(len(tensors)),
                },
            )
            tmp.replace(path)
            tmp_meta.write_text(json.dumps(tp_meta, indent=2, sort_keys=True))
            tmp_meta.replace(tp_meta_path)
            total_bytes = sum(t.numel() * t.element_size()
                              for t in tensors.values())
            logger.info(
                "MultiQuant cache: wrote residuals → %s "
                "(%d tensors, %.2f GB, %d mq-packed params excluded)",
                path, len(tensors), total_bytes / (1024 ** 3), skipped_mq,
            )
            return True
        except (OSError, RuntimeError) as e:
            logger.warning(
                "MultiQuant cache: residuals dump failed (%s): %s", path, e,
            )
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

    @staticmethod
    def _tp_slice_if_needed(
        tensor: torch.Tensor, target_shape: tuple,
    ) -> torch.Tensor:
        """Slice cached tensor along its TP-split dim to match target.

        The cache is TP-agnostic (pack can happen at TP=1, serve at any TP).
        Residual params whose owning modules are ColumnParallel- /
        RowParallelLinear / VocabParallelEmbedding will have a per-rank
        target shape that is smaller than the cached full-size tensor.

        Strategy: find the first dim where sizes differ, assert the ratio
        equals TP world size, narrow along that dim. If no mismatch, return
        tensor unchanged.
        """
        if tuple(tensor.shape) == tuple(target_shape):
            return tensor
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            tp_world = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            return tensor
        if tp_world == 1:
            return tensor
        for dim, (t_sz, c_sz) in enumerate(
                zip(target_shape, tensor.shape)):
            if t_sz == c_sz:
                continue
            if c_sz != t_sz * tp_world:
                # Not a TP-slice mismatch — let the caller raise on
                # copy_() so the operator sees the original error.
                return tensor
            return tensor.narrow(dim, tp_rank * t_sz, t_sz).contiguous()
        return tensor

    def load_residuals(self, model) -> bool:
        """Load residuals.safetensors into the model.

        Expects `save_residuals()` was called on a previous run. Copies
        tensors in-place into matching Parameter/buffer tensors. Raises
        if the file is missing and returns False for soft failures.
        """
        path = self.cache_dir / "residuals.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"MultiQuant cache-only load: residuals missing at {path}. "
                f"Run once with bf16 source to populate the cache."
            )

        try:
            from safetensors import safe_open
        except ImportError:
            logger.error("MultiQuant cache: safetensors unavailable")
            return False

        # Load the per-param tp_role sidecar for v2 caches.
        from vllm.multiquant._tp_slicer import slice_for_tp
        # Reuse vllm's specialized weight_loader semantics for the
        # Linear-base Parameter subclasses (Column / Row / QKV / Merged) —
        # they implement Q-K-V-aware / merged-shard-offset slicing that
        # plain narrow on output_dim cannot replicate. VocabParallelEmbedding
        # has a custom weight_loader with tighter pre-conditions (expects
        # unpadded vocab size as input) that doesn't match our padded
        # cached tensors, so we keep it on the manifest slicer path.
        try:
            from vllm.model_executor.parameter import (
                _ColumnvLLMParameter,
                RowvLLMParameter,
            )
            _LINEAR_PARAM_CLASSES = (_ColumnvLLMParameter, RowvLLMParameter)
        except ImportError:
            _LINEAR_PARAM_CLASSES = ()
        tp_meta_path = self.cache_dir / "residuals_tp_meta.json"
        if tp_meta_path.exists():
            try:
                tp_meta = json.loads(tp_meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                tp_meta = {}
        else:
            tp_meta = {}

        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            tp_world = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            tp_world, tp_rank = 1, 0

        name_to_param = {n: p for n, p in model.named_parameters()}
        name_to_buffer = {n: b for n, b in model.named_buffers()}
        loaded = 0
        skipped: list[str] = []
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith("__buffer__/"):
                    target_name = key[len("__buffer__/"):]
                    target = name_to_buffer.get(target_name)
                else:
                    target = name_to_param.get(key)
                    target_name = key
                if target is None:
                    skipped.append(target_name)
                    continue
                # Step 1: prefer vllm's specialized load methods on
                # _ColumnvLLMParameter / RowvLLMParameter — they handle
                # qkv-aware, merged-shard-offset, and per-rank narrow
                # correctly. Skip for buffers, non-linear params, and
                # disable_tp params (vision tower under mm-encoder-tp-mode=
                # data: param.tp_size==1 means "this param is replicated
                # across TP ranks", so the loader's tp_rank * shard_size
                # narrow goes out of bounds on rank>0).
                used_vllm_loader = False
                if (
                    tp_world > 1
                    and _LINEAR_PARAM_CLASSES
                    and isinstance(target, _LINEAR_PARAM_CLASSES)
                    and not key.startswith("__buffer__/")
                    and getattr(target, "tp_size", None) != 1
                ):
                    try:
                        # Materialize meta-device target if needed.
                        # initialize_streaming_quantload shrinks Linear
                        # params to (0,) (1-D size-0 stub) and stashes the
                        # original per-rank shape in ``_streaming_shape``.
                        # Use that as the source of truth — ``target.shape``
                        # at this point is (0,) and would produce a 1-D
                        # stub that crashes the GEMM later.
                        target_shape = getattr(target, "_streaming_shape",
                                               None) or tuple(target.shape)
                        target_dtype = getattr(target, "_streaming_dtype",
                                               None) or target.data.dtype
                        if (target.device.type == "meta"
                                or target.numel() == 0):
                            target.data = torch.empty(
                                target_shape,
                                dtype=target_dtype,
                                device=_guess_device(model),
                            )
                        # _ColumnvLLMParameter has load_column_parallel_weight
                        # (also covers QKV and Merged via auto-shard). Row
                        # has load_row_parallel_weight.
                        from vllm.model_executor.parameter import (
                            _ColumnvLLMParameter as _Col,
                            RowvLLMParameter as _Row,
                        )
                        cpu_tensor = tensor.to(
                            dtype=target.data.dtype, device=target.device)
                        if isinstance(target, _Col):
                            target.load_column_parallel_weight(cpu_tensor)
                        else:
                            target.load_row_parallel_weight(cpu_tensor)
                        loaded += 1
                        used_vllm_loader = True
                        continue
                    except Exception as e:
                        logger.warning(
                            "residuals: vllm parameter loader for %s "
                            "raised (%s) — falling back to manifest slice",
                            key, e)

                # Step 2: manifest tp_role slicer (for buffers, embed,
                # GatedDeltaNet weights, and any param the vllm loader
                # rejected).
                entry = tp_meta.get(key)
                # disable_tp layers (vision tower under
                # mm-encoder-tp-mode=data) have param.tp_size==1 even at
                # TP > 1. The manifest may still say "column" because it
                # was packed by an older infer_tp_role_from_param without
                # the tp_size check — treat those as replicated.
                if (entry is not None and entry.get("tp_role") == "column"
                        and getattr(target, "tp_size", None) == 1):
                    entry = {"tp_role": "replicated"}
                if entry is not None and tp_world > 1:
                    try:
                        tensor = slice_for_tp(tensor, entry, tp_rank, tp_world)
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "residuals tp_slice failed for %s (%s) — "
                            "falling back to shape-based slice",
                            key, e)
                # Step 3: shape-based fallback (legacy + un-annotated).
                # Prefer ``_streaming_shape`` over ``target.shape`` — the
                # latter is (0,) for stubs from initialize_streaming_quantload.
                target_shape = getattr(target, "_streaming_shape", None)
                if target_shape is None and target.numel() > 0:
                    target_shape = tuple(target.data.shape)
                if (target_shape is not None
                        and tuple(tensor.shape) != tuple(target_shape)):
                    tensor = self._tp_slice_if_needed(tensor, target_shape)
                # Materialize if meta, then copy
                if target.device.type == "meta" or target.numel() == 0:
                    if isinstance(target, torch.nn.Parameter):
                        target.data = tensor.to(
                            dtype=target.data.dtype,
                            device=_guess_device(model),
                        ).contiguous()
                    else:
                        target.data = tensor.to(device=target.device).contiguous()
                else:
                    try:
                        target.data.copy_(tensor.to(
                            dtype=target.data.dtype, device=target.device,
                        ))
                    except RuntimeError as _copy_err:
                        # Cached tensor shape doesn't match the layer's
                        # constructed shape (cache packed before the
                        # disable_tp / save_residuals fixes landed).
                        # Don't paper over with the wrong-shape tensor
                        # — leave the storage at its original allocation
                        # and queue this param for the source-safetensors
                        # fallback below.
                        logger.warning(
                            "residuals copy mismatch for %s "
                            "(target=%s tensor=%s): %s — "
                            "deferring to source safetensors",
                            key, tuple(target.data.shape),
                            tuple(tensor.shape), _copy_err,
                        )
                        if not hasattr(self, "_fallback_force"):
                            self._fallback_force: set[str] = set()
                        self._fallback_force.add(key)
                loaded += 1

        logger.info(
            "MultiQuant cache: loaded residuals ← %s (%d tensors, %d skipped)",
            path, loaded, len(skipped),
        )
        if skipped:
            logger.warning(
                "MultiQuant cache: residuals had unknown keys (first 5): %s",
                skipped[:5],
            )

        # ─── Source-safetensors fallback ───────────────────────────────
        # If the cache was packed before save_residuals captured every
        # non-quant Linear (e.g. vision tower in qwen3_vl with the older
        # streaming-quant init that stubbed UnquantizedLinearMethod), we
        # end up with params still at the (0,) streaming-stub. Recover
        # them from the model's HF safetensors directly so we don't need
        # to re-pack just to fill in unquantized BF16 weights.
        force_keys = getattr(self, "_fallback_force", set())
        self._fallback_load_from_source(model, name_to_param, tp_world,
                                        tp_rank, _LINEAR_PARAM_CLASSES,
                                        force_keys=force_keys)
        return True

    def _fallback_load_from_source(
        self,
        model,
        name_to_param: dict,
        tp_world: int,
        tp_rank: int,
        linear_param_classes: tuple,
        force_keys: set | None = None,
    ) -> None:
        """Materialize stubbed/mis-shaped Linear weights from source HF.

        Triggers (per param):
          1. Param is at (0,) stub with a known _streaming_shape — the
             cache was packed before save_residuals captured this layer.
          2. Param's name starts with ``visual.`` — vision tower weights
             have historically been mis-shaped in the v2 cache (saved at
             TP-shard size instead of full size), so we always reload
             them from source to bypass the cache for that subtree.
          3. Param's key is in ``force_keys`` — the load_residuals copy
             phase raised because the cached tensor's shape didn't match
             the constructed target (e.g. linear_attn.conv1d.weight),
             so the original storage is intact and we must re-fill it
             from source.
        """
        if not self.model_path:
            return
        if force_keys is None:
            force_keys = set()
        # Collect targets: Linear params still at (0,) with a known
        # original shape, or any param whose key the load phase
        # explicitly forced into the fallback (residuals.safetensors copy
        # raised a shape mismatch). With schema v3 the cache should hold
        # full-shape vision/conv1d/gate tensors and this fallback should
        # almost never fire — it stays as a safety net for future cases
        # where save_residuals misses a layer class entirely.
        targets: list[tuple[str, torch.nn.Parameter]] = []
        seen: set[str] = set()
        for name, p in model.named_parameters():
            if name.endswith(".w13_weight") or name.endswith(".w2_weight"):
                continue  # MoE expert tensors handled by XFP cache hit
            stubbed = (p.data.numel() == 0
                       and getattr(p, "_streaming_shape", None))
            forced = name in force_keys
            if stubbed or forced:
                if name in seen:
                    continue
                seen.add(name)
                targets.append((name, p))
        if not targets:
            return

        # Find the safetensors index. Different repos use different
        # filenames; check the common ones.
        from pathlib import Path as _Path
        model_dir = _Path(self.model_path)
        index_path = None
        for candidate in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            cp = model_dir / candidate
            if cp.exists():
                index_path = cp
                break
        if index_path is None:
            # Single-shard model: try a single safetensors directly.
            single = model_dir / "model.safetensors"
            if not single.exists():
                logger.warning(
                    "MultiQuant cache fallback: no safetensors index at %s "
                    "and no model.safetensors — cannot recover %d stubbed "
                    "params (will fail at forward)", model_dir, len(targets))
                return
            shard_map = {None: str(single)}
            weight_map: dict[str, str] = {}
        else:
            try:
                idx = json.loads(index_path.read_text())
                weight_map = idx.get("weight_map", {})
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "MultiQuant cache fallback: cannot read index %s (%s)",
                    index_path, e)
                return
            shard_map = {}

        # Group target params by their source shard so we open each file
        # only once.
        try:
            from safetensors import safe_open
        except ImportError:
            return

        # Some checkpoints prefix everything with "model." (the HF root)
        # while vllm's model has language_model.* / visual.* — the
        # safetensors keys may need a mapping. Try the param name as-is
        # first; if not found, try common rewrites.
        def _candidate_keys(param_name: str) -> list[str]:
            cands = [param_name]
            # Qwen3-VL HF layout: ``model.language_model.layers.*`` and
            # ``model.visual.*`` while vllm exposes
            # ``language_model.model.layers.*`` and ``visual.*``.
            if param_name.startswith("language_model.model."):
                cands.append(
                    "model.language_model."
                    + param_name[len("language_model.model."):])
            if param_name.startswith("language_model.lm_head."):
                # Top-level in HF (no model.* prefix).
                cands.append(
                    "lm_head."
                    + param_name[len("language_model.lm_head."):])
            if param_name.startswith("visual."):
                cands.append("model." + param_name)
            # Generic fallbacks for older HF layouts.
            cands.append(
                param_name.replace("language_model.", "", 1))
            return cands

        from collections import defaultdict
        per_shard: dict[str, list[tuple[str, str, torch.nn.Parameter]]] = (
            defaultdict(list)
        )
        unresolved: list[str] = []
        for name, p in targets:
            resolved_key = None
            shard = None
            for k in _candidate_keys(name):
                if k in weight_map:
                    resolved_key = k
                    shard = weight_map[k]
                    break
            if resolved_key is None and not weight_map:
                # Single-shard fallback: defer key matching to safe_open.
                resolved_key = name
                shard = None
            if resolved_key is None:
                unresolved.append(name)
                continue
            per_shard[shard].append((name, resolved_key, p))

        # device pick
        target_device = _guess_device(model)

        loaded_n = 0
        for shard, items in per_shard.items():
            shard_path = (str(model_dir / shard) if shard
                          else shard_map.get(None))
            try:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    available = set(f.keys())
                    for name, key, p in items:
                        if key not in available:
                            unresolved.append(name)
                            continue
                        full = f.get_tensor(key)
                        # Apply TP slice via the same mechanism as the
                        # main residual path: use the vllm Parameter's
                        # specialized loader if it's column/row/qkv, else
                        # fall back to a plain narrow against the target
                        # per-rank shape (preferred from the param itself,
                        # then _streaming_shape, then full.shape).
                        if p.data.numel() > 0:
                            target_shape = tuple(p.data.shape)
                        else:
                            target_shape = tuple(getattr(
                                p, "_streaming_shape", full.shape))
                        target_dtype = getattr(
                            p, "_streaming_dtype", p.data.dtype)
                        # Allocate the target storage at per-rank shape
                        # only if the existing storage is empty/wrong.
                        if (p.data.numel() == 0
                                or tuple(p.data.shape) != target_shape):
                            p.data = torch.empty(
                                target_shape, dtype=target_dtype,
                                device=target_device,
                            )
                        used_vllm = False
                        if (tp_world > 1 and linear_param_classes
                                and isinstance(p, linear_param_classes)):
                            try:
                                from vllm.model_executor.parameter import (
                                    _ColumnvLLMParameter as _Col,
                                )
                                cpu_t = full.to(
                                    dtype=target_dtype,
                                    device=target_device,
                                )
                                if isinstance(p, _Col):
                                    p.load_column_parallel_weight(cpu_t)
                                else:
                                    p.load_row_parallel_weight(cpu_t)
                                used_vllm = True
                            except Exception:
                                pass
                        if not used_vllm:
                            # Shape-based narrow.
                            if tuple(full.shape) != target_shape:
                                full = self._tp_slice_if_needed(
                                    full, target_shape)
                            if tuple(full.shape) == target_shape:
                                p.data.copy_(full.to(
                                    dtype=target_dtype,
                                    device=target_device))
                            else:
                                # disable_tp layer with full target —
                                # just copy.
                                if full.numel() == p.data.numel():
                                    p.data.copy_(full.view_as(p.data).to(
                                        dtype=target_dtype,
                                        device=target_device))
                        loaded_n += 1
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "MultiQuant cache fallback: failed to read %s (%s)",
                    shard_path, e)

        logger.info(
            "MultiQuant cache fallback: recovered %d stubbed params from "
            "source HF safetensors (%s)%s",
            loaded_n, model_dir,
            f" — {len(unresolved)} unresolved (e.g. {unresolved[:3]})"
            if unresolved else "",
        )

    # ─── Summary ──────────────────────────────────────────────────────

    def log_summary(self) -> None:
        total_hits = sum(self._hits.values())
        total_miss = sum(self._misses.values())
        hit_breakdown = ", ".join(
            f"{m}={n}" for m, n in sorted(self._hits.items())
        ) or "-"
        miss_breakdown = ", ".join(
            f"{m}={n}" for m, n in sorted(self._misses.items())
        ) or "-"
        logger.info(
            "MultiQuant cache summary: hits={%s} misses={%s} "
            "| saves=%d (%.1fs) save_fail=%d load_fail=%d "
            "| cache-io-time=%.1fs",
            hit_breakdown, miss_breakdown,
            self._saves, self._t_saved_s, self._save_fail, self._load_fail,
            self._t_loaded_s,
        )
        if total_hits > 0 and total_miss > 0:
            logger.warning(
                "MultiQuant cache: %d hits + %d misses — cache is PARTIAL. "
                "Some layers re-quantized. Check prefix stability or "
                "missing files.",
                total_hits, total_miss,
            )


def _best_effort_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"
