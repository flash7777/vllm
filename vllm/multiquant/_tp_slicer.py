# SPDX-License-Identifier: Apache-2.0
"""Generic Tensor-Parallel slicer for MultiQuant cache v2 entries.

Rather than re-implement TP-slice semantics per-quant-method, the cache
manifest carries a ``tp_role`` per tensor + the metadata that role needs
(``output_dim``, ``input_dim``, ``head_size``, ``shard_offsets``, ...).
This module dispatches on ``tp_role`` and applies the correct narrow.

Roles cover all MultiQuant pack outputs plus the residuals pulled from
vllm's ColumnParallelLinear / RowParallelLinear / QKVParallelLinear /
MergedColumnParallelLinear / VocabParallelEmbedding / FusedMoE /
GatedDeltaNet primitives.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _slice_dim(
    t: torch.Tensor, dim: int, tp_rank: int, tp_world: int,
) -> torch.Tensor:
    """Narrow ``t`` on ``dim`` to the rank's contiguous slice.

    Asserts ``t.shape[dim]`` is divisible by ``tp_world``.
    """
    sz = t.shape[dim]
    if sz % tp_world != 0:
        raise ValueError(
            f"_slice_dim: shape[{dim}]={sz} not divisible by tp_world={tp_world}")
    chunk = sz // tp_world
    return t.narrow(dim, tp_rank * chunk, chunk)


def _slice_dim_padded(
    t: torch.Tensor, dim: int, tp_rank: int, tp_world: int,
) -> torch.Tensor:
    """Like ``_slice_dim`` but allows the last rank to have a remainder.

    Used for vocab_parallel_embedding which pads the vocab dim. Falls
    back to even split when divisible.
    """
    sz = t.shape[dim]
    chunk = (sz + tp_world - 1) // tp_world
    start = tp_rank * chunk
    actual = min(chunk, sz - start)
    if actual <= 0:
        # This rank gets zero rows (shouldn't happen in well-sized vocab)
        # Return an empty slice to keep shapes consistent downstream.
        return t.narrow(dim, 0, 0)
    return t.narrow(dim, start, actual)


def _slice_qkv(
    t: torch.Tensor, entry: dict, tp_rank: int, tp_world: int,
) -> torch.Tensor:
    """QKVParallelLinear-style: q is split by tp_world, kv may be replicated.

    Manifest entry must contain:
      - output_dim: int
      - shard_offsets: [(name, offset, size), ...] for q, k, v
      - num_kv_head_replicas: int (for replicating kv when num_kv < tp_world)
    """
    output_dim = entry["output_dim"]
    shard_offsets = entry["shard_offsets"]
    num_kv_head_replicas = int(entry.get("num_kv_head_replicas", 1))
    out_chunks = []
    for name, off, size in shard_offsets:
        if name == "q":
            chunk = size // tp_world
            out_chunks.append(t.narrow(output_dim, off + tp_rank * chunk, chunk))
        else:  # "k" or "v"
            # KV may be replicated across num_kv_head_replicas adjacent ranks
            kv_rank = tp_rank // num_kv_head_replicas
            kv_world = tp_world // num_kv_head_replicas
            if kv_world < 1:
                kv_world = 1
            chunk = size // kv_world
            out_chunks.append(t.narrow(output_dim, off + kv_rank * chunk, chunk))
    return torch.cat(out_chunks, dim=output_dim).contiguous()


def _slice_merged(
    t: torch.Tensor, shard_offsets: list, tp_rank: int, tp_world: int,
    output_dim: int = 0,
) -> torch.Tensor:
    """MergedColumnParallelLinear: each shard slot split TP-evenly.

    shard_offsets: [(name, offset, size), ...]. Each slot contributes
    ``size / tp_world`` elements at position ``offset + tp_rank * size_tp``.
    """
    out_chunks = []
    for _name, off, size in shard_offsets:
        chunk = size // tp_world
        out_chunks.append(t.narrow(output_dim, off + tp_rank * chunk, chunk))
    return torch.cat(out_chunks, dim=output_dim).contiguous()


def _slice_mamba(
    t: torch.Tensor, shard_spec: list, tp_rank: int, tp_world: int,
) -> torch.Tensor:
    """GatedDeltaNet/Mamba2 sharded loader.

    shard_spec: [(dim, replication_factor), ...] — one entry per
    contiguous block along output dim. ``replication_factor`` says how
    many ranks share a copy (e.g. 2 for partial replication).
    """
    out_chunks = []
    cursor = 0
    for dim_size, replication in shard_spec:
        # rank within the replication group
        eff_world = max(1, tp_world // replication)
        eff_rank = tp_rank // replication
        chunk = dim_size // eff_world
        out_chunks.append(t.narrow(0, cursor + eff_rank * chunk, chunk))
        cursor += dim_size
    return torch.cat(out_chunks, dim=0).contiguous()


def slice_for_tp(
    tensor: torch.Tensor,
    entry: dict,
    tp_rank: int,
    tp_world: int,
    metadata: dict | None = None,
) -> torch.Tensor:
    """Apply role-specific TP slicing.

    Args:
        tensor: full (TP=1-packed) source tensor, typically on CPU.
        entry: per-tensor manifest entry. Required keys:
            ``tp_role``: see role table below.
            Plus role-specific keys (``output_dim``, ``input_dim``,
            ``input_dim_packed``, ``shard_offsets``, ``shard_spec``,
            ``head_size``, ``num_kv_head_replicas``).
        tp_rank, tp_world: this rank's index + world size.
        metadata: shard-level metadata for cross-checks (e.g. ``bits``,
            ``K_orig`` for bit-alignment validation in the row-packed
            path).

    Returns:
        sliced tensor (still on input device — caller moves to GPU).

    Raises:
        ValueError if shape isn't TP-divisible or alignment violated.
    """
    role = entry.get("tp_role", "replicated")

    if tp_world == 1 or role in ("replicated", "fused_moe_codebook_w2"):
        return tensor

    if role == "column":
        try:
            return _slice_dim(tensor, entry["output_dim"], tp_rank, tp_world)
        except ValueError as e:
            logger.warning(
                "column tp_slice on dim=%s failed (%s) — keeping tensor "
                "replicated", entry.get("output_dim"), e)
            return tensor

    if role == "row":
        # Either bit-packed (input_dim_packed) or unpacked (input_dim)
        ipd = entry.get("input_dim_packed")
        if ipd is not None:
            if metadata:
                K_orig = metadata.get("K_orig")
                bits = metadata.get("bits")
                if K_orig is not None and bits is not None:
                    if (K_orig * bits) % (32 * tp_world) != 0:
                        # Bit-packing of K_orig doesn't divide evenly per
                        # rank — would need a post-hoc bit-level re-pack.
                        # Treat as replicated and let the forward kernel
                        # handle the per-rank consequences. Logged so the
                        # operator notices.
                        logger.warning(
                            "row-packed tp_slice: K_orig*bits=%d not "
                            "divisible by 32*tp_world=%d — keeping cache "
                            "tensor replicated (math may be off for this "
                            "layer at TP>1 unless re-packed with K-pad)",
                            K_orig * bits, 32 * tp_world)
                        return tensor
            try:
                return _slice_dim(tensor, ipd, tp_rank, tp_world)
            except ValueError as e:
                logger.warning(
                    "row-packed tp_slice on dim=%d failed (%s) — keeping "
                    "tensor replicated", ipd, e)
                return tensor
        return _slice_dim(tensor, entry["input_dim"], tp_rank, tp_world)

    if role == "qkv":
        # If full qkv shard metadata is present, do per-Q-K-V slicing.
        # Otherwise fall back to plain column slice — works when the
        # QKV-fused output dim is evenly divisible by tp_world (which is
        # the common case for q_heads + 2*kv_heads with kv_heads >= tp_world).
        if "shard_offsets" in entry and "output_dim" in entry:
            return _slice_qkv(tensor, entry, tp_rank, tp_world)
        try:
            return _slice_dim(
                tensor, entry.get("output_dim", 1), tp_rank, tp_world)
        except ValueError as e:
            logger.warning(
                "qkv tp_slice fallback (no shard_offsets) failed (%s) — "
                "keeping replicated", e)
            return tensor

    if role == "merged_column":
        # Fall back to plain column slicing when shard_offsets aren't
        # recorded (caches written before the merged-shard-offset logic
        # landed, or single-output-element merged layers like
        # GatedDeltaNet's in_proj_qkvz where the TP-slice is just a
        # straight narrow on the output dim).
        shard_offsets = entry.get("shard_offsets")
        if shard_offsets is None:
            return _slice_dim(
                tensor, entry.get("output_dim", 1), tp_rank, tp_world)
        return _slice_merged(
            tensor, shard_offsets, tp_rank, tp_world,
            output_dim=entry.get("output_dim", 0))

    if role == "vocab_parallel":
        return _slice_dim_padded(
            tensor, entry["output_dim"], tp_rank, tp_world)

    if role == "fused_moe_w13":
        # [E, K_packed, N] — slice N (dim 2). Output dim of column-parallel
        # gate_up projection.
        return _slice_dim(tensor, 2, tp_rank, tp_world)

    if role == "fused_moe_codebook_w13":
        # [E, N, lut] — slice N (dim 1).
        return _slice_dim(tensor, 1, tp_rank, tp_world)

    if role == "fused_moe_w2":
        # [E, K_packed, N] — slice K_packed (dim 1). Input dim of row-parallel
        # down projection. Bit-alignment required.
        if metadata:
            K_orig = metadata.get("K_orig_w2") or metadata.get("K_orig")
            bits = metadata.get("bits")
            if K_orig is not None and bits is not None:
                if (K_orig * bits) % (32 * tp_world) != 0:
                    raise ValueError(
                        f"fused_moe_w2 tp_slice: K_orig*bits="
                        f"{K_orig*bits} not divisible by 32*tp_world="
                        f"{32*tp_world}")
        return _slice_dim(tensor, 1, tp_rank, tp_world)

    if role.startswith("gated_deltanet_"):
        return _slice_mamba(
            tensor, entry["shard_spec"], tp_rank, tp_world)

    raise ValueError(f"Unknown tp_role: {role!r}")


def infer_tp_role_from_param(name: str, param: torch.nn.Parameter) -> dict:
    """Infer a TP-role manifest entry from a vllm Parameter's attributes.

    Returns a dict with ``tp_role`` + role-specific keys, suitable for
    inclusion in the v2 manifest. Used by ``save_residuals`` to capture
    the slicing semantics that vllm's Parameter objects already encode.

    Falls back to ``replicated`` when no TP-relevant attributes found.
    """
    # vllm carries TP metadata only on its BasevLLMParameter subclasses
    # (ModelWeightParameter, _ColumnvLLMParameter, RowvLLMParameter,
    # QKVParameter, …). Plain ``torch.nn.Parameter`` objects — biases,
    # routers' mlp.gate.weight, GatedDeltaNet conv1d.weight, RMSNorms,
    # … — get an ``output_dim`` attribute via ``set_weight_attrs`` for
    # the parent layer's weight_loader to use, but they aren't actually
    # TP-sharded by the layer (the per-rank shape is set on the param's
    # ``.data`` directly by the layer constructor, not via slicing).
    # Treat plain Parameters as replicated regardless of any
    # ``output_dim`` attribute, otherwise the load-time _tp_slicer
    # would halve them at TP>1.
    try:
        from vllm.model_executor.parameter import BasevLLMParameter
        _is_vllm_param = isinstance(param, BasevLLMParameter)
    except ImportError:
        _is_vllm_param = False
    if not _is_vllm_param:
        return {"tp_role": "replicated"}

    output_dim = getattr(param, "output_dim", None)
    input_dim = getattr(param, "input_dim", None)
    packed_dim = getattr(param, "packed_dim", None)
    head_size = getattr(param, "head_size", None)

    # disable_tp layers (e.g. Qwen3-VL vision tower under
    # mm-encoder-tp-mode=data) attach output_dim/input_dim attributes
    # for the loader's q/k/v narrow logic, but the param's tp_size is
    # forced to 1 by Linear.__init__. Treat those as replicated so the
    # generic column/row slicer doesn't halve them at TP > 1.
    if getattr(param, "tp_size", None) == 1:
        return {"tp_role": "replicated"}

    # Heuristics — vllm's set_weight_attrs usually sets exactly one of these
    if output_dim is not None:
        # Could be column, qkv, merged_column, vocab_parallel — but without
        # additional context we treat as plain column.
        # Vocab-parallel is identified by the param name pattern.
        if "embed_tokens" in name or "lm_head" in name:
            return {
                "tp_role": "vocab_parallel",
                "output_dim": output_dim,
            }
        return {
            "tp_role": "column",
            "output_dim": output_dim,
            "head_size": head_size,
        }
    if input_dim is not None:
        return {
            "tp_role": "row",
            "input_dim": input_dim,
        }
    return {"tp_role": "replicated"}
