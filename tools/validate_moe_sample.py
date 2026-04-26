"""Validate XFP's 4-expert sample assumption for MoE auto-mode.

For each MoE layer in a model, compare the bits chosen by:
  - Sample:   xfp_auto_select on the first 4 experts (production default)
  - Full:     xfp_auto_select on all E experts
  - Random 4: xfp_auto_select on a random 4-expert sample (reviewer-proof)

Also log per-expert median cos at the sample-chosen bit width on ALL experts,
to catch Fall A (sample too tame → forced low bits on broader experts) as a
distribution rather than only a single median.

Output: measurements/20260421-moe-sample-validation/<model>.md

Usage:
    python3 tools/validate_moe_sample.py <model-dir> <output.md>

Reads `experts.gate_up_proj` and `experts.down_proj` stacked tensors (Qwen
layout), falls back to per-expert `experts.N.gate_proj/up_proj/down_proj`
(GLM layout).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

# Import our production auto-select
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm.multiquant.xfp.xfp_pack import xfp_auto_select


def _try_stacked(
    path: str, layer_idx: int
) -> dict[str, torch.Tensor] | None:
    """Qwen-layout: stacked experts.gate_up_proj [E, 2N, K] + down_proj [E, K, N]."""
    # Find safetensors files
    idx = json.load(open(f"{path}/model.safetensors.index.json"))
    wmap = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
    key_gu = f"{prefix}.gate_up_proj"
    key_dn = f"{prefix}.down_proj"
    if key_gu not in wmap or key_dn not in wmap:
        return None
    out = {}
    for k, f in [(key_gu, wmap[key_gu]), (key_dn, wmap[key_dn])]:
        with safe_open(f"{path}/{f}", framework="pt", device="cpu") as fp:
            out[k.split(".")[-1]] = fp.get_tensor(k)
    return out


def _try_per_expert(
    path: str, layer_idx: int, max_experts: int = 512
) -> dict[str, torch.Tensor] | None:
    """GLM-layout: per-expert .N.{gate_proj, up_proj, down_proj}.weight."""
    idx = json.load(open(f"{path}/model.safetensors.index.json"))
    wmap = idx["weight_map"]
    # Figure out how many experts exist
    prefix = f"model.layers.{layer_idx}.mlp.experts"
    expert_nums = set()
    for k in wmap:
        if k.startswith(prefix + "."):
            tail = k[len(prefix) + 1 :]
            try:
                n = int(tail.split(".")[0])
                expert_nums.add(n)
            except ValueError:
                pass
    if not expert_nums:
        return None
    E = max(expert_nums) + 1
    if E > max_experts:
        return None

    # Load all experts' {gate, up, down} and stack
    gate_rows = []
    up_rows = []
    down_rows = []
    for n in range(E):
        for name, lst in [
            ("gate_proj", gate_rows),
            ("up_proj", up_rows),
            ("down_proj", down_rows),
        ]:
            key = f"{prefix}.{n}.{name}.weight"
            if key not in wmap:
                return None
            with safe_open(f"{path}/{wmap[key]}", framework="pt", device="cpu") as fp:
                lst.append(fp.get_tensor(key))
    # Stack → [E, N, K]
    gate = torch.stack(gate_rows, dim=0)
    up = torch.stack(up_rows, dim=0)
    down = torch.stack(down_rows, dim=0)
    gate_up = torch.cat([gate, up], dim=1)  # [E, 2N, K]
    return {"gate_up_proj": gate_up, "down_proj": down}


def load_layer_experts(path: str, layer_idx: int) -> dict[str, torch.Tensor] | None:
    return _try_stacked(path, layer_idx) or _try_per_expert(path, layer_idx)


def count_moe_layers(path: str) -> list[int]:
    idx = json.load(open(f"{path}/model.safetensors.index.json"))
    wmap = idx["weight_map"]
    layers = set()
    for k in wmap:
        # Qwen
        if "language_model.layers." in k and ".mlp.experts" in k and "mtp" not in k:
            n = int(k.split("language_model.layers.")[1].split(".")[0])
            layers.add(n)
        # GLM
        elif k.startswith("model.layers.") and ".mlp.experts" in k:
            n = int(k.split("model.layers.")[1].split(".")[0])
            layers.add(n)
    return sorted(layers)


def per_expert_cos(W_e: torch.Tensor, bits: int, lloyd_iters: int = 5) -> torch.Tensor:
    """Return per-expert median cosine at `bits` (reconstruction after
    Lloyd per-channel within each expert flattened along expert dim)."""
    E, N, K = W_e.shape
    # Flatten all experts into one matrix for Lloyd (this is how auto-select
    # treats sampled experts). Returns per-row cosines.
    flat = W_e.reshape(E * N, K).float()
    # Run one Lloyd + assignment pass per expert via xfp_auto_select internals?
    # Instead, just compute cos post-reconstruction on the full stack.
    from vllm.multiquant.xfp.xfp_pack import _lloyd_per_channel, _assign_indices
    import torch.nn.functional as F

    # Outlier split (same as auto-select)
    mu = flat.mean()
    sigma = flat.std()
    threshold = 4.0 * sigma
    mask = (flat - mu).abs() > threshold
    max_allowed = int(0.02 * flat.numel())
    if int(mask.sum()) > max_allowed and max_allowed > 0:
        flat_abs = (flat - mu).abs().reshape(-1)
        _, top_idx = torch.topk(flat_abs, max_allowed, largest=True, sorted=False)
        m = torch.zeros_like(flat_abs, dtype=torch.bool)
        m[top_idx] = True
        mask = m.reshape_as(flat)
    W_bulk = flat.clone()
    W_bulk[mask] = mu

    cb = _lloyd_per_channel(W_bulk, 1 << bits, lloyd_iters)
    idx = _assign_indices(W_bulk, cb)
    rec = torch.gather(cb, 1, idx)
    # patch outliers back
    flat_r = rec.reshape(-1).clone()
    flat_r[mask.reshape(-1)] = flat.reshape(-1)[mask.reshape(-1)]
    rec = flat_r.reshape_as(flat)

    cos_per_row = F.cosine_similarity(flat, rec, dim=1)  # [E*N]
    # Reshape to [E, N], median per expert
    per_expert = cos_per_row.reshape(E, N).median(dim=1).values
    return per_expert  # [E]


def analyze_layer(
    W: torch.Tensor,  # [E, N, K]
    tag: str,
    min_cos: float = 0.98,
    sample_n: int = 4,
    seed: int = 42,
) -> dict:
    E = W.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_gpu = W.to(device).float()

    # 1) Sample 4 (first)
    sample_first = W_gpu[:sample_n].reshape(-1, W_gpu.shape[2])
    bits_s_first = xfp_auto_select(
        sample_first, candidates=(2, 3, 4), min_cos=min_cos, lloyd_iters=5
    )

    # 2) Sample 4 (random)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(E, generator=g)[:sample_n]
    sample_rand = W_gpu[perm].reshape(-1, W_gpu.shape[2])
    bits_s_rand = xfp_auto_select(
        sample_rand, candidates=(2, 3, 4), min_cos=min_cos, lloyd_iters=5
    )

    # 3) Full
    full = W_gpu.reshape(-1, W_gpu.shape[2])
    bits_full = xfp_auto_select(
        full, candidates=(2, 3, 4), min_cos=min_cos, lloyd_iters=5
    )

    # 4) Per-expert cos at the FULL-decided bit width (baseline to see spread)
    per_exp = per_expert_cos(W_gpu, bits_full, lloyd_iters=5).cpu().tolist()

    del W_gpu, sample_first, sample_rand, full
    torch.cuda.empty_cache() if device.type == "cuda" else None
    gc.collect()

    return {
        "tag": tag,
        "E": E,
        "bits_sample_first": bits_s_first,
        "bits_sample_random": bits_s_rand,
        "bits_full": bits_full,
        "per_expert_cos_min": min(per_exp),
        "per_expert_cos_median": sorted(per_exp)[len(per_exp) // 2],
        "per_expert_cos_max": max(per_exp),
        "per_expert_below_gate": sum(1 for c in per_exp if c < min_cos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("output_md")
    ap.add_argument("--layers", type=str, default="all",
                    help="comma-separated layer indices, or 'all'")
    ap.add_argument("--min-cos", type=float, default=0.98)
    args = ap.parse_args()

    layers = count_moe_layers(args.model_dir)
    if args.layers != "all":
        want = set(int(x) for x in args.layers.split(","))
        layers = [l for l in layers if l in want]

    print(f"MoE layers to analyze: {len(layers)} (from {layers[0]} to {layers[-1]})")

    results = []
    t0 = time.time()
    for i, li in enumerate(layers):
        tensors = load_layer_experts(args.model_dir, li)
        if tensors is None:
            print(f"[{i+1}/{len(layers)}] layer {li}: experts not found, skip")
            continue
        for name, W in tensors.items():
            if W.dim() != 3:
                continue
            tag = f"layer{li}.{name}"
            print(f"[{i+1}/{len(layers)}] {tag} shape={tuple(W.shape)} ...", flush=True)
            r = analyze_layer(W, tag, min_cos=args.min_cos)
            r["layer"] = li
            r["shape"] = list(W.shape)
            results.append(r)
            print(f"  bits(first4)={r['bits_sample_first']}  "
                  f"bits(rand4)={r['bits_sample_random']}  "
                  f"bits(full)={r['bits_full']}  "
                  f"min/med/max per-exp cos={r['per_expert_cos_min']:.4f}/"
                  f"{r['per_expert_cos_median']:.4f}/{r['per_expert_cos_max']:.4f}  "
                  f"below_gate={r['per_expert_below_gate']}/{r['E']}")

    # Aggregate + write markdown
    total = len(results)
    disagree_first = sum(1 for r in results if r["bits_sample_first"] != r["bits_full"])
    disagree_rand = sum(1 for r in results if r["bits_sample_random"] != r["bits_full"])
    fallA = sum(1 for r in results if r["bits_sample_first"] < r["bits_full"])
    fallB = sum(1 for r in results if r["bits_sample_first"] > r["bits_full"])

    with open(args.output_md, "w") as f:
        f.write(f"# MoE sample validation — {args.model_dir}\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Gate:** cos ≥ {args.min_cos}\n")
        f.write(f"**Sample size:** 4 experts\n")
        f.write(f"**Full population:** per layer, all E experts\n\n")
        f.write(f"## Aggregate\n\n")
        f.write(f"- Total MoE blocks analyzed: {total}\n")
        f.write(f"- First-4 sample disagrees with full: {disagree_first} "
                f"({disagree_first/total*100:.1f}%)\n")
        f.write(f"- Random-4 sample disagrees with full: {disagree_rand} "
                f"({disagree_rand/total*100:.1f}%)\n")
        f.write(f"- **Fall A** (sample too tame → *under*-quantised): {fallA}\n")
        f.write(f"- **Fall B** (sample too wild → over-escalated): {fallB}\n\n")
        f.write(f"## Per-block detail\n\n")
        f.write("| tag | shape | bits(first4) | bits(rand4) | bits(full) | "
                "min cos | med cos | max cos | #below gate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['tag']} | {r['shape']} | {r['bits_sample_first']} | "
                    f"{r['bits_sample_random']} | {r['bits_full']} | "
                    f"{r['per_expert_cos_min']:.4f} | "
                    f"{r['per_expert_cos_median']:.4f} | "
                    f"{r['per_expert_cos_max']:.4f} | "
                    f"{r['per_expert_below_gate']}/{r['E']} |\n")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Wrote {args.output_md}")


if __name__ == "__main__":
    main()
