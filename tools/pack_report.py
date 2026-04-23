#!/usr/bin/env python3
"""Offline aggregator for MultiQuant-packed cache shards.

Scans ``<cache-root>/<model>/<hash>/`` and emits a Markdown report
comparing bit-distributions, cos-similarity, outlier-fractions and
effective-bits-per-param across models × hyperparameter sweeps.

All data is read from the cache artefacts written during ``--quant-only``
runs (no re-quantisation required):

    <cache-root>/<model>/<hash>/
        manifest.json              # top-level policy + env + vllm_version
        xfp_summary.json           # aggregated by-layer-class stats
        <layer_prefix>/_manifest.json   # per-layer XFPPackStats.to_dict()
        residuals.safetensors      # not read by this tool

Usage:
    python tools/pack_report.py \\
        --cache-root /data/tensordata/mq-cache \\
        --models "Qwen*,GLM*" \\
        --out REPORT.md

Output has three sections:

1. **Per-Model Bit Distribution**: for each (model, hash) shard, a table
   of Layer-Class × Bits-Histogram showing how many layers of each
   class quantised to 2, 3, or 4 bits.
2. **Cross-Model Summary**: eff-bits-per-param, avg-cos, avg-outlier for
   each shard, sortable by Pareto memory × quality.
3. **Hyperparameter Sweep**: when multiple shards for the same model
   exist with different env snapshots (XFP_MIN_COS, XFP_MOE_SAMPLE_EXPERTS,
   ...), a diff-table highlighting bit-escalation deltas.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def find_shards(cache_root: Path, model_patterns: list[str]):
    """Iterate (model_name, shard_hash, shard_dir) for matching cache shards."""
    if not cache_root.is_dir():
        print(f"error: cache-root {cache_root} not a directory", file=sys.stderr)
        return
    for model_dir in sorted(cache_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if not any(fnmatch.fnmatch(model_name, p) for p in model_patterns):
            continue
        for shard_dir in sorted(model_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            if not (shard_dir / "manifest.json").exists():
                continue
            yield model_name, shard_dir.name, shard_dir


def load_shard(shard_dir: Path) -> dict | None:
    """Load manifest + xfp_summary for one shard, return None on failure."""
    try:
        manifest = json.loads((shard_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"warn: skip {shard_dir}: manifest unreadable ({e})",
              file=sys.stderr)
        return None
    summary_path = shard_dir / "xfp_summary.json"
    xfp_summary = None
    if summary_path.exists():
        try:
            xfp_summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "shard_dir": shard_dir,
        "hash": shard_dir.name,
        "manifest": manifest,
        "xfp_summary": xfp_summary,
    }


def render_section_1(shards: list[dict]) -> str:
    """Per-model bit-distribution tables.

    Group shards by model_basename. For each shard: show Class × Bits table.
    """
    out = ["## 1. Per-Model Bit Distribution", ""]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in shards:
        mb = s["manifest"].get("model_basename", "?")
        by_model[mb].append(s)

    for model, model_shards in sorted(by_model.items()):
        out.append(f"### {model}")
        out.append("")
        for s in model_shards:
            xfp = s["xfp_summary"]
            if xfp is None:
                out.append(f"- Shard `{s['hash']}`: no xfp_summary.json "
                          "(pre-tooling pack, skipped)")
                out.append("")
                continue
            totals = xfp["totals"]
            hp = xfp.get("hyperparams", {})
            hp_str = ", ".join(
                f"{k}={v}" for k, v in sorted(hp.items())) or "(defaults)"
            out.append(f"**Shard `{s['hash']}`** — {hp_str}")
            out.append("")
            out.append("| Layer Class | N | xfp2 | xfp3 | xfp4 | "
                       "avg cos | avg outlier | eff bits/param |")
            out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for cls, d in sorted(xfp["by_layer_class"].items()):
                bh = d.get("bits_histogram", {})
                row = (f"| {cls} | {d['count']} | "
                       f"{bh.get(2, bh.get('2', 0))} | "
                       f"{bh.get(3, bh.get('3', 0))} | "
                       f"{bh.get(4, bh.get('4', 0))} | "
                       f"{d['avg_cos_sim']:.4f} | "
                       f"{100*d['avg_outlier_fraction']:.3f}% | "
                       f"{d['eff_bits_per_param']:.2f} |")
                out.append(row)
            out.append(f"| **total** | {totals['n_layers']} | — | — | — | "
                       f"— | {100*totals['avg_outlier_fraction']:.3f}% | "
                       f"**{totals['eff_bits_per_param']:.2f}** |")
            out.append("")
    return "\n".join(out)


def render_section_2(shards: list[dict]) -> str:
    """Cross-model Pareto summary."""
    out = ["## 2. Cross-Model Pareto (Memory × Quality)", ""]
    out.append("Sorted by effective bits ascending (more compression first).")
    out.append("")
    out.append("| Model | Shard | Layers | eff bits/param | avg outlier | "
               "Hyperparams |")
    out.append("|---|---|---:|---:|---:|---|")
    rows = []
    for s in shards:
        xfp = s["xfp_summary"]
        if xfp is None:
            continue
        mb = s["manifest"].get("model_basename", "?")
        totals = xfp["totals"]
        hp = xfp.get("hyperparams", {})
        hp_keys = sorted(hp.keys())
        hp_str = ", ".join(f"{k}={hp[k]}" for k in hp_keys) or "(defaults)"
        rows.append((
            totals["eff_bits_per_param"],
            f"| {mb} | `{s['hash']}` | {totals['n_layers']} | "
            f"{totals['eff_bits_per_param']:.3f} | "
            f"{100*totals['avg_outlier_fraction']:.3f}% | {hp_str} |",
        ))
    for _, row in sorted(rows):
        out.append(row)
    out.append("")
    return "\n".join(out)


def render_section_3(shards: list[dict]) -> str:
    """Hyperparameter sweep diff within same model."""
    out = ["## 3. Hyperparameter Sweep (per-model diff)", ""]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in shards:
        mb = s["manifest"].get("model_basename", "?")
        by_model[mb].append(s)

    any_diff = False
    for model, model_shards in sorted(by_model.items()):
        if len(model_shards) < 2:
            continue
        # Collect unique env keys that differ between shards
        all_envs = []
        for s in model_shards:
            if s["xfp_summary"] is None:
                continue
            all_envs.append(s["xfp_summary"].get("hyperparams", {}))
        if len(all_envs) < 2:
            continue
        differing_keys: set[str] = set()
        for k in set().union(*(e.keys() for e in all_envs)):
            vals = {e.get(k) for e in all_envs}
            if len(vals) > 1:
                differing_keys.add(k)
        if not differing_keys:
            continue
        any_diff = True
        out.append(f"### {model}")
        out.append("")
        out.append("| Shard | " + " | ".join(sorted(differing_keys)) +
                   " | eff bits | xfp2 | xfp3 | xfp4 |")
        out.append("|---|" + "|".join(["---"] * len(differing_keys)) +
                   "|---:|---:|---:|---:|")
        for s in model_shards:
            xfp = s["xfp_summary"]
            if xfp is None:
                continue
            hp = xfp.get("hyperparams", {})
            totals = xfp["totals"]
            # Aggregate bit histogram across all classes
            total_bits: dict[int, int] = defaultdict(int)
            for _, d in xfp["by_layer_class"].items():
                for b, cnt in d.get("bits_histogram", {}).items():
                    total_bits[int(b)] += cnt
            row = f"| `{s['hash']}` |"
            for k in sorted(differing_keys):
                row += f" {hp.get(k, '-')} |"
            row += (f" {totals['eff_bits_per_param']:.3f} |"
                    f" {total_bits.get(2, 0)} |"
                    f" {total_bits.get(3, 0)} |"
                    f" {total_bits.get(4, 0)} |")
            out.append(row)
        out.append("")
    if not any_diff:
        out.append("_(no multi-shard per-model differences found)_")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate MultiQuant pack caches into a Markdown report.")
    ap.add_argument("--cache-root", type=Path,
                    default=Path("/data/tensordata/mq-cache"),
                    help="Root directory with <model>/<hash>/ shards.")
    ap.add_argument("--models", default="*",
                    help="Comma-separated glob patterns of model-basenames "
                         "to include (e.g. 'Qwen*,GLM*'). Default: all.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Markdown path.")
    args = ap.parse_args()

    patterns = [p.strip() for p in args.models.split(",") if p.strip()]

    shards = []
    for model, h, sd in find_shards(args.cache_root, patterns):
        s = load_shard(sd)
        if s is not None:
            shards.append(s)
    if not shards:
        print(f"error: no shards found under {args.cache_root} matching "
              f"{patterns}", file=sys.stderr)
        return 1

    import time
    header = [
        "# MultiQuant Pack Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Cache root: `{args.cache_root}`",
        f"Models matched: {args.models}",
        f"Shards read: **{len(shards)}**",
        "",
    ]
    body = [
        render_section_1(shards),
        render_section_2(shards),
        render_section_3(shards),
    ]
    args.out.write_text("\n".join(header + body))
    print(f"wrote {args.out} ({len(shards)} shards)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
