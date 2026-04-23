#!/usr/bin/env python3
"""Aggregate per-seed lm-eval-harness outputs into a summary.json per task.

Input layout (produced by bench.lm-eval wrapper):

    measurements/<date>-eval-harness/<label>/
        server_config.json
        seed0/
            gsm8k/
                results.json              # lm-eval raw
                samples_<task>_*.jsonl    # per-sample predictions
            wikitext/
                results.json
        seed1/  seed2/  ...

Output (one per task, next to seed dirs):

    <label>/
        gsm8k.summary.json
        wikitext.summary.json
        mmlu.summary.json

Each summary has mean / sample-std / 95% t-CI per metric, plus the raw
per-seed values for downstream plotting. Single-seed runs have std/CI
null — documented in the schema.

Usage:
    python -m bench_lm_eval.aggregate \\
        --run-dir measurements/20260423-eval-harness/35b-xfp-4sample
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from scipy.stats import t as student_t
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def find_seed_dirs(run_dir: Path) -> list[Path]:
    """Return sorted list of seedN/ subdirs under run_dir."""
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed")
    )


def find_task_dirs(seed_dir: Path) -> list[Path]:
    """Return subdirs of seedN/ — one per task (gsm8k/, wikitext/, ...)."""
    return sorted(
        p for p in seed_dir.iterdir()
        if p.is_dir() and (p / "results.json").exists()
    )


def extract_metrics(results_json: dict, task: str) -> dict:
    """Pull the metric dict for a given task out of lm-eval's results.

    lm-eval's results.json has structure:
        {"results": {"<task>": {"metric_name,filter": value, ...}, ...}, ...}

    We flatten keys by stripping ',filter-variants' suffix so downstream
    aggregation across seeds sees consistent names. Keep the suffix only
    when it carries meaning (e.g. 'exact_match,strict-match' vs
    'exact_match,flexible-extract').
    """
    res = results_json.get("results", {}).get(task, {})
    out = {}
    for k, v in res.items():
        if not isinstance(v, (int, float)):
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        out[k] = float(v)
    return out


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple:
    """Return (mean, std, ci_low, ci_high) — std/ci null for n<2 or no scipy."""
    n = len(values)
    if n == 0:
        return (None, None, None, None)
    mean = sum(values) / n
    if n < 2 or not HAS_SCIPY:
        return (mean, None, None, None)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = var ** 0.5
    se = std / (n ** 0.5)
    lo, hi = student_t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return (mean, std, float(lo), float(hi))


def aggregate_task(
    run_dir: Path, task: str, task_filename: str | None = None,
) -> dict:
    """Gather all seed results for one task. task_filename defaults to task."""
    task_filename = task_filename or task
    per_seed_values: dict[str, list[float]] = {}
    per_seed_meta = []
    for seed_dir in find_seed_dirs(run_dir):
        results_path = seed_dir / task_filename / "results.json"
        if not results_path.exists():
            continue
        try:
            doc = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            continue
        metrics = extract_metrics(doc, task)
        per_seed_meta.append({
            "seed_dir": seed_dir.name,
            "metrics": metrics,
        })
        for metric, value in metrics.items():
            per_seed_values.setdefault(metric, []).append(value)

    if not per_seed_meta:
        return {
            "task": task,
            "n_seeds": 0,
            "metrics": {},
            "per_seed": [],
            "note": f"no {task_filename}/results.json found under any seed",
        }

    agg_metrics = {}
    for metric, vals in per_seed_values.items():
        mean, std, lo, hi = compute_ci(vals)
        agg_metrics[metric] = {
            "mean": mean,
            "std": std,
            "ci95": [lo, hi] if lo is not None else None,
            "per_seed": vals,
        }

    return {
        "task": task,
        "n_seeds": len(per_seed_meta),
        "metrics": agg_metrics,
        "per_seed": per_seed_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate lm-eval seeds into per-task summary.json.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="<label>/ dir with seedN/ subdirs.")
    ap.add_argument("--tasks", default="gsm8k,wikitext,mmlu",
                    help="Comma-sep tasks to aggregate. Missing task = skip.")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} not a directory", file=sys.stderr)
        return 1

    wrote = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        # lm-eval's canonical task name for wikitext is "wikitext" but
        # many users create dirs named "wikitext2"; support both.
        for task_dir_name in (task, f"{task}2"):
            summary = aggregate_task(args.run_dir, task,
                                     task_filename=task_dir_name)
            if summary["n_seeds"] > 0:
                out_path = args.run_dir / f"{task}.summary.json"
                out_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True))
                wrote.append(out_path.name)
                break

    if not wrote:
        print(f"warn: no tasks had seed results under {args.run_dir}",
              file=sys.stderr)
        return 1
    print(f"wrote: {' '.join(wrote)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
