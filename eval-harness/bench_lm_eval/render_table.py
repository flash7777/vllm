#!/usr/bin/env python3
"""Render a Markdown comparison table across labels in a date-dir.

Reads all ``<date>-eval-harness/<label>/*.summary.json`` files and emits
one comparison table per task, joined on label. Numbers shown as
``mean ± std`` with CI95 in a secondary row when present.

Usage:
    python -m bench_lm_eval.render_table \\
        --date-dir measurements/20260423-eval-harness \\
        --out REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def discover_summaries(date_dir: Path) -> dict:
    """Map task -> [(label, summary_dict), ...] for each *.summary.json."""
    by_task: dict[str, list] = {}
    for label_dir in sorted(date_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for f in sorted(label_dir.glob("*.summary.json")):
            try:
                doc = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            task = doc.get("task", f.stem.replace(".summary", ""))
            by_task.setdefault(task, []).append((label_dir.name, doc))
    return by_task


def fmt_mean_std(metric_entry: dict) -> str:
    mean = metric_entry.get("mean")
    std = metric_entry.get("std")
    n = len(metric_entry.get("per_seed", []))
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.4f} (n={n})"
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def render_task(task: str, entries: list) -> str:
    """Render one task's cross-label comparison as Markdown."""
    out = [f"## {task}", ""]
    # Collect metric names across all labels (union)
    metric_names = set()
    for _, doc in entries:
        metric_names.update(doc.get("metrics", {}).keys())
    if not metric_names:
        out.append(f"_(no metrics found for {task})_")
        out.append("")
        return "\n".join(out)
    metric_list = sorted(metric_names)

    # Header
    out.append("| Label | " + " | ".join(metric_list) + " | n_seeds |")
    out.append("|---" * (len(metric_list) + 2) + "|")
    for label, doc in entries:
        row = [label]
        for m in metric_list:
            entry = doc.get("metrics", {}).get(m, {})
            row.append(fmt_mean_std(entry))
        row.append(str(doc.get("n_seeds", 0)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # CI95 footnote if any metric has one
    ci_labels = [
        (label, doc) for label, doc in entries
        if any(
            (doc.get("metrics", {}).get(m, {}).get("ci95")) is not None
            for m in metric_list)
    ]
    if ci_labels:
        out.append("### 95 % CI (t-dist)")
        out.append("")
        out.append("| Label | " + " | ".join(metric_list) + " |")
        out.append("|---" * (len(metric_list) + 1) + "|")
        for label, doc in ci_labels:
            row = [label]
            for m in metric_list:
                ci = doc.get("metrics", {}).get(m, {}).get("ci95")
                if ci is None or ci[0] is None:
                    row.append("—")
                else:
                    row.append(f"[{ci[0]:.4f}, {ci[1]:.4f}]")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render cross-label Markdown table of lm-eval summaries.")
    ap.add_argument("--date-dir", type=Path, required=True,
                    help="<date>-eval-harness/ with label subdirs.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.date_dir.is_dir():
        print(f"error: {args.date_dir} not a directory", file=sys.stderr)
        return 1

    by_task = discover_summaries(args.date_dir)
    if not by_task:
        print(f"error: no summary.json under {args.date_dir}", file=sys.stderr)
        return 1

    import time
    header = [
        "# Eval-Harness Cross-Label Report",
        "",
        f"Source: `{args.date_dir}`",
        f"Tasks: {', '.join(sorted(by_task.keys()))}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]
    body = [render_task(task, entries) for task, entries in sorted(by_task.items())]
    args.out.write_text("\n".join(header + body))
    print(f"wrote {args.out} ({len(by_task)} tasks, "
          f"{sum(len(v) for v in by_task.values())} summary files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
