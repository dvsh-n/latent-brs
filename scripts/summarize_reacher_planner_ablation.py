#!/usr/bin/env python3
"""Summarize Reacher planner-ablation metrics JSON files as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+", help="metrics.json files from iLQR and CEM runs.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels, one per metrics file.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path for a compact summary JSON.")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "count": int(finite.size),
    }


def fmt_mean_std(summary: dict[str, float | int], *, scale: float = 1.0, digits: int = 3) -> str:
    mean = float(summary["mean"]) * scale
    std = float(summary["std"]) * scale
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def infer_label(path: Path, metrics: dict[str, Any]) -> str:
    method = str(metrics.get("method", ""))
    if method:
        return method
    return path.parent.name


def collect_step_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for case in cases for record in case.get("step_records", [])]


def summarize_metrics(path: Path, label: str | None) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    cases = list(metrics.get("cases", []))
    step_records = collect_step_records(cases)
    success_rate = float(metrics.get("success_rate", 0.0))
    qpos_success_rate = float(metrics.get("qpos_success_rate", success_rate))
    solve_step_ms = [float(record["solve_time_ms"]) for record in step_records if "solve_time_ms" in record]
    solve_replan_ms = [
        float(record["solve_time_ms"])
        for record in step_records
        if float(record.get("cem_replanned", 1.0)) >= 0.5 and "solve_time_ms" in record
    ]
    total_planning_ms = [
        float(case.get("total_planning_time_ms", sum(float(r.get("solve_time_ms", 0.0)) for r in case.get("step_records", []))))
        for case in cases
    ]
    result = {
        "label": label or infer_label(path, metrics),
        "path": str(path.expanduser().resolve()),
        "success_rate": summarize([success_rate]),
        "qpos_success_rate": summarize([qpos_success_rate]),
        "min_qpos_distance": summarize([float(case["min_qpos_distance"]) for case in cases if "min_qpos_distance" in case]),
        "final_qpos_distance": summarize([float(case["final_qpos_distance"]) for case in cases if "final_qpos_distance" in case]),
        "steps_executed": summarize([float(case["steps_executed"]) for case in cases if "steps_executed" in case]),
        "solve_time_ms_per_control_step": summarize(solve_step_ms),
        "solve_time_ms_per_replan": summarize(solve_replan_ms),
        "total_planning_time_s_per_episode": summarize([value / 1000.0 for value in total_planning_ms]),
    }
    return result


def combine_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "success_rate",
        "qpos_success_rate",
        "min_qpos_distance",
        "final_qpos_distance",
        "steps_executed",
        "solve_time_ms_per_control_step",
        "solve_time_ms_per_replan",
        "total_planning_time_s_per_episode",
    ]
    combined: dict[str, Any] = {
        "label": label,
        "path": [row["path"] for row in rows],
        "num_runs": len(rows),
    }
    for key in keys:
        combined[key] = summarize([float(row[key]["mean"]) for row in rows])
    return combined


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Planner | Success (%) | Min qpos dist | Final qpos dist | Step ms | Replan ms | Planner s/ep |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| "
            f"{row['label']} | "
            f"{fmt_mean_std(row['success_rate'], digits=2)} | "
            f"{fmt_mean_std(row['min_qpos_distance'])} | "
            f"{fmt_mean_std(row['final_qpos_distance'])} | "
            f"{fmt_mean_std(row['solve_time_ms_per_control_step'], digits=2)} | "
            f"{fmt_mean_std(row['solve_time_ms_per_replan'], digits=2)} | "
            f"{fmt_mean_std(row['total_planning_time_s_per_episode'], digits=2)} |"
        )


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) not in (0, len(args.metrics)):
        raise ValueError("--labels must provide one label per metrics file.")
    labels = args.labels or [None] * len(args.metrics)
    per_file_rows = [summarize_metrics(path, label) for path, label in zip(args.metrics, labels)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_file_rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    rows = [combine_rows(label, group_rows) if len(group_rows) > 1 else group_rows[0] for label, group_rows in grouped.items()]
    print_markdown(rows)
    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump({"rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
