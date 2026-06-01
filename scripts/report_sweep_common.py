#!/usr/bin/env python3
"""Aggregate planner sweep trajectory summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def avg(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def med(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def std(values: list[float]) -> float | None:
    return float(stdev(values)) if len(values) >= 2 else None


def pct(value: float | None) -> float | None:
    return None if value is None else float(100.0 * value)


def pct_str(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def num_str(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def latex_num(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def latex_pm(mean_value: float | None, std_value: float | None, digits: int = 3) -> str:
    if mean_value is None:
        return "--"
    if std_value is None:
        return latex_num(mean_value, digits)
    return f"{mean_value:.{digits}f} $\\pm$ {std_value:.{digits}f}"


def load_summaries(sweep_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(sweep_dir.glob("*/trajectory_summary.json"))
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            summaries.append((path, json.load(handle)))
    return summaries


def coverage_fraction(metadata: dict[str, Any], key: str) -> float | None:
    coverage = metadata.get(key) or {}
    return finite_float(coverage.get("fraction"))


def post_step_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records = summary.get("step_records") or []
    return [record for record in records if record.get("phase") in {"post_step", "executed_step"}]


def all_state_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary.get("step_records") or [])


def min_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [finite_float(record.get(key)) for record in records]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def min_task_metric(records: list[dict[str, Any]], key: str, absolute: bool = False) -> float | None:
    values: list[float] = []
    for record in records:
        metrics = record.get("task_metrics") or {}
        value = finite_float(metrics.get(key))
        if value is None:
            continue
        values.append(abs(value) if absolute else value)
    return min(values) if values else None


def max_task_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for record in records:
        metrics = record.get("task_metrics") or {}
        value = finite_float(metrics.get(key))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def timing_value(timings: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_float(timings.get(key))
        if value is not None:
            return value
    return None


def timing_per_step(records: list[dict[str, Any]], ignored_initial_solves: int = 2) -> tuple[float | None, float | None, float | None, int]:
    timed_records = records[ignored_initial_solves:]
    with_vit_values: list[float] = []
    without_vit_values: list[float] = []
    vit_values: list[float] = []
    for record in timed_records:
        timings = record.get("timings_sec") or record.get("timings") or {}
        vit = timing_value(timings, "vit_encode", "encode_time_s") or 0.0
        total = timing_value(timings, "total", "step_compute_time_s")
        if total is None:
            mppi = timing_value(timings, "mppi_run", "mppi_time_s") or 0.0
            sls = timing_value(timings, "sls_solve", "sls_solve_time_s") or 0.0
            total = vit + mppi + sls
        with_vit_values.append(total)
        without_vit_values.append(max(total - vit, 0.0))
        vit_values.append(vit)
    return avg(with_vit_values), avg(without_vit_values), avg(vit_values), len(with_vit_values)


def solver_status_counts(summary: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in post_step_records(summary):
        status = record.get("solver_status") or record.get("status")
        if isinstance(status, str):
            counts[status] += 1
    return counts


def base_run_metrics(path: Path, summary: dict[str, Any], sim: str) -> dict[str, Any]:
    metadata = summary.get("metadata") or {}
    records = all_state_records(summary)
    post_records = post_step_records(summary)
    with_vit, without_vit, vit_only, timing_steps_used = timing_per_step(post_records)
    latent_min = min_record_value(records, "latent_goal_error")
    one_step_fraction = coverage_fraction(metadata, "disturbance_ellipsoid_coverage")
    state_fraction = coverage_fraction(metadata, "state_latent_ellipsoid_coverage")
    safe = metadata.get("trajectory_safe_by_classifier")
    goal_reached = bool(metadata.get("goal_reached", False))
    goal_success = bool(metadata.get("reached_ogbench_success", goal_reached))

    return {
        "sim": sim,
        "summary_path": str(path),
        "run_dir": str(path.parent),
        "episode_idx": metadata.get("episode_idx"),
        "seed": metadata.get("episode_seed", metadata.get("seed")),
        "executed_steps": metadata.get("executed_steps"),
        "num_logged_states": metadata.get("num_logged_states"),
        "stop_reason": metadata.get("stop_reason"),
        "goal_reached": goal_reached,
        "goal_success": goal_success,
        "trajectory_safe": None if safe is None else bool(safe),
        "one_step_error_coverage_fraction": one_step_fraction,
        "state_ellipsoid_fraction": state_fraction,
        "min_latent_goal_error": latent_min,
        "avg_solve_time_per_step_with_vit_sec": with_vit,
        "avg_solve_time_per_step_without_vit_sec": without_vit,
        "avg_vit_encode_time_per_step_sec": vit_only,
        "timing_initial_solves_ignored": 2,
        "timing_steps_used": timing_steps_used,
        "timing_totals_sec": metadata.get("timing_totals_sec") or {},
        "solver_status_counts": dict(solver_status_counts(summary)),
        "post_step_count": len(post_records),
    }


def rope_run_metrics(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    out = base_run_metrics(path, summary, "rope")
    records = all_state_records(summary)
    out["min_task_error"] = min_record_value(records, "task_error")
    return out


def reacher_run_metrics(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    out = base_run_metrics(path, summary, "reacher")
    records = all_state_records(summary)
    out["min_qpos_goal_error"] = min_record_value(records, "qpos_goal_error")
    out["min_observation_goal_error"] = min_record_value(records, "observation_goal_error")
    return out


def ogbench_run_metrics(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    out = base_run_metrics(path, summary, "ogbench_cube")
    records = all_state_records(summary)
    metric_keys = [
        "block_pos_error_norm",
        "block_yaw_error",
        "block_pose_error_l2",
        "effector_block_distance",
    ]
    mins: dict[str, float | None] = {}
    for key in metric_keys:
        mins[f"min_{key}"] = min_task_metric(records, key, absolute=key.endswith("yaw_error"))
    mins["max_gripper_contact"] = max_task_metric(records, "gripper_contact")
    out.update(mins)
    return out


RUN_METRIC_BUILDERS = {
    "rope": rope_run_metrics,
    "reacher": reacher_run_metrics,
    "ogbench_cube": ogbench_run_metrics,
    "ogbench": ogbench_run_metrics,
}


PRIMARY_RELEVANT_METRIC = {
    "rope": "min_task_error",
    "reacher": "min_qpos_goal_error",
    "ogbench_cube": "min_block_pose_error_l2",
    "ogbench": "min_block_pose_error_l2",
}


DEFAULT_SWEEP_DIR = {
    "rope": REPO_ROOT / "sweeps" / "rope",
    "reacher": REPO_ROOT / "sweeps" / "reacher",
    "ogbench_cube": REPO_ROOT / "sweeps" / "ogbench_cube",
    "ogbench": REPO_ROOT / "sweeps" / "ogbench_cube",
}


def aggregate_runs(sim: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> list[float]:
        return [value for run in runs if (value := finite_float(run.get(key))) is not None]

    safe_flags = [run.get("trajectory_safe") for run in runs if run.get("trajectory_safe") is not None]
    status_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()
    for run in runs:
        status_counts.update(run.get("solver_status_counts") or {})
        stop_reason = run.get("stop_reason")
        if isinstance(stop_reason, str):
            stop_counts[stop_reason] += 1

    metric_keys = sorted(
        key
        for key in {key for run in runs for key in run}
        if key.startswith("min_") or key.startswith("max_")
    )
    per_run_metric_averages = {key: avg(collect(key)) for key in metric_keys}

    aggregate = {
        "sim": sim,
        "num_runs": len(runs),
        "coverage": {
            "one_step_error_mean_pct": pct(avg(collect("one_step_error_coverage_fraction"))),
            "one_step_error_median_pct": pct(med(collect("one_step_error_coverage_fraction"))),
            "state_ellipsoid_mean_pct": pct(avg(collect("state_ellipsoid_fraction"))),
            "state_ellipsoid_median_pct": pct(med(collect("state_ellipsoid_fraction"))),
        },
        "safety": {
            "trajectory_safety_rate_pct": None if not safe_flags else 100.0 * sum(bool(flag) for flag in safe_flags) / len(safe_flags),
            "checked_runs": len(safe_flags),
        },
        "success": {
            "goal_success_rate_pct": None if not runs else 100.0 * sum(bool(run.get("goal_success")) for run in runs) / len(runs),
            "goal_reached_rate_pct": None if not runs else 100.0 * sum(bool(run.get("goal_reached")) for run in runs) / len(runs),
        },
        "errors": {
            "min_latent_goal_error_mean": avg(collect("min_latent_goal_error")),
            "min_latent_goal_error_std": std(collect("min_latent_goal_error")),
            "min_latent_goal_error_median": med(collect("min_latent_goal_error")),
            "primary_relevant_metric": PRIMARY_RELEVANT_METRIC[sim],
            "primary_relevant_metric_mean": avg(collect(PRIMARY_RELEVANT_METRIC[sim])),
            "primary_relevant_metric_std": std(collect(PRIMARY_RELEVANT_METRIC[sim])),
            "per_run_min_or_max_metric_means": per_run_metric_averages,
            "per_run_min_or_max_metric_stds": {key: std(collect(key)) for key in metric_keys},
        },
        "timing": {
            "avg_solve_time_per_step_with_vit_sec": avg(collect("avg_solve_time_per_step_with_vit_sec")),
            "std_solve_time_per_step_with_vit_sec": std(collect("avg_solve_time_per_step_with_vit_sec")),
            "median_solve_time_per_step_with_vit_sec": med(collect("avg_solve_time_per_step_with_vit_sec")),
            "avg_solve_time_per_step_without_vit_sec": avg(collect("avg_solve_time_per_step_without_vit_sec")),
            "std_solve_time_per_step_without_vit_sec": std(collect("avg_solve_time_per_step_without_vit_sec")),
            "median_solve_time_per_step_without_vit_sec": med(collect("avg_solve_time_per_step_without_vit_sec")),
            "avg_vit_encode_time_per_step_sec": avg(collect("avg_vit_encode_time_per_step_sec")),
            "std_vit_encode_time_per_step_sec": std(collect("avg_vit_encode_time_per_step_sec")),
            "avg_executed_steps": avg(collect("executed_steps")),
        },
        "diagnostics": {
            "stop_reason_counts": dict(stop_counts),
            "solver_status_counts": dict(status_counts),
        },
    }
    return aggregate


def markdown_table(sim: str, aggregate: dict[str, Any]) -> str:
    coverage = aggregate["coverage"]
    safety = aggregate["safety"]
    success = aggregate["success"]
    errors = aggregate["errors"]
    timing = aggregate["timing"]
    primary_name = errors["primary_relevant_metric"]
    rows = [
        ("Runs", aggregate["num_runs"]),
        ("One-step error ellipsoid coverage", pct_str(coverage["one_step_error_mean_pct"])),
        ("In-domain latent ellipsoid time", pct_str(coverage["state_ellipsoid_mean_pct"])),
        ("Trajectory safety rate", pct_str(safety["trajectory_safety_rate_pct"])),
        ("Avg min latent goal error", num_str(errors["min_latent_goal_error_mean"])),
        (f"Avg per-run {primary_name}", num_str(errors["primary_relevant_metric_mean"])),
        ("Avg solve time/step with ViT encode", f"{num_str(timing['avg_solve_time_per_step_with_vit_sec'], 6)} s"),
        ("Avg solve time/step without ViT encode", f"{num_str(timing['avg_solve_time_per_step_without_vit_sec'], 6)} s"),
        ("Avg ViT encode time/step", f"{num_str(timing['avg_vit_encode_time_per_step_sec'], 6)} s"),
        ("Goal success rate", pct_str(success["goal_success_rate_pct"])),
        ("Avg executed steps", num_str(timing["avg_executed_steps"], 2)),
    ]
    std_rows = [
        ("Std min latent goal error", num_str(errors["min_latent_goal_error_std"])),
        (f"Std per-run {primary_name}", num_str(errors["primary_relevant_metric_std"])),
        ("Std solve time/step with ViT encode", f"{num_str(timing['std_solve_time_per_step_with_vit_sec'], 6)} s"),
        ("Std solve time/step without ViT encode", f"{num_str(timing['std_solve_time_per_step_without_vit_sec'], 6)} s"),
        ("Std ViT encode time/step", f"{num_str(timing['std_vit_encode_time_per_step_sec'], 6)} s"),
    ]

    metric_rows = [
        (key, value)
        for key, value in errors["per_run_min_or_max_metric_means"].items()
        if key not in {primary_name, "min_latent_goal_error"} and value is not None
    ]

    lines = [
        f"# {sim} Sweep Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(["", "## Standard Deviations", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {name} | {value} |" for name, value in std_rows)
    if metric_rows:
        lines.extend(["", "## Additional Per-Run Metric Averages", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| {name} | {num_str(value)} |" for name, value in metric_rows)

    diagnostics = aggregate["diagnostics"]
    if diagnostics["stop_reason_counts"]:
        lines.extend(["", "## Stop Reasons", "", "| Reason | Count |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in sorted(diagnostics["stop_reason_counts"].items()))

    if diagnostics["solver_status_counts"]:
        lines.extend(["", "## Solver Status Counts", "", "| Status | Count |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in sorted(diagnostics["solver_status_counts"].items()))

    lines.append("")
    return "\n".join(lines)


def latex_table(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    sim = aggregate["sim"]
    coverage = aggregate["coverage"]
    safety = aggregate["safety"]
    success = aggregate["success"]
    errors = aggregate["errors"]
    timing = aggregate["timing"]
    relevant_label = {
        "rope": "Task error",
        "reacher": "Qpos error",
        "ogbench_cube": "Block pose error",
        "ogbench": "Block pose error",
    }.get(sim, errors["primary_relevant_metric"])
    sim_label = {
        "rope": "Rope",
        "reacher": "Reacher",
        "ogbench_cube": "OGBench",
        "ogbench": "OGBench",
    }.get(sim, sim)
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            (
                "Sim & One-step cov. (\\%) & In-domain (\\%) & Safety (\\%) & "
                "Success (\\%) & Min latent $\\downarrow$ & "
                f"Min {relevant_label} $\\downarrow$ & Time w/ ViT (s) & Time w/o ViT (s) \\\\"
            ),
            "\\midrule",
            (
                f"{sim_label} & "
                f"{latex_num(coverage['one_step_error_mean_pct'], 2)} & "
                f"{latex_num(coverage['state_ellipsoid_mean_pct'], 2)} & "
                f"{latex_num(safety['trajectory_safety_rate_pct'], 2)} & "
                f"{latex_num(success['goal_success_rate_pct'], 2)} & "
                f"{latex_pm(errors['min_latent_goal_error_mean'], errors['min_latent_goal_error_std'], 3)} & "
                f"{latex_pm(errors['primary_relevant_metric_mean'], errors['primary_relevant_metric_std'], 3)} & "
                f"{latex_pm(timing['avg_solve_time_per_step_with_vit_sec'], timing['std_solve_time_per_step_with_vit_sec'], 3)} & "
                f"{latex_pm(timing['avg_solve_time_per_step_without_vit_sec'], timing['std_solve_time_per_step_without_vit_sec'], 3)} \\\\"
            ),
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Planner sweep metrics averaged across runs. Timing excludes the first two solves of each run to remove JAX compilation overhead. Error and timing entries are mean $\\pm$ standard deviation.}",
            f"\\label{{tab:{sim}_sweep_metrics}}",
            "\\end{table}",
            "",
        ]
    )


def sim_from_filename() -> str | None:
    name = Path(__file__).name
    match = re.match(r"report_(.+)_metrics\.py", name)
    return match.group(1) if match else None


def run_report(sim: str, argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=f"Aggregate {sim} sweep metrics.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR[sim], help="Directory containing per-run sweep folders.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "sweeps" / "reports", help="Directory for markdown and JSON outputs.")
    parser.add_argument("--prefix", default=sim, help="Output file prefix.")
    args = parser.parse_args(argv)

    summaries = load_summaries(args.sweep_dir)
    builder = RUN_METRIC_BUILDERS[sim]
    runs = [builder(path, summary) for path, summary in summaries]
    aggregate = aggregate_runs(sim, runs)
    report = {
        "sim": sim,
        "sweep_dir": str(args.sweep_dir),
        "aggregate": aggregate,
        "runs": runs,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.prefix}_metrics.json"
    md_path = args.out_dir / f"{args.prefix}_metrics.md"
    tex_path = args.out_dir / f"{args.prefix}_metrics.tex"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown_table(sim, aggregate))
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(latex_table(report))

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")
    return report


if __name__ == "__main__":
    inferred_sim = sim_from_filename()
    if inferred_sim not in RUN_METRIC_BUILDERS:
        raise SystemExit("Use one of the per-sim wrappers: report_rope_metrics.py, report_reacher_metrics.py, report_ogbench_cube_metrics.py")
    run_report(inferred_sim)
