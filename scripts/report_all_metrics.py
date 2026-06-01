#!/usr/bin/env python3
"""Report aggregate metrics for all planner sweep tasks."""

from __future__ import annotations

import json
from pathlib import Path

from report_sweep_common import (
    DEFAULT_SWEEP_DIR,
    REPO_ROOT,
    RUN_METRIC_BUILDERS,
    aggregate_runs,
    latex_num,
    latex_pm,
    load_summaries,
)


SIMS = ["rope", "reacher", "ogbench_cube"]


def build_report(sim: str) -> dict:
    summaries = load_summaries(DEFAULT_SWEEP_DIR[sim])
    runs = [RUN_METRIC_BUILDERS[sim](path, summary) for path, summary in summaries]
    return {
        "sim": sim,
        "sweep_dir": str(DEFAULT_SWEEP_DIR[sim]),
        "aggregate": aggregate_runs(sim, runs),
        "runs": runs,
    }


def combined_markdown(reports: list[dict]) -> str:
    lines = [
        "# Planner Sweep Metrics",
        "",
        "| Sim | One-step cov. | In-domain | Safety | Success | Min latent | Relevant min | Time/step w ViT | Time/step no ViT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        aggregate = report["aggregate"]
        sim = aggregate["sim"]
        errors = aggregate["errors"]
        timing = aggregate["timing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    {"rope": "Rope", "reacher": "Reacher", "ogbench_cube": "OGBench"}[sim],
                    f"{aggregate['coverage']['one_step_error_mean_pct']:.2f}%",
                    f"{aggregate['coverage']['state_ellipsoid_mean_pct']:.2f}%",
                    f"{aggregate['safety']['trajectory_safety_rate_pct']:.2f}%",
                    f"{aggregate['success']['goal_success_rate_pct']:.2f}%",
                    f"{errors['min_latent_goal_error_mean']:.4f} +/- {errors['min_latent_goal_error_std']:.4f}",
                    f"{errors['primary_relevant_metric_mean']:.4f} +/- {errors['primary_relevant_metric_std']:.4f}",
                    f"{timing['avg_solve_time_per_step_with_vit_sec']:.6f} +/- {timing['std_solve_time_per_step_with_vit_sec']:.6f}",
                    f"{timing['avg_solve_time_per_step_without_vit_sec']:.6f} +/- {timing['std_solve_time_per_step_without_vit_sec']:.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def combined_latex(reports: list[dict]) -> str:
    rows = []
    for report in reports:
        aggregate = report["aggregate"]
        sim = aggregate["sim"]
        errors = aggregate["errors"]
        timing = aggregate["timing"]
        rows.append(
            (
                f"{ {'rope': 'Rope', 'reacher': 'Reacher', 'ogbench_cube': 'OGBench'}[sim] } & "
                f"{latex_num(aggregate['coverage']['one_step_error_mean_pct'], 2)} & "
                f"{latex_num(aggregate['coverage']['state_ellipsoid_mean_pct'], 2)} & "
                f"{latex_num(aggregate['safety']['trajectory_safety_rate_pct'], 2)} & "
                f"{latex_num(aggregate['success']['goal_success_rate_pct'], 2)} & "
                f"{latex_pm(errors['min_latent_goal_error_mean'], errors['min_latent_goal_error_std'], 3)} & "
                f"{latex_pm(errors['primary_relevant_metric_mean'], errors['primary_relevant_metric_std'], 3)} & "
                f"{latex_pm(timing['avg_solve_time_per_step_with_vit_sec'], timing['std_solve_time_per_step_with_vit_sec'], 3)} & "
                f"{latex_pm(timing['avg_solve_time_per_step_without_vit_sec'], timing['std_solve_time_per_step_without_vit_sec'], 3)} \\\\"
            )
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            (
                "Task & One-step cov. (\\%) & In-domain (\\%) & Safety (\\%) & "
                "Success (\\%) & Min latent $\\downarrow$ & Relevant min $\\downarrow$ & "
                "Time w/ ViT (s) & Time w/o ViT (s) \\\\"
            ),
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Planner sweep metrics averaged across 35 runs. Timing excludes the first two solves of each run to remove JAX compilation overhead. Error and timing entries are mean $\\pm$ standard deviation across runs. Relevant min is task error for Rope, qpos error for Reacher, and block pose error for OGBench.}",
            "\\label{tab:planner_sweep_metrics}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    reports = [build_report(sim) for sim in SIMS]
    out_dir = REPO_ROOT / "sweeps" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"reports": reports}
    paths = {
        "json": out_dir / "all_metrics.json",
        "md": out_dir / "all_metrics.md",
        "tex": out_dir / "all_metrics.tex",
    }
    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    paths["md"].write_text(combined_markdown(reports), encoding="utf-8")
    paths["tex"].write_text(combined_latex(reports), encoding="utf-8")
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
