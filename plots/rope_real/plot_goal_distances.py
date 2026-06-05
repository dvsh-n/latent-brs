#!/usr/bin/env python3
"""Plot rope-real task and latent goal distance curves for selected MPC windows."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 300,
    }
)


ROOT = Path(__file__).resolve().parent
RUNS = [
    {
        "metrics": ROOT / "1780621790_shard_0075_waypoint_017" / "metrics.json",
        "max_step": 55,
    },
    {
        "metrics": ROOT / "1780622107_shard_0075_waypoint_002" / "metrics.json",
        "max_step": 40,
    },
]
OUT_PATH = ROOT / "rope_real_goal_distances.pdf"


def load_distances(path: Path, max_step: int) -> tuple[list[float], list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    task_distances = metrics["goal_distances_m"]
    latent_distances = metrics["latent_goal_distances"]
    return (
        [float(value) for value in task_distances[: max_step + 1]],
        [float(value) for value in latent_distances[: max_step + 1]],
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.8), sharex="col", constrained_layout=True)

    for col, run in enumerate(RUNS):
        task_distances, latent_distances = load_distances(run["metrics"], int(run["max_step"]))
        task_steps = list(range(len(task_distances)))
        latent_steps = list(range(len(latent_distances)))

        task_ax = axes[0, col]
        latent_ax = axes[1, col]

        task_ax.plot(task_steps, task_distances, color="#1f77b4", linewidth=2.2)
        task_ax.set_ylabel(r"Task error (m)", fontsize=14)
        task_ax.tick_params(axis="both", which="major", labelsize=12)
        task_ax.grid(True, color="0.86", linewidth=0.8)
        task_ax.set_xlim(0, int(run["max_step"]))

        latent_ax.plot(latent_steps, latent_distances, color="#d62728", linewidth=2.2)
        latent_ax.set_xlabel(r"MPC steps", fontsize=14)
        latent_ax.set_ylabel(r"Latent error", fontsize=14)
        latent_ax.tick_params(axis="both", which="major", labelsize=12)
        latent_ax.grid(True, color="0.86", linewidth=0.8)
        latent_ax.set_xlim(0, int(run["max_step"]))

    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
