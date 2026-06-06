#!/usr/bin/env python3
"""Plot closed-loop rope rollouts on reach vs estimated low-rope height axes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from rope.data.obs_data_collect import half_ellipse_obstacle_height
from rope.plan import benchmark_rope_hard as hard
from rope.shared.lab_env import LabEnv, TABLE_TOP_Z, TaskState


DEFAULT_RUN_DIR = "rope/safety/runs/closed_loop_hj_filter_existing/1779508253_ilqr_hj_seed_42"
DEFAULT_OBSTACLE_DATA = "rope/safety/obstacle_data/obstacle_classifier_data.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--obstacle-data", type=Path, default=Path(DEFAULT_OBSTACLE_DATA))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None, help="Case dir name, e.g. case_0004_episode_11607.")
    parser.add_argument("--show-background-data", action="store_true")
    parser.add_argument("--max-background-points", type=int, default=4000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_obstacle_geometry(path: Path) -> dict[str, Any]:
    artifact = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    metadata = artifact["metadata"]
    dataset = artifact["dataset"]
    sag_profile = artifact["sag_profile"]
    return {
        "metadata": metadata,
        "task_target": np.asarray(dataset["task_target"], dtype=np.float64),
        "label": np.asarray(dataset["label"], dtype=np.int64),
        "low_rope_height": np.asarray(dataset["low_rope_height"], dtype=np.float64),
        "width_values": np.asarray(sag_profile["width_values"], dtype=np.float64),
        "sag_drop": np.asarray(sag_profile["sag_drop"], dtype=np.float64),
    }


def estimate_low_rope_height(task_states: np.ndarray, width_values: np.ndarray, sag_drop: np.ndarray) -> np.ndarray:
    widths = np.asarray(task_states[:, 2], dtype=np.float64)
    estimated_sag = np.interp(widths, width_values, sag_drop)
    return np.asarray(task_states[:, 1], dtype=np.float64) - estimated_sag


def infer_dataset_path(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Need --dataset-path because metrics.json is missing: {metrics_path}")
    metrics = load_json(metrics_path)
    return Path(metrics["dataset_path"]).expanduser().resolve()


def episode_slice(h5: h5py.File, episode_idx: int) -> tuple[int, int]:
    ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
        raise IndexError(f"Episode {episode_idx} is out of range for {ep_len.shape[0]} episodes.")
    start = int(np.sum(ep_len[:episode_idx]))
    stop = start + int(ep_len[episode_idx])
    return start, stop


def load_episode_arrays(dataset_path: Path, episode_idx: int) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as h5:
        start, stop = episode_slice(h5, episode_idx)
        return {
            "qpos": np.asarray(h5["qpos"][start:stop], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][start:stop], dtype=np.float32),
            "control": np.asarray(h5["control"][start:stop], dtype=np.float32),
            "task_target": np.asarray(h5["task_target"][start:stop], dtype=np.float32),
            "time": np.asarray(h5["time"][start:stop], dtype=np.float32),
            "camera": h5.attrs.get("camera", "video_cam"),
            "control_decimation": int(h5.attrs.get("control_decimation", 25)),
            "control_timestep": float(h5.attrs.get("control_timestep", 0.05)),
        }


def reset_env_to_summary_start(env: LabEnv, episode: dict[str, Any], summary: dict[str, Any]) -> dict[str, np.ndarray]:
    start_step = int(summary["start_step"])
    task_target = np.asarray(episode["task_target"][start_step], dtype=np.float64)
    env.reset(TaskState.from_array(task_target))
    qpos = np.asarray(episode["qpos"][start_step], dtype=np.float64)
    qvel = np.asarray(episode["qvel"][start_step], dtype=np.float64)
    control = np.asarray(episode["control"][start_step], dtype=np.float64)
    env.data.qpos[: qpos.shape[0]] = qpos
    env.data.qvel[: qvel.shape[0]] = qvel
    env.joint_controller.set_target(control)
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = control
    mujoco.mj_forward(env.model, env.data)
    elapsed_time = float(np.asarray(episode["time"][start_step]).reshape(-1)[0])
    return hard.extract_step_info(env, elapsed_time=elapsed_time)


def replay_summary_actions(dataset_path: Path, summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    episode = load_episode_arrays(dataset_path, int(summary["episode_idx"]))
    env = LabEnv()
    info = reset_env_to_summary_start(env, episode, summary)
    task_states = [np.asarray(info["task_target"], dtype=np.float32).copy()]

    elapsed_time = float(np.asarray(info["time"]).reshape(-1)[0])
    control_decimation = int(summary.get("control_decimation", episode["control_decimation"]))
    control_timestep = float(summary.get("control_timestep", episode["control_timestep"]))
    for raw_action in summary["executed_actions_raw"]:
        elapsed_time += control_timestep
        env.apply_task_delta(np.asarray(raw_action, dtype=np.float64))
        env.step(control_decimation)
        info = hard.extract_step_info(env, elapsed_time=elapsed_time)
        task_states.append(np.asarray(info["task_target"], dtype=np.float32).copy())

    return {
        "summary": summary,
        "task_states": np.stack(task_states, axis=0),
    }


def discover_cases(run_dir: Path, requested: list[str] | None) -> list[str]:
    nominal_dir = run_dir / "nominal"
    if requested:
        return requested
    return sorted(path.name for path in nominal_dir.glob("case_*") if (path / "summary.json").is_file())


def plot_background(ax: Any, geometry: dict[str, Any], max_points: int) -> None:
    task_states = geometry["task_target"]
    labels = geometry["label"]
    low = geometry["low_rope_height"]
    if task_states.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(task_states.shape[0], size=max_points, replace=False)
        task_states = task_states[indices]
        labels = labels[indices]
        low = low[indices]
    unsafe = labels == 1
    ax.scatter(task_states[~unsafe, 0], low[~unsafe], s=7, c="#009e73", alpha=0.10, edgecolors="none")
    ax.scatter(task_states[unsafe, 0], low[unsafe], s=7, c="#d55e00", alpha=0.13, edgecolors="none")


def add_obstacle_boundary(ax: Any, geometry: dict[str, Any]) -> None:
    metadata = geometry["metadata"]
    obstacle_reach = tuple(float(x) for x in metadata["obstacle_reach"])
    base_height = float(metadata["obstacle_base_height"])
    peak_height = float(metadata["obstacle_height"])
    curve_reach = np.linspace(obstacle_reach[0], obstacle_reach[1], num=300)
    curve_height = half_ellipse_obstacle_height(curve_reach, obstacle_reach, base_height, peak_height)
    ax.fill_between(curve_reach, base_height, curve_height, color="#d55e00", alpha=0.10)
    ax.plot(curve_reach, curve_height, color="#333333", linestyle="--", linewidth=1.2, label="obstacle boundary")
    ax.axhline(float(TABLE_TOP_Z), color="#0072b2", linestyle=":", linewidth=1.0, label="table top")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    dataset_path = infer_dataset_path(run_dir, args.dataset_path)
    geometry = load_obstacle_geometry(args.obstacle_data)
    cases = discover_cases(run_dir, args.case)
    if not cases:
        raise ValueError(f"No case summaries found in {run_dir / 'nominal'}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "closed_loop_reach_height_trajectories.png"
    )

    cols = min(2, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.3 * cols, 5.1 * rows), dpi=180, squeeze=False)
    width_values = geometry["width_values"]
    sag_drop = geometry["sag_drop"]

    for ax, case_name in zip(axes.ravel(), cases, strict=False):
        nominal_path = run_dir / "nominal" / case_name / "summary.json"
        filtered_path = run_dir / "hj_filtered" / case_name / "summary.json"
        if args.show_background_data:
            plot_background(ax, geometry, args.max_background_points)
        add_obstacle_boundary(ax, geometry)

        for label, path, color, linestyle, marker in (
            ("nominal iLQR", nominal_path, "#0072b2", "-", "o"),
            ("HJ filtered", filtered_path, "#cc79a7", "--", "s"),
        ):
            if not path.is_file():
                continue
            rollout = replay_summary_actions(dataset_path, path)
            task_states = rollout["task_states"]
            low = estimate_low_rope_height(task_states, width_values, sag_drop)
            ax.plot(
                task_states[:, 0],
                low,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markersize=2.8,
                markevery=max(1, task_states.shape[0] // 12),
                alpha=0.92,
                label=label,
            )
            ax.scatter(task_states[0, 0], low[0], color=color, marker="*", s=85, edgecolors="black", linewidths=0.35)
            ax.scatter(task_states[-1, 0], low[-1], color=color, marker="X", s=65, edgecolors="black", linewidths=0.35)

        summary = load_json(nominal_path)
        goal = np.asarray(summary["goal_task_target"], dtype=np.float64).reshape(1, 3)
        goal_low = estimate_low_rope_height(goal, width_values, sag_drop)[0]
        ax.scatter(goal[0, 0], goal_low, c="#000000", marker="P", s=70, label="goal")
        ax.set_title(f"{case_name} | episode {summary['episode_idx']}")
        ax.set_xlabel("task reach")
        ax.set_ylabel("estimated low-rope height")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[len(cases) :]:
        ax.axis("off")
    fig.suptitle(f"Closed-loop trajectories: {run_dir.name}", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset_path": str(dataset_path),
                "obstacle_data": str(args.obstacle_data.expanduser().resolve()),
                "cases": cases,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
