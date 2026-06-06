#!/usr/bin/env python3
"""Check saved rope rollouts with the geometric obstacle rule.

This script intentionally does not use the obstacle classifier or HJ value
function. It replays saved rollout actions in LabEnv and labels each resulting
task state with the reach/low-rope-height rule used by
``rope/data/obs_data_collect.py``.
"""

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

import mujoco
import numpy as np
import torch

from rope.data.obs_data_collect import (
    classify_obstacle_states as classify_half_ellipse_obstacle_states,
    half_ellipse_obstacle_height,
)
from rope.plan import benchmark_rope_hard as hard
from rope.safety.collect_obstacle_data import classify_obstacle_states as classify_cutoff_obstacle_states
from rope.shared.lab_env import LabEnv, TaskState

DEFAULT_OBSTACLE_DATA = REPO_ROOT / "rope/safety/obstacle_data/obstacle_classifier_data.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Closed-loop run directory containing metrics.json.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Override dataset path; defaults to metrics.json.")
    parser.add_argument("--obstacle-data", type=Path, default=DEFAULT_OBSTACLE_DATA)
    parser.add_argument("--branch", choices=("nominal", "hj_filtered", "all"), default="nominal")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_obstacle_rule(path: Path) -> dict[str, Any]:
    artifact = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    metadata = artifact["metadata"]
    sag_profile = artifact["sag_profile"]
    obstacle_reach = np.asarray(metadata["obstacle_reach"], dtype=np.float64).reshape(2)
    has_half_ellipse = str(metadata.get("obstacle_profile", "")) == "half_ellipse" or (
        "obstacle_base_height" in metadata and "obstacle_height" in metadata
    )
    obstacle_base_height = float(metadata.get("obstacle_base_height", metadata.get("table_top_z", 0.75)))
    obstacle_peak_height = float(metadata.get("obstacle_height", metadata.get("low_rope_cutoff", 0.925)))
    low_rope_cutoff = float(metadata.get("low_rope_cutoff", obstacle_peak_height))
    return {
        "path": str(path.expanduser().resolve()),
        "rule_type": "half_ellipse" if has_half_ellipse else "constant_cutoff",
        "obstacle_reach": obstacle_reach,
        "obstacle_base_height": obstacle_base_height,
        "obstacle_peak_height": obstacle_peak_height,
        "low_rope_cutoff": low_rope_cutoff,
        "width_values": np.asarray(sag_profile["width_values"], dtype=np.float64),
        "sag_drop_values": np.asarray(sag_profile["sag_drop"], dtype=np.float64),
        "metadata": metadata,
    }


def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def reset_env_no_render(
    env: LabEnv,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
    elapsed_time: float,
) -> dict[str, np.ndarray]:
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    return hard.extract_step_info(env, elapsed_time=elapsed_time)


def step_env_no_render(
    env: LabEnv,
    *,
    action: np.ndarray,
    control_decimation: int,
    elapsed_time: float,
) -> dict[str, np.ndarray]:
    env.apply_task_delta(np.asarray(action, dtype=np.float64))
    env.step(int(control_decimation))
    return hard.extract_step_info(env, elapsed_time=elapsed_time)


def score_task_states(task_states: np.ndarray, rule: dict[str, Any]) -> dict[str, Any]:
    states = np.asarray(task_states, dtype=np.float64)
    if str(rule.get("rule_type", "half_ellipse")) == "constant_cutoff":
        labels, low_rope_height, sag_drop = classify_cutoff_obstacle_states(
            states,
            obstacle_reach=tuple(rule["obstacle_reach"].tolist()),
            low_rope_cutoff=float(rule["low_rope_cutoff"]),
            width_values=np.asarray(rule["width_values"], dtype=np.float64),
            sag_drop_values=np.asarray(rule["sag_drop_values"], dtype=np.float64),
        )
        profile_height = np.full((states.shape[0],), float(rule["low_rope_cutoff"]), dtype=np.float64)
    else:
        labels, low_rope_height, sag_drop = classify_half_ellipse_obstacle_states(
            states,
            obstacle_reach=tuple(rule["obstacle_reach"].tolist()),
            obstacle_base_height=float(rule["obstacle_base_height"]),
            obstacle_peak_height=float(rule["obstacle_peak_height"]),
            width_values=np.asarray(rule["width_values"], dtype=np.float64),
            sag_drop_values=np.asarray(rule["sag_drop_values"], dtype=np.float64),
        )
        profile_height = half_ellipse_obstacle_height(
            states[:, 0],
            tuple(rule["obstacle_reach"].tolist()),
            float(rule["obstacle_base_height"]),
            float(rule["obstacle_peak_height"]),
        )
    in_reach = (
        (states[:, 0] >= float(rule["obstacle_reach"][0]))
        & (states[:, 0] <= float(rule["obstacle_reach"][1]))
    )
    clearance = low_rope_height - profile_height
    unsafe_idx = np.flatnonzero(labels.astype(bool))
    return {
        "labels": labels.astype(np.int64),
        "low_rope_height": low_rope_height.astype(np.float32),
        "sag_drop": sag_drop.astype(np.float32),
        "profile_height": profile_height.astype(np.float32),
        "in_obstacle_reach": in_reach.astype(bool),
        "clearance_to_obstacle": clearance.astype(np.float32),
        "violates": bool(unsafe_idx.size > 0),
        "unsafe_step_count": int(unsafe_idx.size),
        "first_unsafe_step": None if unsafe_idx.size == 0 else int(unsafe_idx[0]),
        "min_clearance": float(np.min(clearance)) if clearance.size else None,
        "min_reach_window_clearance": float(np.min(clearance[in_reach])) if np.any(in_reach) else None,
    }


def replay_summary(summary_path: Path, *, dataset_path: Path, rule: dict[str, Any], env: LabEnv) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_idx = int(summary["episode_idx"])
    start_step = int(summary["start_step"])
    actions = np.asarray(summary.get("executed_actions_raw", []), dtype=np.float32).reshape(-1, 3)
    episode = hard.load_dataset_episode(dataset_path, episode_idx)
    task_target_np = np.asarray(episode["task_target"], dtype=np.float32)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    control_np = np.asarray(episode["control"], dtype=np.float32)
    time_np = np.asarray(episode["time"], dtype=np.float32)
    control_decimation = int(episode["control_decimation"])
    control_timestep = float(episode["control_timestep"])

    info = reset_env_no_render(
        env,
        qpos=qpos_np[start_step],
        qvel=qvel_np[start_step],
        control=control_np[start_step],
        task_target=task_target_np[start_step],
        elapsed_time=float(time_np[start_step, 0]),
    )
    task_states = [np.asarray(info["task_target"], dtype=np.float32)]
    for step, action in enumerate(actions, start=1):
        elapsed_time = float(info["time"][0]) + control_timestep
        info = step_env_no_render(env, action=action, control_decimation=control_decimation, elapsed_time=elapsed_time)
        task_states.append(np.asarray(info["task_target"], dtype=np.float32))

    states = np.stack(task_states, axis=0)
    scored = score_task_states(states, rule)
    labels = scored.pop("labels")
    low_rope_height = scored.pop("low_rope_height")
    sag_drop = scored.pop("sag_drop")
    profile_height = scored.pop("profile_height")
    in_reach = scored.pop("in_obstacle_reach")
    clearance = scored.pop("clearance_to_obstacle")

    return {
        "summary_path": str(summary_path),
        "label": str(summary.get("label", summary_path.parent.parent.name)),
        "episode_idx": episode_idx,
        "start_step": start_step,
        "steps_replayed": int(actions.shape[0]),
        "classifier_reported_safety_violation": bool(summary.get("safety_violation", False)),
        "classifier_min_l": summary.get("min_l"),
        **scored,
        "task_states": states.tolist(),
        "geometry_labels": labels.tolist(),
        "low_rope_height": low_rope_height.tolist(),
        "sag_drop": sag_drop.tolist(),
        "obstacle_profile_height": profile_height.tolist(),
        "in_obstacle_reach": in_reach.tolist(),
        "clearance_to_obstacle": clearance.tolist(),
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"num_cases": 0}
    violations = np.asarray([case["violates"] for case in cases], dtype=bool)
    unsafe_counts = np.asarray([case["unsafe_step_count"] for case in cases], dtype=np.float64)
    return {
        "num_cases": int(len(cases)),
        "geometry_violation_rate": float(np.mean(violations) * 100.0),
        "geometry_violation_count": int(np.sum(violations)),
        "mean_unsafe_step_count": float(np.mean(unsafe_counts)),
        "violating_cases": [
            {
                "episode_idx": int(case["episode_idx"]),
                "summary_path": str(case["summary_path"]),
                "first_unsafe_step": case["first_unsafe_step"],
                "unsafe_step_count": int(case["unsafe_step_count"]),
                "min_clearance": case["min_clearance"],
                "classifier_reported_safety_violation": bool(case["classifier_reported_safety_violation"]),
                "classifier_min_l": case["classifier_min_l"],
            }
            for case in cases
            if bool(case["violates"])
        ],
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metrics = load_metrics(run_dir)
    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else Path(str(metrics["dataset_path"])).expanduser().resolve()
    )
    rule = load_obstacle_rule(args.obstacle_data)
    branches = ["nominal", "hj_filtered"] if args.branch == "all" else [str(args.branch)]
    env = LabEnv()
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "obstacle_data": str(args.obstacle_data.expanduser().resolve()),
        "obstacle_rule": {
            "rule_type": str(rule["rule_type"]),
            "obstacle_reach": np.asarray(rule["obstacle_reach"]).tolist(),
            "obstacle_base_height": float(rule["obstacle_base_height"]),
            "obstacle_peak_height": float(rule["obstacle_peak_height"]),
            "low_rope_cutoff": float(rule["low_rope_cutoff"]),
        },
        "branches": {},
    }
    for branch in branches:
        summary_paths = sorted((run_dir / branch).glob("case_*/summary.json"))
        cases = [replay_summary(path, dataset_path=dataset_path, rule=rule, env=env) for path in summary_paths]
        result["branches"][branch] = {"aggregate": aggregate(cases), "cases": cases}

    output_path = args.output.expanduser().resolve() if args.output else run_dir / "geometry_obstacle_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"output": str(output_path), "branches": {k: v["aggregate"] for k, v in result["branches"].items()}}), indent=2))


if __name__ == "__main__":
    main()
