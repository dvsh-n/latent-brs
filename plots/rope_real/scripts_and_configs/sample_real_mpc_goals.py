#!/usr/bin/env python3
"""Sample reusable real-rope MPC start/goal task configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from tqdm.auto import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rope.data_real.rope_real_data_gen import (  # noqa: E402
    HARDWARE_HOME_Q0_DEG,
    HARDWARE_HOME_Q1_DEG,
    TASK_HEIGHT_BOUNDS,
    TASK_REACH_BOUNDS,
    TASK_WIDTH_BOUNDS,
    HomeAnchoredLinePlanner,
    TaskBounds,
    action_durations,
    validate_collision,
    validate_start_reset_collision,
)


DEFAULT_CONFIG = ROOT / "rope" / "data_real" / "rope_real.yaml"
DEFAULT_OUTPUT = ROOT / "rope" / "plan" / "real_mpc_start_goals.json"
DEFAULT_MIN_TASK_DISTANCE_M = 0.3048


def load_yaml_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(config).__name__}.")
    return {str(key).replace("-", "_"): value for key, value in config.items()}


def parser_defaults() -> dict[str, object]:
    return {
        "task_reach_bounds": list(TASK_REACH_BOUNDS),
        "task_height_bounds": list(TASK_HEIGHT_BOUNDS),
        "task_width_bounds": list(TASK_WIDTH_BOUNDS),
        "home_q0_deg": HARDWARE_HOME_Q0_DEG.tolist(),
        "home_q1_deg": HARDWARE_HOME_Q1_DEG.tolist(),
        "plan_retry_attempts": 20,
        "drake_publish_period": 0.005,
        "max_home_x_gap_m": 0.008,
        "max_home_z_gap_m": 0.004,
        "max_real_joint_speed_deg_s": 15.0,
        "max_task_speed_m_s": 0.03,
        "min_action_duration": 0.35,
        "ik_position_tol": 0.005,
        "ik_max_joint_step_deg": 8.0,
        "ik_joint7_min_deg": -140.0,
        "ik_joint7_max_deg": 140.0,
        "disable_collision_guard": False,
        "arm_arm_min_distance": 0.06,
        "collision_control_samples": 5,
        "collision_reset_samples": 25,
        "start_move_duration": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-task-distance-m", type=float, default=DEFAULT_MIN_TASK_DISTANCE_M)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-reach-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--task-height-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--task-width-bounds", type=float, nargs=2, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    defaults = parser_defaults()
    defaults.update({key: value for key, value in config.items() if key in defaults})
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def sample_pair(rng: np.random.Generator, bounds: TaskBounds, min_distance: float) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(1000):
        start = rng.uniform(bounds.lower(), bounds.upper())
        goal = rng.uniform(bounds.lower(), bounds.upper())
        if float(np.linalg.norm(goal - start)) >= min_distance:
            return start, goal
    raise RuntimeError(f"Could not sample a pair at least {min_distance:.3f} m apart inside task bounds.")


def main() -> None:
    args = parse_args()
    if args.num_pairs < 1:
        raise ValueError("--num-pairs must be positive.")
    if args.min_task_distance_m < 0.0:
        raise ValueError("--min-task-distance-m cannot be negative.")

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")

    planner = HomeAnchoredLinePlanner(args)
    bounds = TaskBounds(
        reach=tuple(float(v) for v in args.task_reach_bounds),
        height=tuple(float(v) for v in args.task_height_bounds),
        width=tuple(float(v) for v in args.task_width_bounds),
    )
    rng = np.random.default_rng(args.seed)
    pairs: list[dict[str, object]] = []

    with tqdm(total=args.num_pairs, desc="Sampling start/goal pairs", unit="pair") as progress:
        attempts = 0
        while len(pairs) < args.num_pairs:
            attempts += 1
            start_task, goal_task = sample_pair(rng, bounds, args.min_task_distance_m)
            try:
                task_path = np.stack([start_task, goal_task], axis=0)
                q_path = planner.plan_joint_path(task_path)
                durations = action_durations(q_path, task_path, args)
                label = f"sampled pair {len(pairs)} attempt {attempts}"
                validate_start_reset_collision(planner, q_path[0], args, label=label)
                validate_collision(q_path, durations, args, label=label)
            except RuntimeError as error:
                print(f"Rejected pair attempt {attempts}: {error}")
                continue

            pairs.append(
                {
                    "pair_idx": len(pairs),
                    "task_start": start_task.astype(float).tolist(),
                    "task_goal": goal_task.astype(float).tolist(),
                    "task_distance_m": float(np.linalg.norm(goal_task - start_task)),
                    "q_start": q_path[0].astype(float).tolist(),
                    "q_goal": q_path[-1].astype(float).tolist(),
                    "expert_task_path": task_path.astype(float).tolist(),
                    "expert_q_path": q_path.astype(float).tolist(),
                    "expert_durations_s": durations.astype(float).tolist(),
                    "valid": True,
                }
            )
            progress.update(1)

    payload = {
        "format": "real_rope_mpc_start_goal_pairs_v1",
        "created_unix_time": time.time(),
        "seed": int(args.seed),
        "num_pairs": len(pairs),
        "min_task_distance_m": float(args.min_task_distance_m),
        "task_bounds": {
            "reach": list(bounds.reach),
            "height": list(bounds.height),
            "width": list(bounds.width),
        },
        "planner_config": {
            key: getattr(args, key)
            for key in (
                "home_q0_deg",
                "home_q1_deg",
                "max_real_joint_speed_deg_s",
                "max_task_speed_m_s",
                "min_action_duration",
                "ik_position_tol",
                "ik_max_joint_step_deg",
                "ik_joint7_min_deg",
                "ik_joint7_max_deg",
                "disable_collision_guard",
                "arm_arm_min_distance",
                "collision_control_samples",
                "collision_reset_samples",
            )
        },
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {len(pairs)} start/goal pairs to: {output}")


if __name__ == "__main__":
    main()
