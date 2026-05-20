#!/usr/bin/env python3
"""Debug local obstacle perturbations around a saved nominal Reacher rollout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from reacher.plan import obstacle_net_train as obstacle_train
from reacher.plan import plan_ilqr_mpc as planner

DEFAULT_ROLLOUT_DIR = "reacher/plan/ilqr_mpc_mlpdyn/1779300191_episode_00163"
DEFAULT_ROLLOUT_PATH = str(Path(DEFAULT_ROLLOUT_DIR) / "nominal_rollout.pt")
DEFAULT_OUT_PATH = str(Path(DEFAULT_ROLLOUT_DIR) / "debug_obstacle_overlay.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-path", type=Path, default=Path(DEFAULT_ROLLOUT_PATH))
    parser.add_argument("--out-path", type=Path, default=Path(DEFAULT_OUT_PATH))
    parser.add_argument("--obstacle-step", type=int, default=-1)
    parser.add_argument("--joint1-range", type=float, default=0.25)
    parser.add_argument("--joint2-range", type=float, default=0.15)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_debug_perturbations(
    center_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    joint_ranges: np.ndarray,
    count: int,
) -> np.ndarray:
    noise = rng.uniform(-joint_ranges, joint_ranges, size=(count, center_qpos.shape[0]))
    sampled = center_qpos[None, :] + noise

    # Some Reacher joints are effectively angular coordinates with degenerate
    # MuJoCo jnt_range entries (for example [0, 0]). For those, wrap instead
    # of clipping so the samples stay local to the nominal configuration.
    valid_width = np.isfinite(lower) & np.isfinite(upper) & ((upper - lower) > 1e-8)
    if np.any(valid_width):
        sampled[:, valid_width] = np.clip(
            sampled[:, valid_width],
            lower[None, valid_width],
            upper[None, valid_width],
        )
    if np.any(~valid_width):
        sampled[:, ~valid_width] = ((sampled[:, ~valid_width] + np.pi) % (2.0 * np.pi)) - np.pi
    return sampled.astype(np.float64)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rollout = obstacle_train.load_nominal_rollout(args.rollout_path.expanduser().resolve())
    rollout_qpos = np.asarray(rollout["rollout_qpos"], dtype=np.float64)

    obstacle_step = int(args.obstacle_step)
    if obstacle_step == -1:
        obstacle_step = int(rollout_qpos.shape[0] - 1)
    if obstacle_step < 0 or obstacle_step >= rollout_qpos.shape[0]:
        raise ValueError(f"--obstacle-step must be in [0, {rollout_qpos.shape[0] - 1}], got {args.obstacle_step}.")

    env = planner.make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    try:
        lower, upper = obstacle_train.joint_limits_from_env(env)
    finally:
        env.close()

    center_qpos = rollout_qpos[obstacle_step]
    joint_ranges = np.array([float(args.joint1_range), float(args.joint2_range)], dtype=np.float64)
    obstacle_qpos = sample_debug_perturbations(
        center_qpos,
        lower,
        upper,
        rng,
        joint_ranges=joint_ranges,
        count=int(args.sample_count),
    )

    obstacle_train.save_obstacle_overlay(
        planner_module=planner,
        rollout=rollout,
        center_qpos=center_qpos,
        obstacle_qpos=obstacle_qpos,
        out_path=args.out_path.expanduser().resolve(),
    )

    print(f"Rollout path: {args.rollout_path.expanduser().resolve()}")
    print(f"Obstacle step: {obstacle_step}")
    print(f"Center qpos: {center_qpos.tolist()}")
    print(f"Lower limits: {lower.tolist()}")
    print(f"Upper limits: {upper.tolist()}")
    print(f"Joint ranges: {joint_ranges.tolist()}")
    print(f"Samples shape: {obstacle_qpos.shape}")
    print(f"Output image: {args.out_path.expanduser().resolve()}")
    print("Sampled qpos:")
    for idx, qpos in enumerate(obstacle_qpos):
        print(f"  [{idx:02d}] {qpos.tolist()}")


if __name__ == "__main__":
    main()
