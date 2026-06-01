#!/usr/bin/env python3
"""Scripted PushT insertion expert for easy near-slot starts.

The controller is intentionally narrow: it assumes the T starts above the slot,
near the slot center, with only modest yaw error. It emits absolute pusher target
positions for the insertion env defined in ``pusht_insertion_test.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pusht.test.pusht_insertion_test import (
    ENV_FPS,
    WALL_MAX,
    WALL_MIN,
    InsertionGeometry,
    angle_diff,
    get_state,
    make_env,
    success_metrics,
)


DEFAULT_OUT_DIR = Path("pusht/test/pusht_insertion_scripted_expert")


@dataclass(frozen=True)
class ScriptedExpertConfig:
    x_tol: float = 8.0
    y_tol: float = 16.0
    yaw_tol: float = 0.20
    approach_y: float = 66.0
    center_approach_y: float = 42.0
    yaw_side_x: float = 43.0
    yaw_push_y: float = 46.0
    side_contact_x: float = 48.0
    side_push_x: float = 74.0
    side_y: float = -2.0
    insert_push_y: float = 86.0
    waypoint_tol: float = 32.0
    center_drive_x: float = 24.0
    insert_drive_y: float = 28.0
    yaw_drive_y: float = 80.0


def clip_workspace(xy: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(xy, dtype=np.float64), WALL_MIN, WALL_MAX)


def random_easy_geometry(args: argparse.Namespace, rng: np.random.Generator) -> InsertionGeometry:
    return InsertionGeometry(
        buffer=args.buffer,
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        goal_block_y=args.goal_y,
        start_dx=float(rng.uniform(-args.start_dx_range, args.start_dx_range)),
        start_dy=float(rng.uniform(args.start_dy_min, args.start_dy_max)),
        start_dtheta=float(rng.uniform(-args.start_dtheta_range, args.start_dtheta_range)),
        start_agent_y_offset=args.start_agent_y_offset,
    )


class ScriptedInsertionExpert:
    def __init__(self, geometry: InsertionGeometry, config: ScriptedExpertConfig):
        self.geometry = geometry
        self.config = config

    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        agent = np.asarray(state[:2], dtype=np.float64)
        block = np.asarray(state[2:5], dtype=np.float64)
        goal = self.geometry.goal_block_pose

        bx, by, theta = float(block[0]), float(block[1]), float(block[2])
        gx, gy, gtheta = float(goal[0]), float(goal[1]), float(goal[2])
        x_err = bx - gx
        y_err = by - gy
        yaw_err = angle_diff(theta, gtheta)

        if abs(yaw_err) > cfg.yaw_tol:
            action, target, phase = self._yaw_action(agent, bx, by, yaw_err)
        elif abs(x_err) > cfg.x_tol:
            action, target, phase = self._center_action(agent, bx, by, x_err)
        else:
            action, target, phase = self._insert_action(agent, bx, by, gx, gy)

        meta = {
            "phase": phase,
            "target_xy": target.tolist(),
            "action": action.tolist(),
            "block_pose": block.tolist(),
            "agent_xy": agent.tolist(),
            "x_error": float(x_err),
            "y_error": float(y_err),
            "yaw_error": float(yaw_err),
        }
        return action.astype(np.float32), meta

    def _approach_or_push(
        self,
        agent: np.ndarray,
        pre_contact: np.ndarray,
        push_target: np.ndarray,
        *,
        approach_phase: str,
        push_phase: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        pre_contact = clip_workspace(pre_contact)
        push_target = clip_workspace(push_target)
        if float(np.linalg.norm(agent - pre_contact)) > cfg.waypoint_tol:
            return pre_contact, pre_contact, approach_phase
        return push_target, push_target, push_phase

    def _yaw_action(self, agent: np.ndarray, bx: float, by: float, yaw_err: float) -> tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        # Positive yaw is corrected by pushing downward on the left side; negative
        # yaw uses the mirrored contact. This is a local heuristic, not a general
        # manipulation policy.
        side = -np.sign(yaw_err) if abs(yaw_err) > 1e-6 else -1.0
        contact_x = bx + side * cfg.yaw_side_x
        pre_contact = np.asarray([contact_x, by - cfg.approach_y], dtype=np.float64)
        push_target = np.asarray([contact_x, by - cfg.approach_y + cfg.yaw_drive_y], dtype=np.float64)
        return self._approach_or_push(
            agent,
            pre_contact,
            push_target,
            approach_phase="approach_yaw_contact",
            push_phase="drag_yaw",
        )

    def _center_action(self, agent: np.ndarray, bx: float, by: float, x_err: float) -> tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        # If the block is right of the slot, contact from the right and push left;
        # if it is left of the slot, mirror the maneuver.
        side = np.sign(x_err) if abs(x_err) > 1e-6 else 1.0
        pre_contact = np.asarray([bx + side * cfg.side_contact_x, by + cfg.side_y], dtype=np.float64)
        push_target = np.asarray([bx + side * (cfg.side_contact_x - cfg.center_drive_x), by + cfg.side_y], dtype=np.float64)
        return self._approach_or_push(
            agent,
            pre_contact,
            push_target,
            approach_phase="approach_center_contact",
            push_phase="drag_center",
        )

    def _insert_action(self, agent: np.ndarray, bx: float, by: float, gx: float, gy: float) -> tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        pre_contact = np.asarray([gx, by - cfg.center_approach_y], dtype=np.float64)
        push_y = max(by - cfg.center_approach_y + cfg.insert_drive_y, gy + cfg.y_tol)
        push_target = np.asarray([gx, push_y], dtype=np.float64)
        return self._approach_or_push(
            agent,
            pre_contact,
            push_target,
            approach_phase="approach_insert_contact",
            push_phase="drag_insert",
        )


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)


def rollout_episode(args: argparse.Namespace, episode_idx: int, rng: np.random.Generator) -> dict[str, Any]:
    geometry = random_easy_geometry(args, rng)
    config = ScriptedExpertConfig(
        x_tol=args.x_tol,
        y_tol=args.y_tol,
        yaw_tol=args.yaw_tol,
        approach_y=args.approach_y,
        center_approach_y=args.center_approach_y,
        yaw_side_x=args.yaw_side_x,
        yaw_push_y=args.yaw_push_y,
        side_contact_x=args.side_contact_x,
        side_push_x=args.side_push_x,
        side_y=args.side_y,
        insert_push_y=args.insert_push_y,
        waypoint_tol=args.waypoint_tol,
        center_drive_x=args.center_drive_x,
        insert_drive_y=args.insert_drive_y,
        yaw_drive_y=args.yaw_drive_y,
    )
    expert = ScriptedInsertionExpert(geometry, config)
    env = make_env(geometry, render_size=args.render_size, obs_size=args.obs_size)

    frames: list[np.ndarray] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    step_meta: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    try:
        env.reset(seed=args.seed + episode_idx)
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        states.append(get_state(env).tolist())

        for step_idx in range(args.max_steps):
            state = get_state(env)
            action, meta = expert.select_action(state)
            _, _, _, _, info = env.step(action)

            next_state = get_state(env)
            metrics = success_metrics(next_state, geometry.goal_block_pose)
            meta.update(
                {
                    "step": step_idx,
                    "n_contacts": int(info.get("n_contacts", 0)),
                    "metrics": metrics,
                }
            )
            actions.append(action.astype(float).tolist())
            step_meta.append(meta)
            states.append(next_state.tolist())

            if (step_idx + 1) % args.save_every == 0 or metrics["success"]:
                frames.append(np.asarray(env.render(), dtype=np.uint8))
            if metrics["success"]:
                break

        if not metrics:
            metrics = success_metrics(get_state(env), geometry.goal_block_pose)

        episode_dir = args.out_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_path = episode_dir / "scripted_expert.mp4"
        save_video(video_path, frames, fps=args.fps)

        npz_path = episode_dir / "trajectory.npz"
        np.savez_compressed(
            npz_path,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            goal_block_pose=geometry.goal_block_pose.astype(np.float32),
            start_state=geometry.start_state.astype(np.float32),
        )

        summary = {
            "episode": episode_idx,
            "seed": args.seed + episode_idx,
            "steps": len(actions),
            "success": bool(metrics["success"]),
            "metrics": metrics,
            "geometry": asdict(geometry),
            "controller": asdict(config),
            "video_path": str(video_path),
            "trajectory_path": str(npz_path),
            "steps_meta": step_meta,
        }
        with (episode_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--obs-size", type=int, default=96)
    parser.add_argument("--fps", type=int, default=ENV_FPS)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--buffer", type=float, default=25.0)
    parser.add_argument("--obstacle-width", type=float, default=512.0 / 4.0)
    parser.add_argument("--obstacle-height", type=float, default=135.0)
    parser.add_argument("--goal-y", type=float, default=330.0)
    parser.add_argument("--start-dx-range", type=float, default=18.0)
    parser.add_argument("--start-dy-min", type=float, default=-130.0)
    parser.add_argument("--start-dy-max", type=float, default=-95.0)
    parser.add_argument("--start-dtheta-range", type=float, default=0.12)
    parser.add_argument("--start-agent-y-offset", type=float, default=-70.0)

    parser.add_argument("--x-tol", type=float, default=8.0)
    parser.add_argument("--y-tol", type=float, default=16.0)
    parser.add_argument("--yaw-tol", type=float, default=0.20)
    parser.add_argument("--approach-y", type=float, default=66.0)
    parser.add_argument("--center-approach-y", type=float, default=42.0)
    parser.add_argument("--yaw-side-x", type=float, default=43.0)
    parser.add_argument("--yaw-push-y", type=float, default=46.0)
    parser.add_argument("--side-contact-x", type=float, default=48.0)
    parser.add_argument("--side-push-x", type=float, default=74.0)
    parser.add_argument("--side-y", type=float, default=-2.0)
    parser.add_argument("--insert-push-y", type=float, default=86.0)
    parser.add_argument("--waypoint-tol", type=float, default=32.0)
    parser.add_argument("--center-drive-x", type=float, default=24.0)
    parser.add_argument("--insert-drive-y", type=float, default=28.0)
    parser.add_argument("--yaw-drive-y", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1.")
    if args.center_drive_x < 0.0:
        raise ValueError("--center-drive-x must be >= 0.")
    if args.insert_drive_y < 0.0:
        raise ValueError("--insert-drive-y must be >= 0.")
    if args.yaw_drive_y < 0.0:
        raise ValueError("--yaw-drive-y must be >= 0.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries = []
    for episode_idx in tqdm(range(args.episodes), desc="scripted insertion expert", unit="episode"):
        summaries.append(rollout_episode(args, episode_idx, rng))

    aggregate = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "episodes": summaries,
        "success_rate": float(np.mean([episode["success"] for episode in summaries])),
        "mean_steps": float(np.mean([episode["steps"] for episode in summaries])),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(
        json.dumps(
            {
                "episodes": len(summaries),
                "success_rate": aggregate["success_rate"],
                "mean_steps": aggregate["mean_steps"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
