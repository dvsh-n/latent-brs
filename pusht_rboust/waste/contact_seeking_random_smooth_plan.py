#!/usr/bin/env python3
"""Visualize contact-seeking smooth-random PushT rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from pusht_rboust.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
    set_pusht_state,
)
from pusht_rboust.shared.utils import render_frame

DEFAULT_OUT_DIR = Path("pusht_rboust/plan/contact_seeking_random_smooth_plan")
ARENA_MIN = 32.0
ARENA_MAX = 480.0
ENV_ACTION_SCALE = 100.0
DEFAULT_MAX_STEPS = 150
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
TEE_BAR_X_MIN = -TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_X_MAX = TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_Y_MIN = 0.0
TEE_BAR_Y_MAX = TEE_SCALE
TEE_STEM_X_MIN = -TEE_SCALE / 2.0
TEE_STEM_X_MAX = TEE_SCALE / 2.0
TEE_STEM_Y_MIN = TEE_SCALE
TEE_STEM_Y_MAX = TEE_LENGTH * TEE_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--video-name", default="contact_seeking_random_smooth_plan.mp4")
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Update the controller every N env steps and hold the last action in between.",
    )
    parser.add_argument("--contact-tol", type=float, default=14.0)
    parser.add_argument("--waypoint-tol", type=float, default=18.0)
    parser.add_argument("--approach-offset-min", type=float, default=55.0)
    parser.add_argument("--approach-offset-max", type=float, default=95.0)
    parser.add_argument("--push-step-min", type=float, default=45.0)
    parser.add_argument("--push-step-max", type=float, default=85.0)
    parser.add_argument("--push-angle-jitter-deg", type=float, default=20.0)
    parser.add_argument("--push-hold-min", type=int, default=3)
    parser.add_argument("--push-hold-max", type=int, default=8)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=0.18)
    parser.add_argument("--max-action-delta", type=float, default=80.0)
    return parser.parse_args()


def _clip_xy(xy: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(xy, dtype=np.float32), ARENA_MIN, ARENA_MAX)


def _sample_xy(rng: np.random.Generator, *, margin: float = 48.0) -> np.ndarray:
    return rng.uniform(ARENA_MIN + margin, ARENA_MAX - margin, size=(2,)).astype(np.float32)


def _sample_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(1024):
        agent_xy = _sample_xy(rng, margin=56.0)
        block_xy = _sample_xy(rng, margin=96.0)
        if np.linalg.norm(agent_xy - block_xy) >= 90.0:
            theta = float(rng.uniform(-np.pi, np.pi))
            state = np.asarray(
                [agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], theta, 0.0, 0.0],
                dtype=np.float64,
            )
            return state, np.asarray([block_xy[0], block_xy[1], theta], dtype=np.float32)
    raise RuntimeError("Failed to sample a valid random PushT state.")


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _sample_point_on_t(block_pose: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | str]]:
    bar_area = (TEE_BAR_X_MAX - TEE_BAR_X_MIN) * (TEE_BAR_Y_MAX - TEE_BAR_Y_MIN)
    stem_area = (TEE_STEM_X_MAX - TEE_STEM_X_MIN) * (TEE_STEM_Y_MAX - TEE_STEM_Y_MIN)
    if float(rng.uniform()) < bar_area / (bar_area + stem_area):
        local_xy = np.asarray(
            [
                rng.uniform(TEE_BAR_X_MIN, TEE_BAR_X_MAX),
                rng.uniform(TEE_BAR_Y_MIN, TEE_BAR_Y_MAX),
            ],
            dtype=np.float32,
        )
        region = "bar"
    else:
        local_xy = np.asarray(
            [
                rng.uniform(TEE_STEM_X_MIN, TEE_STEM_X_MAX),
                rng.uniform(TEE_STEM_Y_MIN, TEE_STEM_Y_MAX),
            ],
            dtype=np.float32,
        )
        region = "stem"
    world_xy = (_rotation_matrix(float(block_pose[2])) @ local_xy) + block_pose[:2]
    return world_xy.astype(np.float32), {"region": region, "local_x": float(local_xy[0]), "local_y": float(local_xy[1])}


def _get_video_fps(env: Any, control_interval: int) -> int:
    metadata = getattr(env, "metadata", None) or getattr(getattr(env, "unwrapped", None), "metadata", None) or {}
    base_fps = metadata.get("render_fps")
    if base_fps is None:
        dt = getattr(getattr(env, "unwrapped", None), "dt", None)
        if dt is not None and dt > 1e-6:
            base_fps = float(round(1.0 / dt))
    if base_fps is None:
        base_fps = 10.0
    return max(1, int(round(float(base_fps) / float(control_interval))))


def _target_xy_to_env_action(env: Any, agent_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        high = np.asarray(action_space.high)
        low = np.asarray(action_space.low)
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)
    return target_xy.astype(np.float32)


def _pd_target(
    agent_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    prev_error: np.ndarray,
    *,
    kp: float,
    kd: float,
    max_action_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    error = waypoint_xy - agent_xy
    correction = kp * error + kd * (error - prev_error)
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_action_delta and correction_norm > 1e-6:
        correction = correction * (max_action_delta / correction_norm)
    return agent_xy + correction, error


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(info["is_success"])
    if "success" in info:
        return bool(info["success"])
    return bool(terminated or reward >= 0.95)


@dataclass
class ContactSeekingRandomSmoothController:
    contact_xy: np.ndarray
    approach_xy: np.ndarray
    push_dir: np.ndarray
    push_step_min: float
    push_step_max: float
    push_angle_jitter_rad: float
    push_hold_min: int
    push_hold_max: int
    waypoint_tol: float
    contact_tol: float
    kp: float
    kd: float
    max_action_delta: float
    rng: np.random.Generator
    prev_error: np.ndarray
    push_action_xy: np.ndarray | None = None
    push_hold_remaining: int = 0
    phase: str = "approach"

    @classmethod
    def from_state(
        cls,
        *,
        agent_xy: np.ndarray,
        block_pose: np.ndarray,
        rng: np.random.Generator,
        approach_offset_min: float,
        approach_offset_max: float,
        push_step_min: float,
        push_step_max: float,
        push_angle_jitter_deg: float,
        push_hold_min: int,
        push_hold_max: int,
        waypoint_tol: float,
        contact_tol: float,
        kp: float,
        kd: float,
        max_action_delta: float,
    ) -> "ContactSeekingRandomSmoothController":
        if approach_offset_max < approach_offset_min:
            raise ValueError("--approach-offset-max must be >= --approach-offset-min.")
        if push_step_max < push_step_min:
            raise ValueError("--push-step-max must be >= --push-step-min.")
        if push_hold_min < 1 or push_hold_max < push_hold_min:
            raise ValueError("Push hold bounds must satisfy 1 <= min <= max.")

        for _ in range(1024):
            contact_xy, _ = _sample_point_on_t(block_pose, rng)
            push_angle = float(rng.uniform(-np.pi, np.pi))
            push_dir = np.asarray([np.cos(push_angle), np.sin(push_angle)], dtype=np.float32)
            offset = float(rng.uniform(approach_offset_min, approach_offset_max))
            approach_xy = _clip_xy(contact_xy - push_dir * offset)
            if np.linalg.norm(approach_xy - contact_xy) >= 0.7 * approach_offset_min:
                return cls(
                    contact_xy=contact_xy.astype(np.float32),
                    approach_xy=approach_xy.astype(np.float32),
                    push_dir=push_dir.astype(np.float32),
                    push_step_min=float(push_step_min),
                    push_step_max=float(push_step_max),
                    push_angle_jitter_rad=float(np.deg2rad(push_angle_jitter_deg)),
                    push_hold_min=int(push_hold_min),
                    push_hold_max=int(push_hold_max),
                    waypoint_tol=float(waypoint_tol),
                    contact_tol=float(contact_tol),
                    kp=float(kp),
                    kd=float(kd),
                    max_action_delta=float(max_action_delta),
                    rng=rng,
                    prev_error=np.zeros(2, dtype=np.float32),
                )
        raise RuntimeError("Failed to sample a valid contact-seeking controller setup.")

    def _sample_push_waypoint(self, agent_xy: np.ndarray) -> np.ndarray:
        if self.push_action_xy is None or self.push_hold_remaining <= 0:
            angle = float(np.arctan2(self.push_dir[1], self.push_dir[0]))
            angle += float(self.rng.uniform(-self.push_angle_jitter_rad, self.push_angle_jitter_rad))
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            step_size = float(self.rng.uniform(self.push_step_min, self.push_step_max))
            self.push_action_xy = _clip_xy(agent_xy + direction * step_size)
            self.push_hold_remaining = int(self.rng.integers(self.push_hold_min, self.push_hold_max + 1))
        self.push_hold_remaining -= 1
        return self.push_action_xy.copy()

    def select_action(self, agent_xy: np.ndarray, block_pose: np.ndarray) -> tuple[np.ndarray, str]:
        if self.phase == "approach":
            if np.linalg.norm(agent_xy - self.approach_xy) <= self.waypoint_tol:
                self.phase = "contact"
            else:
                target_xy, self.prev_error = _pd_target(
                    agent_xy,
                    self.approach_xy,
                    self.prev_error,
                    kp=self.kp,
                    kd=self.kd,
                    max_action_delta=self.max_action_delta,
                )
                return _clip_xy(target_xy), self.phase

        if self.phase == "contact":
            if np.linalg.norm(agent_xy - block_pose[:2]) <= self.contact_tol:
                self.phase = "push"
            else:
                target_xy, self.prev_error = _pd_target(
                    agent_xy,
                    self.contact_xy,
                    self.prev_error,
                    kp=self.kp,
                    kd=self.kd,
                    max_action_delta=self.max_action_delta,
                )
                return _clip_xy(target_xy), self.phase

        self.phase = "push"
        target_xy, self.prev_error = _pd_target(
            agent_xy,
            self._sample_push_waypoint(agent_xy),
            self.prev_error,
            kp=self.kp,
            kd=self.kd,
            max_action_delta=self.max_action_delta,
        )
        return _clip_xy(target_xy), self.phase


def rollout_episode(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")

    episode_seed = None if args.seed is None else args.seed + episode_idx
    rng = np.random.default_rng(episode_seed)
    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    try:
        env.reset(seed=episode_seed)
        state, block_pose = _sample_state(rng)
        set_pusht_state(env.unwrapped, state)

        controller = ContactSeekingRandomSmoothController.from_state(
            agent_xy=get_pusht_agent_pos(env),
            block_pose=block_pose,
            rng=rng,
            approach_offset_min=args.approach_offset_min,
            approach_offset_max=args.approach_offset_max,
            push_step_min=args.push_step_min,
            push_step_max=args.push_step_max,
            push_angle_jitter_deg=args.push_angle_jitter_deg,
            push_hold_min=args.push_hold_min,
            push_hold_max=args.push_hold_max,
            waypoint_tol=args.waypoint_tol,
            contact_tol=args.contact_tol,
            kp=args.kp,
            kd=args.kd,
            max_action_delta=args.max_action_delta,
        )

        frames = [render_frame(env)]
        agent_positions = [get_pusht_agent_pos(env).tolist()]
        block_poses = [get_pusht_block_pose(env).tolist()]
        rewards: list[float] = []
        phase_trace: list[str] = [controller.phase]
        success = False
        contacted_block = False
        terminated = False
        truncated = False
        control_updates = 0
        push_updates = 0
        action = None

        for step_idx in range(args.max_steps):
            agent_xy = get_pusht_agent_pos(env)
            block_pose_now = get_pusht_block_pose(env)
            if np.linalg.norm(agent_xy - block_pose_now[:2]) <= args.contact_tol:
                contacted_block = True

            if action is None or step_idx % args.control_interval == 0:
                target_xy, phase = controller.select_action(agent_xy, block_pose_now)
                action = _target_xy_to_env_action(env, agent_xy, target_xy)
                control_updates += 1
                if phase == "push":
                    push_updates += 1

            _, reward, terminated, truncated, info = env.step(action)
            success = _extract_success(terminated, float(reward), info)

            if (step_idx + 1) % args.control_interval == 0 or terminated or truncated:
                frames.append(render_frame(env))
                agent_positions.append(get_pusht_agent_pos(env).tolist())
                block_poses.append(get_pusht_block_pose(env).tolist())
                rewards.append(float(reward))
                phase_trace.append(controller.phase)

            if terminated or truncated:
                break

        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            video_path,
            frames,
            fps=_get_video_fps(env, args.control_interval),
            quality=8,
            macro_block_size=1,
        )

        final_block_pose = get_pusht_block_pose(env)
        final_agent_xy = get_pusht_agent_pos(env)
        goal_pose = np.asarray(env.unwrapped.goal_pose, dtype=np.float32).copy()
        block_goal_distance = float(np.linalg.norm(final_block_pose[:2] - goal_pose[:2]))
        return {
            "episode": episode_idx,
            "seed": episode_seed,
            "start_agent_xy": state[:2].tolist(),
            "start_block_pose": block_pose.tolist(),
            "goal_block_pose": goal_pose.tolist(),
            "contact_xy": controller.contact_xy.tolist(),
            "approach_xy": controller.approach_xy.tolist(),
            "push_dir": controller.push_dir.tolist(),
            "final_agent_xy": final_agent_xy.tolist(),
            "final_block_pose": final_block_pose.tolist(),
            "contacted_block": contacted_block,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "env_steps": step_idx + 1 if (terminated or truncated) else args.max_steps,
            "stored_steps": len(frames),
            "control_updates": control_updates,
            "push_updates": push_updates,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "block_goal_distance": block_goal_distance,
            "phase_trace": phase_trace,
            "video_path": str(video_path),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [rollout_episode(args, episode_idx) for episode_idx in range(args.episodes)]
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} env_steps={result['env_steps']} "
            f"stored_steps={result['stored_steps']} control_updates={result['control_updates']} "
            f"push_updates={result['push_updates']} contacted_block={result['contacted_block']} "
            f"success={result['success']} block_goal_distance={result['block_goal_distance']:.3f} "
            f"video={result['video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
