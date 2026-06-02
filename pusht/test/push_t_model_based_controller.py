#!/usr/bin/env python3
"""Model-based PushT insertion controller in the real env coordinate system.

This is a non-learning MPC-style baseline for ``pusht_insertion_test.py``.  It
uses a lightweight analytic contact model to choose among contact-preserving push
primitives, emits the absolute pusher target expected by ``PushTEnv.step()``, and
can validate its predicted rollout against a shadow pymunk environment.
"""

from __future__ import annotations

import argparse
import json
import math
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
    AGENT_RADIUS,
    ENV_FPS,
    TEE_LENGTH,
    TEE_SCALE,
    WALL_MAX,
    WALL_MIN,
    InsertionGeometry,
    angle_diff,
    get_state,
    make_env,
    set_state,
    success_metrics,
)


DEFAULT_OUT_DIR = Path("pusht/test/push_t_model_based_controller")


Array = np.ndarray


def rot2(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def clip_workspace(xy: Array) -> Array:
    return np.clip(np.asarray(xy, dtype=np.float64), WALL_MIN, WALL_MAX)


def clamp_norm(v: Array, max_norm: float) -> Array:
    norm = float(np.linalg.norm(v))
    if norm <= max_norm or norm < 1e-9:
        return v
    return v * (max_norm / norm)


@dataclass(frozen=True)
class RectPart:
    center: Array
    half_extents: Array


@dataclass(frozen=True)
class RealPushTShape:
    """Real PushT tee geometry from ``gym_pusht.envs.pusht.PushTEnv.add_tee``."""

    parts: tuple[RectPart, ...]

    @staticmethod
    def from_gym_pusht(scale: float = TEE_SCALE, length: float = TEE_LENGTH) -> "RealPushTShape":
        cap = RectPart(
            center=np.asarray([0.0, 0.5 * scale], dtype=np.float64),
            half_extents=np.asarray([0.5 * length * scale, 0.5 * scale], dtype=np.float64),
        )
        stem = RectPart(
            center=np.asarray([0.0, 0.5 * (length + 1.0) * scale], dtype=np.float64),
            half_extents=np.asarray([0.5 * scale, 0.5 * (length - 1.0) * scale], dtype=np.float64),
        )
        return RealPushTShape(parts=(cap, stem))

    def signed_distance_and_normal_local(self, q_local: Array) -> tuple[float, Array, Array]:
        best_dist = float("inf")
        best_normal = np.asarray([1.0, 0.0], dtype=np.float64)
        best_closest = q_local.copy()

        for part in self.parts:
            p = q_local - part.center
            h = part.half_extents
            d = np.abs(p) - h

            if np.any(d > 0.0):
                closest_rel = np.clip(p, -h, h)
                closest = part.center + closest_rel
                normal_vec = q_local - closest
                normal_norm = float(np.linalg.norm(normal_vec))
                normal = normal_vec / normal_norm if normal_norm > 1e-9 else np.asarray([1.0, 0.0])
                dist = float(np.linalg.norm(np.maximum(d, 0.0)))
            else:
                face_gaps = h - np.abs(p)
                axis = int(np.argmin(face_gaps))
                sign = 1.0 if p[axis] >= 0.0 else -1.0
                normal = np.zeros(2, dtype=np.float64)
                normal[axis] = sign
                closest = q_local.copy()
                closest[axis] = part.center[axis] + sign * h[axis]
                dist = -float(face_gaps[axis])

            if dist < best_dist:
                best_dist = dist
                best_normal = normal
                best_closest = closest

        return best_dist, best_normal, best_closest

    def world_contact_query(self, block_xy: Array, theta: float, pusher_xy: Array) -> tuple[float, Array, Array]:
        R = rot2(theta)
        q_local = R.T @ (pusher_xy - block_xy)
        dist, normal_local, closest_local = self.signed_distance_and_normal_local(q_local)
        return dist, R @ normal_local, block_xy + R @ closest_local

    def world_vertices(self, block_xy: Array, theta: float) -> Array:
        R = rot2(theta)
        vertices: list[Array] = []
        for part in self.parts:
            h = part.half_extents
            for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
                local = part.center + np.asarray([sx * h[0], sy * h[1]], dtype=np.float64)
                vertices.append(block_xy + R @ local)
        return np.stack(vertices, axis=0)


@dataclass(frozen=True)
class PushTInsertionState:
    agent_xy: Array
    agent_vel: Array
    block_xy: Array
    block_theta: float

    @staticmethod
    def from_env_state(state: Array, agent_vel: Array | None = None) -> "PushTInsertionState":
        state = np.asarray(state, dtype=np.float64)
        return PushTInsertionState(
            agent_xy=state[:2].copy(),
            agent_vel=np.zeros(2, dtype=np.float64) if agent_vel is None else np.asarray(agent_vel, dtype=np.float64),
            block_xy=state[2:4].copy(),
            block_theta=float(state[4]),
        )

    def as_env_state(self) -> Array:
        return np.asarray(
            [self.agent_xy[0], self.agent_xy[1], self.block_xy[0], self.block_xy[1], self.block_theta],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class InsertionMPCConfig:
    horizon: int = 12
    max_target_step: float = 58.0
    pusher_speed_gain: float = 0.55
    contact_margin: float = 5.0
    normal_push_gain: float = 0.52
    tangential_push_gain: float = 0.08
    angular_gain: float = 0.0018
    object_step_limit: float = 18.0
    omega_step_limit: float = 0.11

    approach_y: float = 66.0
    center_approach_y: float = 42.0
    yaw_side_x: float = 43.0
    side_contact_x: float = 48.0
    side_y: float = -2.0
    center_drive_x: float = 24.0
    insert_drive_y: float = 28.0
    yaw_drive_y: float = 80.0

    w_center_x: float = 1.6
    w_y_goal: float = 1.0
    w_y_progress: float = 0.25
    w_yaw: float = 2600.0
    w_terminal: float = 8.0
    w_contact: float = 0.18
    w_obstacle: float = 18.0
    w_action_delta: float = 0.006
    obstacle_margin: float = 10.0


@dataclass(frozen=True)
class PushPrimitive:
    name: str
    mode: str
    side: float = 1.0


@dataclass(frozen=True)
class InsertionObstacle:
    center: Array
    half_extents: Array

    @staticmethod
    def from_center_size(center: tuple[float, float], width: float, height: float) -> "InsertionObstacle":
        return InsertionObstacle(
            center=np.asarray(center, dtype=np.float64),
            half_extents=np.asarray([0.5 * width, 0.5 * height], dtype=np.float64),
        )


class PixelInsertionModel:
    def __init__(self, geometry: InsertionGeometry, config: InsertionMPCConfig):
        self.geometry = geometry
        self.config = config
        self.shape = RealPushTShape.from_gym_pusht()
        self.obstacles = (
            InsertionObstacle.from_center_size(
                geometry.left_obstacle_center,
                geometry.obstacle_width,
                geometry.obstacle_height,
            ),
            InsertionObstacle.from_center_size(
                geometry.right_obstacle_center,
                geometry.obstacle_width,
                geometry.obstacle_height,
            ),
        )

    def step(self, state: PushTInsertionState, target_xy: Array) -> PushTInsertionState:
        cfg = self.config
        target_xy = clip_workspace(target_xy)
        agent_delta = clamp_norm(target_xy - state.agent_xy, cfg.max_target_step)
        new_agent_vel = cfg.pusher_speed_gain * agent_delta
        new_agent_xy = clip_workspace(state.agent_xy + new_agent_vel)

        dist, normal_out, contact_world = self.shape.world_contact_query(
            state.block_xy,
            state.block_theta,
            new_agent_xy,
        )
        contact_gap = dist - AGENT_RADIUS
        block_xy = state.block_xy.copy()
        block_theta = float(state.block_theta)

        if contact_gap <= cfg.contact_margin:
            inward_normal = -normal_out
            inward_speed = float(np.dot(new_agent_vel, inward_normal))
            if inward_speed > 0.0:
                tangential = new_agent_vel - inward_speed * inward_normal
                block_step = cfg.normal_push_gain * inward_speed * inward_normal + cfg.tangential_push_gain * tangential
                block_step = clamp_norm(block_step, cfg.object_step_limit)
                arm = contact_world - state.block_xy
                torque_like = float(arm[0] * block_step[1] - arm[1] * block_step[0])
                theta_step = float(np.clip(cfg.angular_gain * torque_like, -cfg.omega_step_limit, cfg.omega_step_limit))
                block_xy = state.block_xy + block_step
                block_theta = angle_diff(state.block_theta + theta_step, 0.0)

        return PushTInsertionState(new_agent_xy, new_agent_vel, block_xy, block_theta)

    def rollout(self, state: PushTInsertionState, actions: Array) -> list[PushTInsertionState]:
        states = [state]
        current = state
        for action in np.asarray(actions, dtype=np.float64):
            current = self.step(current, action)
            states.append(current)
        return states

    def obstacle_penalty(self, block_xy: Array, theta: float) -> float:
        vertices = self.shape.world_vertices(block_xy, theta)
        penalty = 0.0
        for obstacle in self.obstacles:
            for point in vertices:
                d = np.abs(point - obstacle.center) - obstacle.half_extents
                outside = np.maximum(d, 0.0)
                outside_dist = float(np.linalg.norm(outside))
                inside_depth = float(max(-np.max(d), 0.0)) if np.all(d <= 0.0) else 0.0
                clearance = outside_dist if inside_depth <= 0.0 else -inside_depth
                violation = self.config.obstacle_margin - clearance
                if violation > 0.0:
                    penalty += violation * violation
        return penalty


class InsertionModelBasedController:
    def __init__(self, geometry: InsertionGeometry, config: InsertionMPCConfig):
        self.geometry = geometry
        self.config = config
        self.model = PixelInsertionModel(geometry, config)
        self.goal_pose = geometry.goal_block_pose
        self.primitives = (
            PushPrimitive("center_from_right", "center", 1.0),
            PushPrimitive("center_from_left", "center", -1.0),
            PushPrimitive("insert_from_top", "insert", 1.0),
            PushPrimitive("yaw_from_left", "yaw", -1.0),
            PushPrimitive("yaw_from_right", "yaw", 1.0),
        )

    def primitive_target(self, state: PushTInsertionState, primitive: PushPrimitive) -> Array:
        cfg = self.config
        bx, by = state.block_xy
        gx, gy, _ = self.goal_pose
        x_err = bx - gx
        yaw_err = angle_diff(state.block_theta, float(self.goal_pose[2]))

        if primitive.mode == "center":
            side = np.sign(x_err) if abs(x_err) > 1e-6 else primitive.side
            if side != primitive.side and abs(x_err) > 4.0:
                return state.agent_xy.copy()
            return np.asarray(
                [bx + side * (cfg.side_contact_x - cfg.center_drive_x), by + cfg.side_y],
                dtype=np.float64,
            )

        if primitive.mode == "yaw":
            side = -np.sign(yaw_err) if abs(yaw_err) > 1e-6 else primitive.side
            if side != primitive.side and abs(yaw_err) > 0.04:
                return state.agent_xy.copy()
            contact_x = bx + side * cfg.yaw_side_x
            return np.asarray([contact_x, by - cfg.approach_y + cfg.yaw_drive_y], dtype=np.float64)

        push_y = max(by - cfg.center_approach_y + cfg.insert_drive_y, gy + 16.0)
        x_bias = -0.25 * (bx - gx)
        yaw_bias = -28.0 * angle_diff(state.block_theta, float(self.goal_pose[2]))
        return np.asarray([gx + x_bias + yaw_bias, push_y], dtype=np.float64)

    def action_sequence_for_primitive(
        self,
        state: PushTInsertionState,
        primitive: PushPrimitive,
    ) -> tuple[Array, list[PushTInsertionState]]:
        actions: list[Array] = []
        states = [state]
        current = state
        for _ in range(self.config.horizon):
            target = clip_workspace(self.primitive_target(current, primitive))
            actions.append(target)
            current = self.model.step(current, target)
            states.append(current)
        return np.stack(actions, axis=0), states

    def trajectory_cost(self, states: list[PushTInsertionState], actions: Array, start_state: PushTInsertionState) -> float:
        cfg = self.config
        gx, gy, gtheta = self.goal_pose
        initial_y_err = abs(start_state.block_xy[1] - gy)
        cost = 0.0
        prev_action = start_state.agent_xy

        for idx, state in enumerate(states[1:], start=1):
            terminal_scale = cfg.w_terminal if idx == len(states) - 1 else 1.0
            x_err = state.block_xy[0] - gx
            y_err = state.block_xy[1] - gy
            yaw_err = angle_diff(state.block_theta, float(gtheta))
            progress = max(initial_y_err - abs(y_err), 0.0)

            cost += terminal_scale * (
                cfg.w_center_x * x_err * x_err
                + cfg.w_y_goal * y_err * y_err
                + cfg.w_yaw * yaw_err * yaw_err
            )
            cost -= cfg.w_y_progress * progress

            dist, _, _ = self.model.shape.world_contact_query(state.block_xy, state.block_theta, state.agent_xy)
            contact_gap = abs(dist - AGENT_RADIUS)
            cost += cfg.w_contact * min(contact_gap * contact_gap, 1600.0)
            cost += cfg.w_obstacle * self.model.obstacle_penalty(state.block_xy, state.block_theta)

        for action in actions:
            cost += cfg.w_action_delta * float(np.sum((action - prev_action) ** 2))
            prev_action = action
        return float(cost)

    def solve(self, env_state: Array, agent_vel: Array | None = None) -> tuple[Array, dict[str, Any]]:
        start = PushTInsertionState.from_env_state(env_state, agent_vel=agent_vel)
        best: tuple[float, PushPrimitive, Array, list[PushTInsertionState]] | None = None

        for primitive in self.primitives:
            actions, states = self.action_sequence_for_primitive(start, primitive)
            cost = self.trajectory_cost(states, actions, start)
            if best is None or cost < best[0]:
                best = (cost, primitive, actions, states)

        assert best is not None
        cost, primitive, actions, states = best
        action = clip_workspace(actions[0]).astype(np.float32)
        info = {
            "primitive": primitive.name,
            "cost": float(cost),
            "planned_actions": actions.astype(float).tolist(),
            "predicted_states": [state.as_env_state().astype(float).tolist() for state in states],
        }
        return action, info


def validate_model_rollout_against_env(
    model: PixelInsertionModel,
    env: Any,
    start_env_state: Array,
    action_seq: Array,
) -> dict[str, Any]:
    start = PushTInsertionState.from_env_state(start_env_state)
    model_states = model.rollout(start, action_seq)

    set_state(env, start_env_state)
    real_states = [np.asarray(start_env_state, dtype=np.float64)]
    for action in action_seq:
        env.step(np.asarray(action, dtype=np.float32))
        real_states.append(get_state(env))

    model_arr = np.stack([state.as_env_state() for state in model_states], axis=0)
    real_arr = np.stack(real_states, axis=0)
    block_pos_err = np.linalg.norm(model_arr[:, 2:4] - real_arr[:, 2:4], axis=1)
    agent_pos_err = np.linalg.norm(model_arr[:, :2] - real_arr[:, :2], axis=1)
    yaw_err = np.asarray([abs(angle_diff(m, r)) for m, r in zip(model_arr[:, 4], real_arr[:, 4])], dtype=np.float64)
    return {
        "mean_agent_position_error": float(np.mean(agent_pos_err)),
        "final_agent_position_error": float(agent_pos_err[-1]),
        "mean_block_position_error": float(np.mean(block_pos_err)),
        "final_block_position_error": float(block_pos_err[-1]),
        "mean_yaw_error": float(np.mean(yaw_err)),
        "final_yaw_error": float(yaw_err[-1]),
    }


def save_video(path: Path, frames: list[Array], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)


def make_episode_geometry(args: argparse.Namespace, episode_idx: int) -> InsertionGeometry:
    start_dx = float(args.start_dx)
    start_dy = float(args.start_dy)
    start_dtheta = float(args.start_dtheta)

    if args.randomize_start:
        rng = np.random.default_rng(args.seed + episode_idx)
        start_dx += float(rng.uniform(-args.start_dx_jitter, args.start_dx_jitter))
        start_dy += float(rng.uniform(-args.start_dy_jitter, args.start_dy_jitter))
        start_dtheta = math.radians(
            float(rng.uniform(args.start_dtheta_min_deg, args.start_dtheta_max_deg))
        )

    return InsertionGeometry(
        buffer=args.buffer,
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        goal_block_y=args.goal_y,
        start_dx=start_dx,
        start_dy=start_dy,
        start_dtheta=start_dtheta,
        start_agent_y_offset=args.start_agent_y_offset,
    )


def rollout_episode(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    geometry = make_episode_geometry(args, episode_idx)
    config = InsertionMPCConfig(
        horizon=args.horizon,
        max_target_step=args.max_target_step,
        pusher_speed_gain=args.pusher_speed_gain,
        normal_push_gain=args.normal_push_gain,
        tangential_push_gain=args.tangential_push_gain,
        angular_gain=args.angular_gain,
        w_obstacle=args.w_obstacle,
        w_contact=args.w_contact,
    )
    controller = InsertionModelBasedController(geometry, config)
    env = make_env(geometry, render_size=args.render_size, obs_size=args.obs_size)
    validation_env = make_env(geometry, render_size=args.render_size, obs_size=args.obs_size) if args.validate_rollout else None

    frames: list[Array] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    step_meta: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    previous_agent_xy: Array | None = None

    try:
        env.reset(seed=args.seed + episode_idx)
        if validation_env is not None:
            validation_env.reset(seed=args.seed + episode_idx)

        frames.append(np.asarray(env.render(), dtype=np.uint8))
        states.append(get_state(env).tolist())

        for step_idx in range(args.max_steps):
            state = get_state(env)
            agent_vel = np.zeros(2, dtype=np.float64) if previous_agent_xy is None else state[:2] - previous_agent_xy
            previous_agent_xy = state[:2].copy()

            action, info = controller.solve(state, agent_vel=agent_vel)
            if validation_env is not None:
                validation = validate_model_rollout_against_env(
                    controller.model,
                    validation_env,
                    state,
                    np.asarray(info["planned_actions"], dtype=np.float64),
                )
                info["model_validation"] = validation

            _, _, _, _, env_info = env.step(action)
            next_state = get_state(env)
            metrics = success_metrics(next_state, geometry.goal_block_pose)

            info.update(
                {
                    "step": step_idx,
                    "action": action.astype(float).tolist(),
                    "block_pose": next_state[2:5].astype(float).tolist(),
                    "agent_xy": next_state[:2].astype(float).tolist(),
                    "n_contacts": int(env_info.get("n_contacts", 0)),
                    "metrics": metrics,
                }
            )
            actions.append(action.astype(float).tolist())
            states.append(next_state.tolist())
            step_meta.append(info)

            if (step_idx + 1) % args.save_every == 0 or metrics["success"]:
                frames.append(np.asarray(env.render(), dtype=np.uint8))
            if metrics["success"]:
                break

        if not metrics:
            metrics = success_metrics(get_state(env), geometry.goal_block_pose)

        episode_dir = args.out_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_path = episode_dir / "model_based_controller.mp4"
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
        if validation_env is not None:
            validation_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--obs-size", type=int, default=96)
    parser.add_argument("--fps", type=int, default=ENV_FPS)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--validate-rollout", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--buffer", type=float, default=25.0)
    parser.add_argument("--obstacle-width", type=float, default=512.0 / 4.0)
    parser.add_argument("--obstacle-height", type=float, default=135.0)
    parser.add_argument("--goal-y", type=float, default=330.0)
    parser.add_argument("--start-dx", type=float, default=18.0)
    parser.add_argument("--start-dy", type=float, default=-155.0)
    parser.add_argument("--start-dtheta", type=float, default=math.radians(30.0))
    parser.add_argument("--start-agent-y-offset", type=float, default=-70.0)
    parser.add_argument("--randomize-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-dx-jitter", type=float, default=8.0)
    parser.add_argument("--start-dy-jitter", type=float, default=15.0)
    parser.add_argument("--start-dtheta-min-deg", type=float, default=-30.0)
    parser.add_argument("--start-dtheta-max-deg", type=float, default=30.0)

    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max-target-step", type=float, default=58.0)
    parser.add_argument("--pusher-speed-gain", type=float, default=0.55)
    parser.add_argument("--normal-push-gain", type=float, default=0.52)
    parser.add_argument("--tangential-push-gain", type=float, default=0.08)
    parser.add_argument("--angular-gain", type=float, default=0.0018)
    parser.add_argument("--w-obstacle", type=float, default=18.0)
    parser.add_argument("--w-contact", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1.")
    if not 0.0 <= args.start_dtheta <= math.radians(30.0):
        raise ValueError("--start-dtheta must be between 0 and 30 degrees, in radians.")
    if args.start_dx_jitter < 0.0:
        raise ValueError("--start-dx-jitter must be >= 0.")
    if args.start_dy_jitter < 0.0:
        raise ValueError("--start-dy-jitter must be >= 0.")
    if not -30.0 <= args.start_dtheta_min_deg <= 30.0:
        raise ValueError("--start-dtheta-min-deg must be between -30 and 30.")
    if not -30.0 <= args.start_dtheta_max_deg <= 30.0:
        raise ValueError("--start-dtheta-max-deg must be between -30 and 30.")
    if args.start_dtheta_min_deg > args.start_dtheta_max_deg:
        raise ValueError("--start-dtheta-min-deg must be <= --start-dtheta-max-deg.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [rollout_episode(args, episode_idx) for episode_idx in tqdm(range(args.episodes), desc="model-based insertion", unit="episode")]

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
