#!/usr/bin/env python3
"""PushT insertion offshoot with fixed slot obstacles and exact-env MPPI."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


WS = 512.0
WALL_MIN = 5.0
WALL_MAX = 506.0
AGENT_RADIUS = 15.0
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
ENV_FPS = 10
DEFAULT_OUT_DIR = Path("pusht_rboust/test/pusht_insertion_test")


@dataclass(frozen=True)
class InsertionGeometry:
    workspace_size: float = WS
    obstacle_width: float = WS / 4.0
    obstacle_height: float = 135.0
    stem_width: float = TEE_SCALE
    buffer: float = 25.0
    slot_center_x: float = WS / 2.0
    obstacle_bottom_y: float = 500.0
    goal_block_y: float = 330.0
    goal_theta: float = 0.0
    start_dx: float = 22.0
    start_dy: float = -80.0
    start_dtheta: float = 0.12
    start_agent_y_offset: float = -55.0

    @property
    def gap_width(self) -> float:
        return self.stem_width + 2.0 * self.buffer

    @property
    def obstacle_center_y(self) -> float:
        return self.obstacle_bottom_y - self.obstacle_height / 2.0

    @property
    def obstacle_top_y(self) -> float:
        return self.obstacle_bottom_y - self.obstacle_height

    @property
    def left_obstacle_center(self) -> tuple[float, float]:
        x = self.slot_center_x - self.gap_width / 2.0 - self.obstacle_width / 2.0
        return (x, self.obstacle_center_y)

    @property
    def right_obstacle_center(self) -> tuple[float, float]:
        x = self.slot_center_x + self.gap_width / 2.0 + self.obstacle_width / 2.0
        return (x, self.obstacle_center_y)

    @property
    def goal_block_pose(self) -> np.ndarray:
        return np.asarray([self.slot_center_x, self.goal_block_y, self.goal_theta], dtype=np.float64)

    @property
    def start_block_pose(self) -> np.ndarray:
        goal = self.goal_block_pose
        return np.asarray(
            [goal[0] + self.start_dx, goal[1] + self.start_dy, goal[2] + self.start_dtheta],
            dtype=np.float64,
        )

    @property
    def start_agent_pos(self) -> np.ndarray:
        start = self.start_block_pose
        y = max(WALL_MIN + AGENT_RADIUS, start[1] + self.start_agent_y_offset)
        return np.asarray([start[0], y], dtype=np.float64)

    @property
    def start_state(self) -> np.ndarray:
        return np.concatenate([self.start_agent_pos, self.start_block_pose])


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def make_insertion_env_class():
    import pygame
    import pymunk
    from gym_pusht.envs.pusht import PushTEnv
    from gym_pusht.envs.pymunk_override import DrawOptions

    class PushTInsertionEnv(PushTEnv):
        def __init__(self, geometry: InsertionGeometry | None = None, *args: Any, **kwargs: Any):
            self.insertion_geometry = geometry or InsertionGeometry()
            super().__init__(*args, **kwargs)

        def _setup(self):
            geom = self.insertion_geometry
            self.space = pymunk.Space()
            self.space.gravity = 0, 0
            self.space.damping = self.damping if self.damping is not None else 0.0
            self.teleop = False

            walls = [
                self.add_segment(self.space, (5, 506), (5, 5), 2),
                self.add_segment(self.space, (5, 5), (506, 5), 2),
                self.add_segment(self.space, (506, 5), (506, 506), 2),
                self.add_segment(self.space, (5, 506), (506, 506), 2),
            ]
            self.space.add(*walls)

            self.agent = self.add_circle(self.space, tuple(geom.start_agent_pos), AGENT_RADIUS)
            self.block, self._block_shapes = self.add_tee(self.space, tuple(geom.start_block_pose[:2]), geom.start_block_pose[2])
            self.goal_pose = geom.goal_block_pose.copy()
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog

            self.slot_obstacles = [
                self._add_static_box(geom.left_obstacle_center, geom.obstacle_width, geom.obstacle_height),
                self._add_static_box(geom.right_obstacle_center, geom.obstacle_width, geom.obstacle_height),
            ]

            self.collision_handeler = self.space.add_collision_handler(0, 0)
            self.collision_handeler.post_solve = self._handle_collision
            self.n_contact_points = 0

        def _add_static_box(self, center: tuple[float, float], width: float, height: float):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = center
            shape = pymunk.Poly.create_box(body, (width, height))
            shape.color = pygame.Color("Orange")
            shape.friction = 1.0
            self.space.add(body, shape)
            return shape

        def reset(self, seed=None, options=None):
            options = dict(options or {})
            options.setdefault("reset_to_state", self.insertion_geometry.start_state)
            return super().reset(seed=seed, options=options)

        def _draw(self):
            screen = pygame.Surface((512, 512))
            screen.fill((255, 255, 255))
            draw_options = DrawOptions(screen)

            goal_body = self.get_goal_pose_body(self.goal_pose)
            for shape in self.block.shapes:
                goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
                goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

            self.space.debug_draw(draw_options)
            return screen

        def _get_img(self, screen, width, height, render_action=False):
            return super()._get_img(screen, width=width, height=height, render_action=False)

        def _get_coverage(self):
            current = np.asarray([self.block.position.x, self.block.position.y, self.block.angle], dtype=np.float64)
            position_error = np.linalg.norm(current[:2] - self.goal_pose[:2])
            yaw_error = abs(angle_diff(float(current[2]), float(self.goal_pose[2])))
            return float(position_error < 18.0 and yaw_error < 0.20)

    return PushTInsertionEnv


def make_env(geometry: InsertionGeometry, *, render_size: int, obs_size: int, render_mode: str = "rgb_array"):
    PushTInsertionEnv = make_insertion_env_class()
    return PushTInsertionEnv(
        geometry=geometry,
        obs_type="pixels_agent_pos",
        render_mode=render_mode,
        observation_width=obs_size,
        observation_height=obs_size,
        visualization_width=render_size,
        visualization_height=render_size,
    )


def get_state(env: Any) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    return np.asarray(
        [
            base.agent.position.x,
            base.agent.position.y,
            base.block.position.x,
            base.block.position.y,
            base.block.angle,
        ],
        dtype=np.float64,
    )


def set_state(env: Any, state: np.ndarray) -> None:
    base = getattr(env, "unwrapped", env)
    state = np.asarray(state, dtype=np.float64)
    base.agent.velocity = [0.0, 0.0]
    base.block.velocity = [0.0, 0.0]
    base.block.angular_velocity = 0.0
    base.agent.position = list(state[:2])
    base.block.angle = float(state[4])
    base.block.position = list(state[2:4])
    base.space.step(base.dt)
    base.agent.velocity = [0.0, 0.0]
    base.block.velocity = [0.0, 0.0]
    base.block.angular_velocity = 0.0
    base._last_action = None


def pose_cost(block_pose: np.ndarray, goal_pose: np.ndarray, *, q_pos: float, q_yaw: float) -> float:
    pos = float(np.sum((block_pose[:2] - goal_pose[:2]) ** 2))
    yaw = angle_diff(float(block_pose[2]), float(goal_pose[2]))
    return q_pos * pos + q_yaw * yaw * yaw


def rollout_cost(
    env: Any,
    start_state: np.ndarray,
    action_seq: np.ndarray,
    goal_pose: np.ndarray,
    *,
    q_stage: float,
    q_terminal: float,
    q_yaw: float,
    r_action: float,
    collision_cost: float,
) -> float:
    set_state(env, start_state)
    cost = 0.0
    prev_action = start_state[:2].astype(np.float64)
    for action in action_seq:
        _, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        block_pose = np.asarray(info["block_pose"], dtype=np.float64)
        cost += q_stage * pose_cost(block_pose, goal_pose, q_pos=1.0, q_yaw=q_yaw)
        cost += r_action * float(np.sum((action - prev_action) ** 2))
        cost += collision_cost * float(info.get("n_contacts", 0))
        prev_action = action
    final_pose = get_state(env)[2:5]
    cost += q_terminal * pose_cost(final_pose, goal_pose, q_pos=1.0, q_yaw=q_yaw)
    return float(cost)


_WORKER_ENV: Any | None = None


def _init_worker(geometry_dict: dict[str, float], render_size: int, obs_size: int) -> None:
    global _WORKER_ENV
    _WORKER_ENV = make_env(InsertionGeometry(**geometry_dict), render_size=render_size, obs_size=obs_size)
    _WORKER_ENV.reset(seed=0)


def _worker_rollout(payload: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]) -> float:
    if _WORKER_ENV is None:
        raise RuntimeError("MPPI worker was not initialized.")
    start_state, action_seq, goal_pose, weights = payload
    return rollout_cost(_WORKER_ENV, start_state, action_seq, goal_pose, **weights)


def sample_action_sequences(
    rng: np.random.Generator,
    mean: np.ndarray,
    *,
    samples: int,
    sigma: float,
    low: float,
    high: float,
) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=(samples,) + mean.shape)
    seqs = mean[None, :, :] + noise
    return np.clip(seqs, low, high).astype(np.float64)


def mppi_plan(
    env: Any,
    rng: np.random.Generator,
    mean: np.ndarray,
    *,
    samples: int,
    iterations: int,
    sigma: float,
    temperature: float,
    geometry: InsertionGeometry,
    weights: dict[str, float],
    executor: ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    start_state = get_state(env)
    goal_pose = geometry.goal_block_pose
    best_cost = float("inf")
    best_seq = mean.copy()

    for _ in range(iterations):
        seqs = sample_action_sequences(rng, mean, samples=samples, sigma=sigma, low=WALL_MIN, high=WALL_MAX)
        seqs[0] = np.clip(mean, WALL_MIN, WALL_MAX)
        if executor is None:
            costs = np.asarray(
                [rollout_cost(env, start_state, seq, goal_pose, **weights) for seq in seqs],
                dtype=np.float64,
            )
        else:
            payloads = [(start_state, seq, goal_pose, weights) for seq in seqs]
            costs = np.fromiter(executor.map(_worker_rollout, payloads), dtype=np.float64, count=len(payloads))

        idx = int(np.argmin(costs))
        if float(costs[idx]) < best_cost:
            best_cost = float(costs[idx])
            best_seq = seqs[idx].copy()

        scaled = -(costs - float(np.min(costs))) / max(temperature, 1e-6)
        probs = np.exp(np.clip(scaled, -60.0, 0.0))
        probs /= max(float(np.sum(probs)), 1e-12)
        mean = np.sum(seqs * probs[:, None, None], axis=0)

    set_state(env, start_state)
    return best_seq, {"best_cost": best_cost}


def success_metrics(state: np.ndarray, goal_pose: np.ndarray) -> dict[str, float | bool]:
    block_pose = state[2:5]
    pos_error = float(np.linalg.norm(block_pose[:2] - goal_pose[:2]))
    yaw_error = abs(angle_diff(float(block_pose[2]), float(goal_pose[2])))
    return {
        "position_error": pos_error,
        "yaw_error": yaw_error,
        "success": bool(pos_error <= 18.0 and yaw_error <= 0.20),
    }


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="Exact rollout worker processes. 0/1 uses the main process.")
    parser.add_argument("--sigma", type=float, default=55.0)
    parser.add_argument("--temperature", type=float, default=8000.0)
    parser.add_argument("--buffer", type=float, default=25.0)
    parser.add_argument("--obstacle-width", type=float, default=WS / 4.0)
    parser.add_argument("--obstacle-height", type=float, default=135.0)
    parser.add_argument("--goal-y", type=float, default=330.0)
    parser.add_argument("--start-dx", type=float, default=22.0)
    parser.add_argument("--start-dy", type=float, default=-80.0)
    parser.add_argument("--start-dtheta", type=float, default=0.12)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--obs-size", type=int, default=96)
    parser.add_argument("--fps", type=int, default=ENV_FPS)
    parser.add_argument("--no-plan", action="store_true", help="Only reset/render the insertion scene.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--q-stage", type=float, default=0.02)
    parser.add_argument("--q-terminal", type=float, default=4.0)
    parser.add_argument("--q-yaw", type=float, default=1600.0)
    parser.add_argument("--r-action", type=float, default=0.002)
    parser.add_argument("--collision-cost", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1.")
    if args.samples < 1:
        raise ValueError("--samples must be >= 1.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")

    geometry = InsertionGeometry(
        buffer=args.buffer,
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        goal_block_y=args.goal_y,
        start_dx=args.start_dx,
        start_dy=args.start_dy,
        start_dtheta=args.start_dtheta,
    )
    rng = np.random.default_rng(args.seed)
    env = make_env(geometry, render_size=args.render_size, obs_size=args.obs_size)
    env.reset(seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [np.asarray(env.render(), dtype=np.uint8)]
    imageio.imwrite(args.out_dir / "initial_frame.png", frames[-1])
    states = [get_state(env).tolist()]
    actions: list[list[float]] = []
    plan_stats: list[dict[str, float]] = []
    metrics = success_metrics(get_state(env), geometry.goal_block_pose)

    if args.no_plan:
        save_video(args.out_dir / "insertion_geometry.mp4", frames, fps=args.fps)
    else:
        weights = {
            "q_stage": float(args.q_stage),
            "q_terminal": float(args.q_terminal),
            "q_yaw": float(args.q_yaw),
            "r_action": float(args.r_action),
            "collision_cost": float(args.collision_cost),
        }
        mean = np.repeat(geometry.start_agent_pos[None, :], args.horizon, axis=0).astype(np.float64)
        workers = max(0, int(args.workers))
        executor = None
        try:
            if workers > 1:
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(asdict(geometry), args.render_size, args.obs_size),
                )
            progress = tqdm(range(args.max_steps), desc="PushT insertion MPPI", unit="step")
            for step_idx in progress:
                best_seq, stat = mppi_plan(
                    env,
                    rng,
                    mean,
                    samples=args.samples,
                    iterations=args.iterations,
                    sigma=args.sigma,
                    temperature=args.temperature,
                    geometry=geometry,
                    weights=weights,
                    executor=executor,
                )
                action = np.asarray(best_seq[0], dtype=np.float32)
                _, _, _, _, info = env.step(action)
                actions.append(action.astype(float).tolist())
                plan_stats.append(stat)
                state = get_state(env)
                states.append(state.tolist())
                metrics = success_metrics(state, geometry.goal_block_pose)
                progress.set_postfix(
                    pos=f"{float(metrics['position_error']):.2f}",
                    yaw=f"{float(metrics['yaw_error']):.3f}",
                    best=f"{stat['best_cost']:.1f}",
                )
                if (step_idx + 1) % args.save_every == 0 or metrics["success"]:
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
                mean = np.concatenate([best_seq[1:], best_seq[-1:]], axis=0)
                if metrics["success"]:
                    break
        finally:
            if executor is not None:
                executor.shutdown()

        save_video(args.out_dir / "insertion_mppi.mp4", frames, fps=args.fps)

    imageio.imwrite(args.out_dir / "final_frame.png", frames[-1])
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "geometry": asdict(geometry),
        "args": jsonable_args(args),
        "start_state": geometry.start_state.tolist(),
        "goal_block_pose": geometry.goal_block_pose.tolist(),
        "final_state": states[-1],
        "metrics": metrics,
        "steps": max(0, len(states) - 1),
        "actions": actions,
        "states": states,
        "plan_stats": plan_stats,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("steps", "metrics", "goal_block_pose", "final_state")}, indent=2))


if __name__ == "__main__":
    main()
