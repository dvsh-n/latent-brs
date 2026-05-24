#!/usr/bin/env python3
"""Random Reacher joint-space planning with a double-integrator controller."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_OUT_DIR = Path("reacher/plan/double_integrator")
PHYSICS_FREQ_HZ = 50.0
CONTROL_FREQ_HZ = 50.0
ACTION_NOISE_STD = 10.0


def configure_offscreen_framebuffer(env: DmControlGymEnv, width: int, height: int) -> None:
    global_ = env._env.physics.model.vis.global_
    global_.offheight = max(height, int(global_.offheight))
    global_.offwidth = max(width, int(global_.offwidth))


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def compute_control_substeps(physics_freq_hz: float, control_freq_hz: float) -> int:
    ratio = physics_freq_hz / control_freq_hz
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1e-8):
        raise ValueError(
            "--physics-freq-hz must be an integer multiple of --control-freq-hz "
            f"(got {physics_freq_hz:g} Hz and {control_freq_hz:g} Hz)."
        )
    return substeps


def reset_env_to_state(
    env: DmControlGymEnv,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def tip_root_distance(env: DmControlGymEnv) -> float:
    physics = env._env.physics
    return float(
        np.linalg.norm(
            np.asarray(physics.named.data.geom_xpos["finger", :2])
            - np.asarray(physics.named.data.geom_xpos["root", :2])
        )
    )


def compute_joint2_clearance_limit(env: DmControlGymEnv, min_tip_root_distance: float) -> float:
    if min_tip_root_distance <= 0.0:
        return float(np.pi)
    physics = env._env.physics
    original_qpos = np.asarray(physics.data.qpos[:2], dtype=np.float64).copy()
    original_qvel = np.asarray(physics.data.qvel[:2], dtype=np.float64).copy()

    low = 0.0
    high = float(np.pi)
    for _ in range(40):
        mid = 0.5 * (low + high)
        with physics.reset_context():
            physics.data.qpos[:2] = [0.0, mid]
            physics.data.qvel[:2] = 0.0
        if tip_root_distance(env) >= min_tip_root_distance:
            low = mid
        else:
            high = mid

    with physics.reset_context():
        physics.data.qpos[:2] = original_qpos
        physics.data.qvel[:2] = original_qvel
    return low


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", choices=("easy", "hard"), default="hard")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--physics-freq-hz", type=float, default=PHYSICS_FREQ_HZ)
    parser.add_argument("--control-freq-hz", type=float, default=CONTROL_FREQ_HZ)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--goal-threshold", type=float, default=0.05)
    parser.add_argument("--min-goal-distance", type=float, default=0.75)
    parser.add_argument("--min-tip-root-distance", type=float, default=0.045)
    parser.add_argument("--control-gain", type=float, default=0.06)
    parser.add_argument("--velocity-gain", type=float, default=0.015)
    parser.add_argument("--action-noise", action="store_true", help="Add Gaussian noise to the desired acceleration.")
    parser.add_argument("--action-noise-std", type=float, default=ACTION_NOISE_STD)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--video-fps", type=int, default=60)
    return parser.parse_args()


def make_env(args: argparse.Namespace, *, env_seed: int) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name=args.task,
        seed=env_seed,
        time_limit=args.time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=env_seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / args.physics_freq_hz, time_limit=args.time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, args.width, args.height)
    return env


def sample_start_goal(
    env: DmControlGymEnv,
    rng: np.random.Generator,
    *,
    min_goal_distance: float,
    min_tip_root_distance: float,
    max_abs_joint2: float,
) -> tuple[np.ndarray, np.ndarray]:
    def sample_valid_qpos() -> np.ndarray:
        for _ in range(10_000):
            env.reset(seed=int(rng.integers(0, np.iinfo(np.int32).max)))
            hide_target(env)
            physics = env._env.physics
            qpos = np.asarray(physics.data.qpos[:2], dtype=np.float32).copy()
            if tip_root_distance(env) >= min_tip_root_distance and abs(float(wrap_to_pi(qpos[1]))) <= max_abs_joint2:
                return qpos
        raise RuntimeError("Failed to sample a non-overlapping DM Control Reacher qpos.")

    for _ in range(10_000):
        start_qpos = sample_valid_qpos()
        goal_qpos = sample_valid_qpos()
        if angular_distance(start_qpos, goal_qpos) >= min_goal_distance:
            return start_qpos, goal_qpos
    raise RuntimeError("Failed to sample a start/goal pair with the requested minimum distance.")


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def nearest_equivalent_goal(
    qpos: np.ndarray,
    goal_qpos: np.ndarray,
    *,
    max_abs_joint2: float,
) -> np.ndarray:
    goal = (qpos + wrap_to_pi(goal_qpos - qpos)).astype(np.float32)
    joint2_candidates = np.asarray([goal_qpos[1] - 2.0 * np.pi, goal_qpos[1], goal_qpos[1] + 2.0 * np.pi])
    valid_joint2 = joint2_candidates[np.abs(joint2_candidates) <= max_abs_joint2]
    if valid_joint2.size > 0:
        goal[1] = valid_joint2[np.argmin(np.abs(valid_joint2 - qpos[1]))]
    return goal.astype(np.float32)


def angular_distance(qpos: np.ndarray, goal_qpos: np.ndarray) -> float:
    return float(np.linalg.norm(wrap_to_pi(goal_qpos - qpos)))


def double_integrator_plan(
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_qpos: np.ndarray,
    *,
    dt: float,
    horizon: int,
    regularization: float,
) -> np.ndarray:
    """Minimum-norm acceleration sequence to drive [q, qdot] to [goal, 0]."""
    state_dim = 4
    action_dim = 2
    a_mat = np.block(
        [
            [np.eye(2), dt * np.eye(2)],
            [np.zeros((2, 2)), np.eye(2)],
        ]
    )
    b_mat = np.vstack((0.5 * dt * dt * np.eye(2), dt * np.eye(2)))
    x0 = np.concatenate((qpos, qvel), axis=0).astype(np.float64)
    x_goal = np.concatenate((goal_qpos, np.zeros(2, dtype=np.float32)), axis=0).astype(np.float64)

    powers = [np.eye(state_dim)]
    for _ in range(horizon):
        powers.append(a_mat @ powers[-1])

    controllability = np.zeros((state_dim, horizon * action_dim), dtype=np.float64)
    for step in range(horizon):
        controllability[:, step * action_dim : (step + 1) * action_dim] = powers[horizon - 1 - step] @ b_mat

    residual = x_goal - powers[horizon] @ x0
    gram = controllability @ controllability.T + regularization * np.eye(state_dim)
    acc_flat = controllability.T @ np.linalg.solve(gram, residual)
    return acc_flat.reshape(horizon, action_dim).astype(np.float32)


def save_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rollout.mp4"
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    return path


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive.")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.physics_freq_hz <= 0.0:
        raise ValueError("--physics-freq-hz must be positive.")
    if args.control_freq_hz <= 0.0:
        raise ValueError("--control-freq-hz must be positive.")
    if args.control_gain <= 0.0:
        raise ValueError("--control-gain must be positive.")
    if args.velocity_gain < 0.0:
        raise ValueError("--velocity-gain cannot be negative.")
    if args.action_noise_std < 0.0:
        raise ValueError("--action-noise-std cannot be negative.")
    if args.min_tip_root_distance < 0.0:
        raise ValueError("--min-tip-root-distance cannot be negative.")

    rng = np.random.default_rng()
    env_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    zero_qvel = np.zeros(2, dtype=np.float32)
    physics_timestep = 1.0 / float(args.physics_freq_hz)
    control_timestep = 1.0 / float(args.control_freq_hz)
    control_substeps = compute_control_substeps(float(args.physics_freq_hz), float(args.control_freq_hz))

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_npz_path = out_dir / "rollout.npz"
    if legacy_npz_path.exists():
        legacy_npz_path.unlink()
    env = make_env(args, env_seed=env_seed)
    max_abs_joint2 = compute_joint2_clearance_limit(env, float(args.min_tip_root_distance))
    start_qpos, goal_qpos = sample_start_goal(
        env,
        rng,
        min_goal_distance=float(args.min_goal_distance),
        min_tip_root_distance=float(args.min_tip_root_distance),
        max_abs_joint2=max_abs_joint2,
    )
    start_frame = reset_env_to_state(
        env,
        seed=env_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=args.height,
        width=args.width,
    )
    start_tip_root_distance = tip_root_distance(env)
    goal_frame = reset_env_to_state(
        env,
        seed=env_seed,
        qpos=goal_qpos,
        qvel=zero_qvel,
        height=args.height,
        width=args.width,
    )
    goal_tip_root_distance = tip_root_distance(env)
    save_rgb_image(out_dir / "start.png", start_frame)
    save_rgb_image(out_dir / "goal.png", goal_frame)
    reset_env_to_state(
        env,
        seed=env_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=args.height,
        width=args.width,
    )

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    frames = [start_frame]
    qpos_log = [start_qpos.copy()]
    qvel_log = [zero_qvel.copy()]
    action_log: list[np.ndarray] = []
    accel_log: list[np.ndarray] = []
    distance_log = [angular_distance(start_qpos, goal_qpos)]
    stop_reason = "max_steps"
    raw_action: np.ndarray | None = None
    accel_cmd: np.ndarray | None = None
    executed_control_steps = 0

    for physics_step in tqdm(range(args.max_steps), desc="Double-integrator rollout", unit="step"):
        physics = env._env.physics
        if physics_step % control_substeps == 0:
            qpos = np.asarray(physics.data.qpos[:2], dtype=np.float32).copy()
            qvel = np.asarray(physics.data.qvel[:2], dtype=np.float32).copy()

            goal_for_plan = nearest_equivalent_goal(qpos, goal_qpos, max_abs_joint2=max_abs_joint2)
            accel_plan = double_integrator_plan(
                qpos,
                qvel,
                goal_for_plan,
                dt=control_timestep,
                horizon=int(args.horizon),
                regularization=float(args.regularization),
            )
            accel_cmd = accel_plan[0]
            if args.action_noise:
                accel_cmd = accel_cmd + rng.normal(loc=0.0, scale=args.action_noise_std, size=accel_cmd.shape).astype(np.float32)
            raw_action = args.control_gain * accel_cmd - args.velocity_gain * qvel
            raw_action = np.clip(raw_action, action_low, action_high).astype(np.float32)
            executed_control_steps += 1
        if raw_action is None or accel_cmd is None:
            raise RuntimeError("Controller action was not initialized.")

        _, _, terminated, truncated, _ = env.step(raw_action)
        next_qpos = np.asarray(physics.data.qpos[:2], dtype=np.float32).copy()
        next_qvel = np.asarray(physics.data.qvel[:2], dtype=np.float32).copy()
        distance = angular_distance(next_qpos, goal_qpos)

        action_log.append(raw_action.copy())
        accel_log.append(accel_cmd.copy())
        qpos_log.append(next_qpos)
        qvel_log.append(next_qvel)
        distance_log.append(distance)
        frames.append(physics.render(height=args.height, width=args.width, camera_id=0))

        if distance <= args.goal_threshold and float(np.linalg.norm(next_qvel)) <= 0.25:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    video_path = save_video(frames, out_dir, int(args.video_fps))
    final_qpos = qpos_log[-1]
    final_qvel = qvel_log[-1]
    summary = {
        "out_dir": str(out_dir),
        "video_path": str(video_path),
        "env_seed_sampled_from_entropy": env_seed,
        "start_qpos": start_qpos.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "final_qpos": final_qpos.tolist(),
        "final_qvel": final_qvel.tolist(),
        "initial_distance": float(distance_log[0]),
        "final_distance": float(distance_log[-1]),
        "executed_steps": len(action_log),
        "executed_physics_steps": int(len(action_log)),
        "executed_control_steps": int(executed_control_steps),
        "stop_reason": stop_reason,
        "horizon": int(args.horizon),
        "physics_freq_hz": float(args.physics_freq_hz),
        "control_freq_hz": float(args.control_freq_hz),
        "physics_timestep": float(physics_timestep),
        "control_timestep": float(control_timestep),
        "control_substeps": int(control_substeps),
        "frames_per_physics_step": 1,
        "control_gain": float(args.control_gain),
        "velocity_gain": float(args.velocity_gain),
        "action_noise": bool(args.action_noise),
        "action_noise_std": float(args.action_noise_std),
        "min_tip_root_distance": float(args.min_tip_root_distance),
        "start_tip_root_distance": float(start_tip_root_distance),
        "goal_tip_root_distance": float(goal_tip_root_distance),
        "max_abs_joint2_without_overlap": float(max_abs_joint2),
        "distance_metric": "wrapped_joint_angle_l2",
        "qpos_source": "dm_control randomize_limited_and_rotational_joints via env.reset",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
