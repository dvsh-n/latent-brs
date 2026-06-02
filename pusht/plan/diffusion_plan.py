#!/usr/bin/env python3
"""Roll out the LeRobot PushT diffusion policy in this repo's PushT env."""

from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
    reset_pusht_env_to_obstacle_init,
    reset_pusht_env_to_state,
)
from pusht.shared.utils import load_expert_policy_bundle, render_frame, select_expert_action

DEFAULT_MODEL_DIR = Path("pusht/models")
DEFAULT_OUT_DIR = Path("pusht/plan/diffusion_plan")
DEFAULT_MODE = "noised_obstacle"
DEFAULT_RENDER_SIZE = 512
INIT_MODES = ("normal", "edge", "obstacle", "noised_obstacle")
PUSHT_WALL_MIN = 5.0
PUSHT_WALL_MAX = 506.0
PUSHT_WALL_RADIUS = 2.0
PUSHT_AGENT_RADIUS = 15.0
PUSHT_CANVAS_SIZE = 512.0
PUSHT_TEE_SCALE = 30.0
PUSHT_TEE_LENGTH = 4.0
OBSTACLE_COLOR_RGBA = (255, 140, 0, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--video-name", default="diffusion_plan.mp4")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE, help="Resolution for the full env render video.")
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-noise-std", type=float, default=8.5)
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=INIT_MODES,
        help=(
            "Initialization/evaluation mode: normal env reset, edge pusher respawn, obstacle custom init, "
            "or obstacle custom init with noised expert actions."
        ),
    )
    parser.add_argument(
        "--edge-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for --mode edge/normal.",
    )
    parser.add_argument(
        "--obstacle-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Alias for --mode obstacle.",
    )
    parser.add_argument(
        "--obstacle-init-block-offset",
        type=float,
        default=165.0,
        help="Distance to move the T from the goal along the negative local vertical symmetry axis.",
    )
    parser.add_argument(
        "--obstacle-init-max-tilt-deg",
        dest="obstacle_init_max_tilt_deg",
        type=float,
        default=20.0,
        help="Maximum absolute tilt sampled for --obstacle-init, in degrees.",
    )
    parser.add_argument(
        "--obstacle-init-axis-threshold",
        type=float,
        default=10.0,
        help="Obstacle-init success threshold for block distance to the goal along the sampled tilted axis.",
    )
    parser.add_argument(
        "--obstacle-init-pusher-face-offset",
        type=float,
        default=15.0,
        help="Distance from the T top face center to the pusher center for --obstacle-init.",
    )
    parser.add_argument(
        "--obstacle-visual-buffer",
        type=float,
        default=12.0,
        help="Pixel gap between the visual-only orange obstacle and the goal T insertion slot.",
    )
    parser.add_argument(
        "--obstacle-visual-thickness",
        type=float,
        default=80.0,
        help="Pixel thickness of the visual-only orange obstacle walls.",
    )
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Query the policy every N env steps and hold the last action in between.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    args.mode = resolve_mode(args)
    return args


def resolve_mode(args: argparse.Namespace) -> str:
    mode = str(args.mode)
    if args.edge_sample is not None:
        mode = "edge" if args.edge_sample else "normal"
    if args.obstacle_init and mode != "noised_obstacle":
        mode = "obstacle"
    elif args.obstacle_init is False and _is_obstacle_mode(mode):
        mode = "normal"
    return mode


def _is_obstacle_mode(mode: str) -> bool:
    return mode in {"obstacle", "noised_obstacle"}


def _uses_noised_expert(args: argparse.Namespace, mode: str) -> bool:
    return bool(args.control_noise or mode == "noised_obstacle")


def maybe_display(frame: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install opencv-python or run without --display.") from exc

    cv2.imshow("PushT diffusion plan", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to save videos, or pass --no-video.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _clip_action_to_space(action: np.ndarray, env: Any) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != action.shape or low.shape != action.shape:
        return np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _local_rect_to_frame_polygon(
    goal_pose: np.ndarray,
    rect: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = rect
    corners = np.asarray(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float64,
    )
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    rotation = _rotation_matrix(float(goal_pose[2]))
    world = goal_pose[:2] + corners @ rotation.T
    height, width = frame_shape[:2]
    scale = np.asarray([float(width) / PUSHT_CANVAS_SIZE, float(height) / PUSHT_CANVAS_SIZE], dtype=np.float64)
    pixels = world * scale
    return [(float(x), float(y)) for x, y in pixels]


def _obstacle_visual_rects(*, buffer: float, thickness: float) -> list[tuple[float, float, float, float]]:
    buffer = float(buffer)
    thickness = float(thickness)
    stem_xmin = -0.5 * PUSHT_TEE_SCALE
    stem_xmax = 0.5 * PUSHT_TEE_SCALE
    cap_ymax = PUSHT_TEE_SCALE
    stem_ymax = PUSHT_TEE_LENGTH * PUSHT_TEE_SCALE

    inner_xmin = stem_xmin - buffer
    inner_xmax = stem_xmax + buffer
    inner_ymin = cap_ymax + buffer
    inner_ymax = stem_ymax + buffer
    outer_xmin = inner_xmin - thickness
    outer_xmax = inner_xmax + thickness
    outer_ymax = inner_ymax + thickness

    return [
        (outer_xmin, inner_xmin, inner_ymin, outer_ymax),
        (inner_xmax, outer_xmax, inner_ymin, outer_ymax),
        (outer_xmin, outer_xmax, inner_ymax, outer_ymax),
    ]


def render_obstacle_frame(
    frame: np.ndarray,
    goal_pose: np.ndarray | None,
    *,
    buffer: float,
    thickness: float,
) -> np.ndarray:
    if goal_pose is None:
        return np.asarray(frame, dtype=np.uint8).copy()

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect in _obstacle_visual_rects(buffer=buffer, thickness=thickness):
        polygon = _local_rect_to_frame_polygon(np.asarray(goal_pose, dtype=np.float64), rect, frame.shape)
        draw.polygon(polygon, fill=OBSTACLE_COLOR_RGBA)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _pusht_agent_bounds() -> tuple[np.ndarray, np.ndarray]:
    min_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    low = np.full((2,), min_coord, dtype=np.float32)
    high = np.full((2,), max_coord, dtype=np.float32)
    return low, high


def _sample_agent_pos_on_edge(rng: np.random.Generator) -> np.ndarray:
    low, high = _pusht_agent_bounds()
    edge = int(rng.integers(4))
    coord = float(rng.uniform(float(low[0]), float(high[0])))
    if edge == 0:
        return np.asarray([low[0], coord], dtype=np.float32)
    if edge == 1:
        return np.asarray([high[0], coord], dtype=np.float32)
    if edge == 2:
        return np.asarray([coord, low[1]], dtype=np.float32)
    return np.asarray([coord, high[1]], dtype=np.float32)


def _refresh_observation(
    observation: dict[str, Any],
    *,
    pixels: np.ndarray,
    agent_pos: np.ndarray,
) -> dict[str, Any]:
    updated = dict(observation)
    if "pixels" in updated:
        updated["pixels"] = pixels
    if "image" in updated:
        updated["image"] = pixels
    if "agent_pos" in updated:
        updated["agent_pos"] = agent_pos.astype(np.float32, copy=True)
    if "proprio" in updated:
        proprio = np.asarray(updated["proprio"], dtype=np.float32).copy()
        if proprio.shape[0] < 2:
            raise ValueError("Expected PushT proprio observations with at least 2 entries for agent xy.")
        proprio[:2] = agent_pos
        updated["proprio"] = proprio
    if "state" in updated:
        state = np.asarray(updated["state"], dtype=np.float32).copy()
        if state.shape[0] < 2:
            raise ValueError("Expected PushT state observations with at least 2 entries for agent xy.")
        state[:2] = agent_pos
        updated["state"] = state
    return updated


def _edge_sample_observation(
    env: Any,
    observation: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if not isinstance(observation, dict):
        raise ValueError("--mode edge requires dict observations containing agent position fields.")

    block_pose = get_pusht_block_pose(env)
    sampled_agent_pos = _sample_agent_pos_on_edge(rng)
    state = np.asarray(
        [sampled_agent_pos[0], sampled_agent_pos[1], block_pose[0], block_pose[1], block_pose[2], 0.0, 0.0],
        dtype=np.float64,
    )
    pixels = reset_pusht_env_to_state(env, state)
    return (
        _refresh_observation(observation, pixels=pixels, agent_pos=sampled_agent_pos),
        sampled_agent_pos,
        block_pose.astype(np.float32, copy=True),
    )


def _obstacle_init_observation(
    env: Any,
    observation: dict[str, Any],
    *,
    block_offset: float,
    max_tilt_deg: float,
    pusher_face_offset: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    if not isinstance(observation, dict):
        raise ValueError("--mode obstacle requires dict observations containing agent position fields.")

    tilt_deg = float(rng.uniform(-max_tilt_deg, max_tilt_deg))
    pixels, state = reset_pusht_env_to_obstacle_init(
        env,
        block_offset=block_offset,
        tilt_deg=tilt_deg,
        pusher_face_offset=pusher_face_offset,
    )
    agent_pos = state[:2].astype(np.float32, copy=True)
    block_pose = state[2:5].astype(np.float32, copy=True)
    return _refresh_observation(observation, pixels=pixels, agent_pos=agent_pos), agent_pos, block_pose, tilt_deg


def _obstacle_init_axis(block_theta: float) -> np.ndarray:
    return np.asarray([-np.sin(block_theta), np.cos(block_theta)], dtype=np.float64)


def obstacle_init_axis_distance(block_pose: np.ndarray, goal_pose: np.ndarray, axis: np.ndarray) -> float:
    block_xy = np.asarray(block_pose, dtype=np.float64).reshape(-1)[:2]
    goal_xy = np.asarray(goal_pose, dtype=np.float64).reshape(-1)[:2]
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("obstacle-init axis must have nonzero norm.")
    axis = axis / norm
    return abs(float(np.dot(block_xy - goal_xy, axis)))


def extract_goal_pose(env) -> list[float] | None:
    goal_pose = getattr(env.unwrapped, "goal_pose", None)
    if goal_pose is None:
        return None
    return np.asarray(goal_pose, dtype=np.float32).tolist()


def rollout_episode(args: argparse.Namespace, bundle, episode_idx: int) -> dict[str, object]:
    mode = resolve_mode(args)
    if mode not in INIT_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {INIT_MODES}.")
    use_noised_expert = _uses_noised_expert(args, mode)
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.control_noise_std < 0.0:
        raise ValueError("--control-noise-std must be >= 0.")
    if args.render_size < 1:
        raise ValueError("--render-size must be >= 1.")
    if args.obstacle_init_block_offset < 0.0:
        raise ValueError("--obstacle-init-block-offset must be >= 0.")
    if args.obstacle_init_pusher_face_offset < 0.0:
        raise ValueError("--obstacle-init-pusher-face-offset must be >= 0.")
    if args.obstacle_init_max_tilt_deg < 0.0:
        raise ValueError("--obstacle-init-max-tilt-deg must be >= 0.")
    if args.obstacle_init_axis_threshold < 0.0:
        raise ValueError("--obstacle-init-axis-threshold must be >= 0.")
    if args.obstacle_visual_buffer < 0.0:
        raise ValueError("--obstacle-visual-buffer must be >= 0.")
    if args.obstacle_visual_thickness < 0.0:
        raise ValueError("--obstacle-visual-thickness must be >= 0.")

    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        visualization_width=args.render_size,
        visualization_height=args.render_size,
    )
    bundle.policy.reset()
    episode_seed = None if args.seed is None else args.seed + episode_idx
    observation, info = env.reset(seed=episode_seed)
    rng = np.random.default_rng(episode_seed)
    obstacle_axis = None
    obstacle_axis_distance = None
    obstacle_init_sampled_tilt_deg = None
    obstacle_init_horizontal_displacement = None
    if _is_obstacle_mode(mode):
        observation, initial_agent_pos, initial_block_pose, obstacle_init_sampled_tilt_deg = _obstacle_init_observation(
            env,
            observation,
            block_offset=args.obstacle_init_block_offset,
            max_tilt_deg=args.obstacle_init_max_tilt_deg,
            pusher_face_offset=args.obstacle_init_pusher_face_offset,
            rng=rng,
        )
        obstacle_init_horizontal_displacement = float(
            args.obstacle_init_block_offset * np.tan(np.deg2rad(obstacle_init_sampled_tilt_deg))
        )
        goal_pose_np = np.asarray(extract_goal_pose(env), dtype=np.float32)
        obstacle_axis = _obstacle_init_axis(float(initial_block_pose[2]))
        obstacle_axis_distance = obstacle_init_axis_distance(initial_block_pose, goal_pose_np, obstacle_axis)
    elif mode == "edge":
        observation, initial_agent_pos, initial_block_pose = _edge_sample_observation(
            env,
            observation,
            rng=rng,
        )
    else:
        initial_agent_pos = get_pusht_agent_pos(env).astype(np.float32, copy=True)
        initial_block_pose = get_pusht_block_pose(env).astype(np.float32, copy=True)

    goal_pose_for_render = extract_goal_pose(env)
    initial_render_frame = render_frame(env)
    render_frames = [initial_render_frame]
    obstacle_render_frames = [
        render_obstacle_frame(
            initial_render_frame,
            None if goal_pose_for_render is None else np.asarray(goal_pose_for_render, dtype=np.float32),
            buffer=args.obstacle_visual_buffer,
            thickness=args.obstacle_visual_thickness,
        )
    ]
    total_reward = 0.0
    success = False
    steps = 0
    control_updates = 0
    action = None

    for step_idx in trange(args.max_steps, desc=f"episode {episode_idx}", unit="step"):
        if action is None or step_idx % args.control_interval == 0:
            action = select_expert_action(bundle, observation, env=env, action_mode=args.action_mode)
            if use_noised_expert:
                noise = rng.normal(loc=0.0, scale=args.control_noise_std, size=action.shape).astype(np.float32)
                action = _clip_action_to_space(action + noise, env)
            control_updates += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if _is_obstacle_mode(mode):
            current_block_pose = get_pusht_block_pose(env)
            goal_pose_np = np.asarray(extract_goal_pose(env), dtype=np.float32)
            assert obstacle_axis is not None
            obstacle_axis_distance = obstacle_init_axis_distance(current_block_pose, goal_pose_np, obstacle_axis)
            success = bool(obstacle_axis_distance <= args.obstacle_init_axis_threshold)
            stop_episode = success or truncated
        else:
            success = bool(terminated or info.get("is_success", False) or info.get("success", False))
            stop_episode = bool(terminated or truncated)

        if steps % args.control_interval == 0 or stop_episode:
            frame = render_frame(env)
            render_frames.append(frame)
            obstacle_render_frames.append(
                render_obstacle_frame(
                    frame,
                    None if goal_pose_for_render is None else np.asarray(goal_pose_for_render, dtype=np.float32),
                    buffer=args.obstacle_visual_buffer,
                    thickness=args.obstacle_visual_thickness,
                )
            )
            maybe_display(frame, args.display)

        observation = next_observation
        if stop_episode:
            break

    final_block_pose = get_pusht_block_pose(env).tolist()
    final_agent_pos = get_pusht_agent_pos(env).tolist()
    goal_pose = extract_goal_pose(env)
    env.close()
    stored_steps = len(render_frames)

    video_path = None
    obstacle_video_path = None
    if not args.no_video:
        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        obstacle_video_path = (
            args.out_dir / f"{Path(args.video_name).stem}_obstacle{suffix}{Path(args.video_name).suffix}"
        )
        save_video(video_path, render_frames, max(1, int(args.fps)))
        save_video(obstacle_video_path, obstacle_render_frames, max(1, int(args.fps)))

    return {
        "episode": episode_idx,
        "seed": episode_seed,
        "env_steps": steps,
        "stored_steps": stored_steps,
        "control_updates": control_updates,
        "total_reward": total_reward,
        "success": success,
        "mode": mode,
        "control_noise": use_noised_expert,
        "control_noise_std": float(args.control_noise_std),
        "goal_pose": goal_pose,
        "initial_agent_pos": initial_agent_pos.tolist(),
        "initial_block_pose": initial_block_pose.tolist(),
        "final_block_pose": final_block_pose,
        "final_agent_pos": final_agent_pos,
        "edge_sample": mode == "edge",
        "obstacle_init": _is_obstacle_mode(mode),
        "noised_obstacle": mode == "noised_obstacle",
        "obstacle_init_block_offset": float(args.obstacle_init_block_offset),
        "obstacle_init_horizontal_displacement": obstacle_init_horizontal_displacement,
        "obstacle_init_max_tilt_deg": float(args.obstacle_init_max_tilt_deg),
        "obstacle_init_sampled_tilt_deg": obstacle_init_sampled_tilt_deg,
        "obstacle_init_pusher_face_offset": float(args.obstacle_init_pusher_face_offset),
        "obstacle_init_axis_threshold": float(args.obstacle_init_axis_threshold),
        "obstacle_init_axis": obstacle_axis.astype(float).tolist() if obstacle_axis is not None else None,
        "obstacle_init_axis_distance": float(obstacle_axis_distance) if obstacle_axis_distance is not None else None,
        "obstacle_visual_buffer": float(args.obstacle_visual_buffer),
        "obstacle_visual_thickness": float(args.obstacle_visual_thickness),
        "video_path": str(video_path) if video_path is not None else None,
        "obstacle_video_path": str(obstacle_video_path) if obstacle_video_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading diffusion policy from: {args.model_dir}")
    bundle = load_expert_policy_bundle(args.model_dir, device=args.device)
    results = [rollout_episode(args, bundle, episode_idx) for episode_idx in range(args.episodes)]

    metrics_path = args.out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} stored_steps={result['stored_steps']} "
            f"env_steps={result['env_steps']} control_updates={result['control_updates']} "
            f"reward={result['total_reward']:.3f} "
            f"success={result['success']} video={result['video_path']} obstacle_video={result['obstacle_video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
