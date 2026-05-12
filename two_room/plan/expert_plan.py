from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from two_room.shared import TwoRoomEnv, make_two_room_env


DEFAULT_GIF_PATH = Path("two_room/plan/expert_plan/latest.gif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--agent-speed", type=float, default=5.0)
    parser.add_argument("--render-target", action="store_true")
    parser.add_argument("--save-gif", type=Path, default=None)
    return parser.parse_args()


def _choose_waypoint(env: TwoRoomEnv, agent_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    room_idx = 0 if env.wall_axis == 1 else 1
    agent_side = agent_pos[room_idx] > env.WALL_CENTER
    goal_side = goal_pos[room_idx] > env.WALL_CENTER

    if agent_side == goal_side:
        return goal_pos

    best_door = None
    best_dist = float("inf")
    for door_center_1d, door_size in zip(env.door_positions[: env.num_doors], env.door_sizes[: env.num_doors]):
        if float(door_size) < 1.1 * env.agent_radius:
            continue
        if env.wall_axis == 1:
            door_center = np.asarray([env.WALL_CENTER, float(door_center_1d)], dtype=np.float32)
        else:
            door_center = np.asarray([float(door_center_1d), env.WALL_CENTER], dtype=np.float32)

        dist = float(np.linalg.norm(door_center - agent_pos))
        if dist < best_dist:
            best_dist = dist
            best_door = door_center

    if best_door is None:
        if env.wall_axis == 1:
            return np.asarray([env.WALL_CENTER, goal_pos[1]], dtype=np.float32)
        return np.asarray([goal_pos[0], env.WALL_CENTER], dtype=np.float32)

    door_reach_tol = env.agent_speed
    if np.linalg.norm(best_door - agent_pos) > door_reach_tol:
        return best_door
    return goal_pos


def expert_action(env: TwoRoomEnv, agent_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    waypoint = _choose_waypoint(env, agent_pos, goal_pos)
    direction = waypoint - agent_pos
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    return np.clip(direction / norm, -1.0, 1.0).astype(np.float32)


def rollout_episode(env: TwoRoomEnv, *, max_steps: int) -> dict[str, object]:
    obs, info = env.reset()
    frames = [env.render()]
    positions = [np.asarray(info["state"], dtype=np.float32)]
    actions = []

    terminated = False
    truncated = False
    for step_idx in range(max_steps):
        agent_pos = np.asarray(info["state"], dtype=np.float32)
        goal_pos = np.asarray(info["goal_state"], dtype=np.float32)
        action = expert_action(env, agent_pos, goal_pos)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        positions.append(np.asarray(info["state"], dtype=np.float32))
        frames.append(env.render())
        if terminated or truncated:
            break

    return {
        "observation": obs,
        "actions": np.asarray(actions, dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
        "goal_state": np.asarray(info["goal_state"], dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "frames": frames,
        "num_steps": len(actions),
        "final_distance": float(info["distance_to_target"]),
    }


def maybe_save_gif(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=0.1)


def main() -> None:
    args = parse_args()
    env = make_two_room_env(
        render_mode="rgb_array",
        render_target=args.render_target,
        agent_speed=args.agent_speed,
    )
    result = rollout_episode(env, max_steps=args.max_steps)
    gif_path = args.save_gif or DEFAULT_GIF_PATH

    start = result["positions"][0]
    goal = result["goal_state"]
    print(f"start: {start.tolist()}")
    print(f"goal: {goal.tolist()}")
    print(f"num_steps: {result['num_steps']}")
    print(f"terminated: {result['terminated']}")
    print(f"final_distance: {result['final_distance']:.3f}")

    maybe_save_gif(gif_path, result["frames"])
    print(f"saved_gif: {gif_path}")


if __name__ == "__main__":
    main()
