from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_PUSHT_ENV_ID = "gym_pusht/PushT-v0"


def _import_gymnasium():
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install gymnasium in the active environment.") from exc
    return gym


def _ensure_env_package(env_id: str) -> None:
    package_name = env_id.split("/", maxsplit=1)[0] if "/" in env_id else None
    if package_name is None:
        return
    try:
        __import__(package_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Environment package '{package_name}' is not installed. "
            "For the default PushT env, install gym-pusht."
        ) from exc


def _import_pusht_no_target_env():
    import pygame
    import pymunk
    from gym_pusht.envs.pusht import PushTEnv
    from gym_pusht.envs.pymunk_override import DrawOptions

    class PushTNoTargetEnv(PushTEnv):
        def _setup(self):
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

            self.agent = self.add_circle(self.space, (256, 400), 15)
            self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
            self.goal_pose = np.array([256, 256, np.pi / 4])
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            self.n_contact_points = 0

        def _draw(self):
            screen = pygame.Surface((512, 512))
            screen.fill((255, 255, 255))
            draw_options = DrawOptions(screen)
            self.space.debug_draw(draw_options)
            return screen

    return PushTNoTargetEnv


def make_pusht_env(
    env_id: str = DEFAULT_PUSHT_ENV_ID,
    *,
    obs_type: str = "pixels_agent_pos",
    render_mode: str = "rgb_array",
    max_episode_steps: int = 300,
    observation_width: int | None = None,
    observation_height: int | None = None,
    visualization_width: int | None = 384,
    visualization_height: int | None = 384,
    hide_target: bool = False,
):
    """Create a PushT environment with the repo's shared defaults."""
    if hide_target:
        if env_id != DEFAULT_PUSHT_ENV_ID:
            raise ValueError(f"hide_target=True only supports env_id={DEFAULT_PUSHT_ENV_ID!r}, got {env_id!r}.")

        env_cls = _import_pusht_no_target_env()
        env = env_cls(
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width if visualization_width is not None else observation_width,
            visualization_height=visualization_height if visualization_height is not None else observation_height,
        )
        env.reset(seed=0)
        return env

    gym = _import_gymnasium()
    _ensure_env_package(env_id)

    kwargs = {
        "obs_type": obs_type,
        "render_mode": render_mode,
        "max_episode_steps": max_episode_steps,
    }
    if observation_width is not None:
        kwargs["observation_width"] = observation_width
    if observation_height is not None:
        kwargs["observation_height"] = observation_height
    if visualization_width is not None:
        kwargs["visualization_width"] = visualization_width
    if visualization_height is not None:
        kwargs["visualization_height"] = visualization_height

    try:
        return gym.make(env_id, disable_env_checker=True, **kwargs)
    except TypeError:
        for key in (
            "observation_width",
            "observation_height",
            "visualization_width",
            "visualization_height",
        ):
            kwargs.pop(key, None)
        return gym.make(env_id, disable_env_checker=True, **kwargs)


def make_no_target_env(*, height: int, width: int, max_episode_steps: int = 300):
    return make_pusht_env(
        obs_type="pixels",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
        hide_target=True,
    )


def set_pusht_state(env: Any, state: np.ndarray) -> None:
    # Use the environment's legacy setter because the block has a non-default
    # center of gravity. Direct assignment to block.position does not reproduce
    # the rendered pose from the original PushT dataset.
    env.agent.velocity = [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._set_state(np.asarray(state[:5], dtype=np.float64))
    env.agent.velocity = [float(state[5]), float(state[6])] if state.shape[0] >= 7 else [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._last_action = None


def reset_pusht_env_to_state(env: Any, state: np.ndarray) -> np.ndarray:
    set_pusht_state(env, np.asarray(state, dtype=np.float64))
    return np.asarray(env._render(visualize=False), dtype=np.uint8)


def get_pusht_block_pose(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    return np.asarray([base_env.block.position.x, base_env.block.position.y, base_env.block.angle], dtype=np.float32)


def get_pusht_agent_pos(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    return np.asarray([base_env.agent.position.x, base_env.agent.position.y], dtype=np.float32)
