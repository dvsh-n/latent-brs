from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


@dataclass(frozen=True)
class TwoRoomLayout:
    wall_axis: int = 1
    wall_thickness: int = 10
    door_positions: tuple[float, ...] = (49.0,)
    door_sizes: tuple[float, ...] = (14.0,)


class TwoRoomEnv(gym.Env):
    """Standalone Two Room environment derived from stable-worldmodel.

    This version keeps the original geometry, collision, and rendering logic,
    but replaces the stable_worldmodel variation-space dependency with simple
    constructor arguments and reset-time sampling.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    IMG_SIZE = 224
    BORDER_SIZE = 14
    MAX_SPEED = 10.5
    WALL_CENTER = 112
    WALL_WIDTH_DEFAULT = 10
    MAX_DOOR = 3
    SUCCESS_DISTANCE = 16.0
    DOOR_MARGIN = 1.75
    BORDER_LINE_THICKNESS = 4

    def __init__(
        self,
        render_mode: str = "rgb_array",
        render_target: bool = False,
        *,
        agent_color: Sequence[int] = (255, 0, 0),
        target_color: Sequence[int] = (0, 255, 0),
        wall_color: Sequence[int] = (0, 0, 0),
        door_color: Sequence[int] = (255, 255, 255),
        background_color: Sequence[int] = (255, 255, 255),
        agent_radius: float = 7.0,
        target_radius: float = 7.0,
        agent_speed: float = 5.0,
        min_steps: int = 25,
        layout: TwoRoomLayout | None = None,
    ) -> None:
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")

        self.render_mode = render_mode
        self.render_target_flag = bool(render_target)

        layout = layout or TwoRoomLayout()
        self.layout = layout

        self.agent_color = np.asarray(agent_color, dtype=np.uint8)
        self.target_color = np.asarray(target_color, dtype=np.uint8)
        self.wall_color = np.asarray(wall_color, dtype=np.uint8)
        self.door_color = np.asarray(door_color, dtype=np.uint8)
        self.background_color = np.asarray(background_color, dtype=np.uint8)
        self.agent_radius = float(agent_radius)
        self.target_radius = float(target_radius)
        self.agent_speed = float(agent_speed)
        self.min_steps = int(min_steps)

        y = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        x = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing="ij")

        state_dim = 2 + 2 + self.MAX_DOOR * 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(self.IMG_SIZE),
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.env_name = "TwoRoom"
        self.wall_axis = int(layout.wall_axis)
        self.wall_thickness = int(layout.wall_thickness)
        self.num_doors = min(len(layout.door_positions), len(layout.door_sizes), self.MAX_DOOR)
        if self.num_doors < 1:
            raise ValueError("At least one door is required.")
        self.door_positions = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.door_sizes = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self._set_layout(layout)

        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.target_position = torch.zeros(2, dtype=torch.float32)
        self._target_img: torch.Tensor | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        if "layout" in options:
            self._set_layout(options["layout"])

        self.agent_position = torch.as_tensor(
            options.get("state", self._sample_agent_position()),
            dtype=torch.float32,
        )

        default_target = self._sample_target_position(self.agent_position)
        self.target_position = torch.as_tensor(
            options.get("target_state", default_target),
            dtype=torch.float32,
        )

        self._target_img = self._render_frame(agent_pos=self.target_position)
        obs = self._get_obs()
        info = self._get_info()
        info["distance_to_target"] = float(torch.norm(self.agent_position - self.target_position))
        return obs, info

    def step(self, action: np.ndarray | Sequence[float]):
        action_t = torch.as_tensor(action, dtype=torch.float32)
        action_t = torch.clamp(action_t, -1.0, 1.0)

        pos_next = self.agent_position + action_t * self.agent_speed
        self.agent_position = self._apply_collisions(self.agent_position, pos_next)

        dist = float(torch.norm(self.agent_position - self.target_position))
        terminated = dist < self.SUCCESS_DISTANCE
        truncated = False
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info["distance_to_target"] = dist
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        img_chw = self._render_frame(agent_pos=self.agent_position).cpu().numpy()
        return img_chw.transpose(1, 2, 0)

    def set_state(self, state: np.ndarray | Sequence[float]) -> None:
        self.agent_position = torch.as_tensor(state, dtype=torch.float32)

    def set_goal_state(self, goal_state: np.ndarray | Sequence[float]) -> None:
        self.target_position = torch.as_tensor(goal_state, dtype=torch.float32)
        self._target_img = self._render_frame(agent_pos=self.target_position)

    def _set_layout(self, layout: TwoRoomLayout | dict[str, Any]) -> None:
        if isinstance(layout, dict):
            layout = TwoRoomLayout(**layout)
        self.wall_axis = int(layout.wall_axis)
        self.wall_thickness = int(layout.wall_thickness)
        door_positions = tuple(float(v) for v in layout.door_positions[: self.MAX_DOOR])
        door_sizes = tuple(float(v) for v in layout.door_sizes[: self.MAX_DOOR])
        self.num_doors = min(len(door_positions), len(door_sizes), self.MAX_DOOR)
        if self.num_doors < 1:
            raise ValueError("At least one door is required.")
        if not any(size >= 1.1 * self.agent_radius for size in door_sizes[: self.num_doors]):
            raise ValueError("At least one door must fit the agent radius.")

        self.door_positions.zero_()
        self.door_sizes.zero_()
        self.door_positions[: self.num_doors] = torch.as_tensor(door_positions, dtype=torch.float32)
        self.door_sizes[: self.num_doors] = torch.as_tensor(door_sizes, dtype=torch.float32)

    def _get_obs(self) -> np.ndarray:
        door_coords: list[float] = []
        for i in range(self.MAX_DOOR):
            if i < self.num_doors:
                center_1d = float(self.door_positions[i].item())
                if self.wall_axis == 1:
                    door_coords.extend([float(self.WALL_CENTER), center_1d])
                else:
                    door_coords.extend([center_1d, float(self.WALL_CENTER)])
            else:
                door_coords.extend([0.0, 0.0])
        return np.asarray(
            [
                float(self.agent_position[0]),
                float(self.agent_position[1]),
                float(self.target_position[0]),
                float(self.target_position[1]),
                *door_coords,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "proprio": self.agent_position.detach().cpu().numpy(),
            "state": self.agent_position.detach().cpu().numpy(),
            "goal_state": self.target_position.detach().cpu().numpy(),
        }

    def _sample_agent_position(self) -> np.ndarray:
        side = int(self.np_random.integers(0, 2))
        return self._sample_position_in_room(side)

    def _sample_target_position(self, agent_position: torch.Tensor) -> np.ndarray:
        room_idx = 0 if self.wall_axis == 1 else 1
        agent_side = int(float(agent_position[room_idx]) >= self.WALL_CENTER)
        target_side = 1 - agent_side

        for _ in range(256):
            target = self._sample_position_in_room(target_side)
            if self._target_satisfies_min_steps(agent_position.detach().cpu().numpy(), target):
                return target
        return self._sample_position_in_room(target_side)

    def _sample_position_in_room(self, side: int) -> np.ndarray:
        lo = self.BORDER_SIZE + self.agent_radius
        hi = self.IMG_SIZE - self.BORDER_SIZE - self.agent_radius
        pos = np.asarray(
            [
                self.np_random.uniform(lo, hi),
                self.np_random.uniform(lo, hi),
            ],
            dtype=np.float32,
        )

        half_thickness = self.wall_thickness // 2
        wall_min = self.WALL_CENTER - half_thickness - self.agent_radius
        wall_max = self.WALL_CENTER + half_thickness + self.agent_radius
        room_idx = 0 if self.wall_axis == 1 else 1

        if side == 0:
            pos[room_idx] = self.np_random.uniform(lo, wall_min - 1e-3)
        else:
            pos[room_idx] = self.np_random.uniform(wall_max + 1e-3, hi)
        return pos

    def _target_satisfies_min_steps(
        self, agent_pos: np.ndarray, target_pos: np.ndarray
    ) -> bool:
        if self.min_steps <= 0:
            return True

        min_path_length = float("inf")
        for i in range(self.num_doors):
            door_size = float(self.door_sizes[i])
            if door_size < 1.1 * self.agent_radius:
                continue

            door_center_1d = float(self.door_positions[i])
            if self.wall_axis == 1:
                door_center = np.asarray([float(self.WALL_CENTER), door_center_1d], dtype=np.float32)
            else:
                door_center = np.asarray([door_center_1d, float(self.WALL_CENTER)], dtype=np.float32)

            path_length = float(np.linalg.norm(agent_pos - door_center) + np.linalg.norm(target_pos - door_center))
            min_path_length = min(min_path_length, path_length)

        if not np.isfinite(min_path_length):
            return True
        return (min_path_length / self.agent_speed) >= self.min_steps

    def _render_frame(self, agent_pos: torch.Tensor) -> torch.Tensor:
        img = torch.empty((3, self.IMG_SIZE, self.IMG_SIZE), dtype=torch.uint8)
        for c in range(3):
            img[c].fill_(int(self.background_color[c]))

        wall_mask, door_mask = self._wall_and_door_masks()
        if door_mask.any():
            for c in range(3):
                img[c, door_mask] = int(self.door_color[c])
        if wall_mask.any():
            for c in range(3):
                img[c, wall_mask] = int(self.wall_color[c])

        if self.render_target_flag:
            target_dot = self._gaussian_dot(self.target_position, self.target_radius)
            img = self._alpha_blend(img, target_dot, self.target_color)

        agent_dot = self._gaussian_dot(agent_pos, self.agent_radius)
        return self._alpha_blend(img, agent_dot, self.agent_color)

    @staticmethod
    def _alpha_blend(img_u8: torch.Tensor, alpha_01: torch.Tensor, rgb_u8: np.ndarray) -> torch.Tensor:
        alpha = alpha_01.clamp(0.0, 1.0).to(torch.float32)
        out = img_u8.to(torch.float32)
        for c in range(3):
            out[c] = out[c] * (1.0 - alpha) + float(rgb_u8[c]) * alpha
        return out.to(torch.uint8)

    def _gaussian_dot(self, pos_xy: torch.Tensor, radius: float) -> torch.Tensor:
        dx = self.grid_x - float(pos_xy[0])
        dy = self.grid_y - float(pos_xy[1])
        dist2 = dx * dx + dy * dy
        std = max(1e-6, float(radius))
        dot = torch.exp(-dist2 / (2.0 * std * std))
        max_value = dot.max()
        if max_value > 0:
            dot = dot / max_value
        return dot

    def _wall_and_door_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.wall_thickness // 2
        if self.wall_axis == 1:
            wall_stripe = (self.grid_x >= (self.WALL_CENTER - half)) & (self.grid_x <= (self.WALL_CENTER + half))
            door_span = torch.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=torch.bool)
            for i in range(self.num_doors):
                center = self.door_positions[i]
                half_extent = self.door_sizes[i]
                door_span |= (self.grid_y >= (center - half_extent)) & (self.grid_y <= (center + half_extent))
        else:
            wall_stripe = (self.grid_y >= (self.WALL_CENTER - half)) & (self.grid_y <= (self.WALL_CENTER + half))
            door_span = torch.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=torch.bool)
            for i in range(self.num_doors):
                center = self.door_positions[i]
                half_extent = self.door_sizes[i]
                door_span |= (self.grid_x >= (center - half_extent)) & (self.grid_x <= (center + half_extent))

        door_mask = wall_stripe & door_span
        wall_mask = wall_stripe & (~door_span)

        bs = self.BORDER_SIZE
        t = self.BORDER_LINE_THICKNESS
        wall_mask[:, bs - t : bs] = True
        wall_mask[:, self.IMG_SIZE - bs : self.IMG_SIZE - bs + t] = True
        wall_mask[bs - t : bs, :] = True
        wall_mask[self.IMG_SIZE - bs : self.IMG_SIZE - bs + t, :] = True
        return wall_mask, door_mask

    def _apply_collisions(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        bs = float(self.BORDER_SIZE)
        x2, y2 = float(pos2[0]), float(pos2[1])
        x2 = min(max(x2, bs + self.agent_radius), self.IMG_SIZE - bs - self.agent_radius)
        y2 = min(max(y2, bs + self.agent_radius), self.IMG_SIZE - bs - self.agent_radius)
        pos2c = torch.tensor([x2, y2], dtype=torch.float32)

        half = self.wall_thickness // 2
        center = float(self.WALL_CENTER)

        if self.wall_axis == 1:
            wall_left = center - half
            wall_right = center + half
            effective_left = wall_left - self.agent_radius
            effective_right = wall_right + self.agent_radius

            x1 = float(pos1[0])
            x2_val = float(pos2c[0])
            y2_val = float(pos2c[1])
            started_left = x1 < center

            if started_left:
                if x2_val > effective_left and not self._in_any_door_1d(y2_val, self.DOOR_MARGIN):
                    pos2c[0] = effective_left - 0.5
            else:
                if x2_val < effective_right and not self._in_any_door_1d(y2_val, self.DOOR_MARGIN):
                    pos2c[0] = effective_right + 0.5
        else:
            wall_top = center - half
            wall_bottom = center + half
            effective_top = wall_top - self.agent_radius
            effective_bottom = wall_bottom + self.agent_radius

            y1 = float(pos1[1])
            y2_val = float(pos2c[1])
            x2_val = float(pos2c[0])
            started_top = y1 < center

            if started_top:
                if y2_val > effective_top and not self._in_any_door_1d(x2_val, self.DOOR_MARGIN):
                    pos2c[1] = effective_top - 0.5
            else:
                if y2_val < effective_bottom and not self._in_any_door_1d(x2_val, self.DOOR_MARGIN):
                    pos2c[1] = effective_bottom + 0.5

        return pos2c

    def _in_any_door_1d(self, coord_1d: float, margin: float) -> bool:
        for i in range(self.num_doors):
            center = float(self.door_positions[i])
            half_extent = float(self.door_sizes[i])
            if (center - half_extent - margin) <= coord_1d <= (center + half_extent + margin):
                return True
        return False


def make_two_room_env(**kwargs: Any) -> TwoRoomEnv:
    return TwoRoomEnv(**kwargs)


def set_two_room_state(env: Any, state: np.ndarray | Sequence[float]) -> None:
    base_env = getattr(env, "unwrapped", env)
    base_env.set_state(np.asarray(state, dtype=np.float32))


def set_two_room_goal_state(env: Any, goal_state: np.ndarray | Sequence[float]) -> None:
    base_env = getattr(env, "unwrapped", env)
    base_env.set_goal_state(np.asarray(goal_state, dtype=np.float32))


def reset_two_room_env_to_state(
    env: Any,
    state: np.ndarray | Sequence[float],
    *,
    goal_state: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    set_two_room_state(env, state)
    if goal_state is not None:
        set_two_room_goal_state(env, goal_state)
    return np.asarray(env.render(), dtype=np.uint8)
