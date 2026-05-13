from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from two_room.shared import make_two_room_env


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "temp" / "sampling_bounds.png"


def rectangle_perimeter_points(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    num_points_per_edge: int = 400,
) -> np.ndarray:
    xs = np.linspace(x_min, x_max, num_points_per_edge, dtype=np.float32)
    ys = np.linspace(y_min, y_max, num_points_per_edge, dtype=np.float32)

    top = np.stack([xs, np.full_like(xs, y_min)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, y_max)], axis=1)
    left = np.stack([np.full_like(ys, x_min), ys], axis=1)
    right = np.stack([np.full_like(ys, x_max), ys], axis=1)
    return np.concatenate([top, bottom, left, right], axis=0)


def valid_sampling_rectangles(env) -> dict[str, tuple[float, float, float, float]]:
    lo = env.BORDER_SIZE + env.agent_radius
    hi = env.IMG_SIZE - env.BORDER_SIZE - env.agent_radius
    half_thickness = env.wall_thickness // 2
    wall_min = env.WALL_CENTER - half_thickness - env.agent_radius
    wall_max = env.WALL_CENTER + half_thickness + env.agent_radius

    if env.wall_axis == 1:
        return {
            "left_room": (lo, wall_min, lo, hi),
            "right_room": (wall_max, hi, lo, hi),
        }

    return {
        "top_room": (lo, hi, lo, wall_min),
        "bottom_room": (lo, hi, wall_max, hi),
    }


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int = 1) -> None:
    height, width = image.shape[:2]
    for x, y in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = xi + dx
                yy = yi + dy
                if 0 <= xx < width and 0 <= yy < height:
                    image[yy, xx] = color


def main() -> None:
    env = make_two_room_env(render_mode="rgb_array", render_target=True)
    env.reset()
    image = env.render().copy()

    rectangles = valid_sampling_rectangles(env)
    colors = {
        "left_room": (255, 0, 0),
        "right_room": (0, 128, 255),
        "top_room": (255, 0, 0),
        "bottom_room": (0, 128, 255),
    }

    for name, (x_min, x_max, y_min, y_max) in rectangles.items():
        points = rectangle_perimeter_points(x_min, x_max, y_min, y_max)
        draw_points(image, points, colors[name], radius=1)

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(DEFAULT_OUTPUT_PATH, image)
    print(f"saved: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
