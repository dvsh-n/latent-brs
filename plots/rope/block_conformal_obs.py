#!/usr/bin/env python3
"""Draw the conformal obstacle classifier block diagram as a PDF."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "obstacle_membership"
DEFAULT_OUT = SCRIPT_DIR / "block_conformal_obs.pdf"

# Manual layout, in axes coordinates.
FIG_SIZE = (6.35, 1.55)
OUTER_BOX = (0.025, 0.06, 0.95, 0.88)
TITLE_POS = (0.055, 0.79)
IMAGE_BOX = (0.055, 0.16, 0.305, 0.49)
UNSAFE_IMAGE = (0.086, 0.225, 0.1025, 0.42)
SAFE_IMAGE = (0.226, 0.225, 0.1025, 0.42)
VIT_CENTER = (0.435, 0.40)
ARROW_START = (0.505, 0.40)
ARROW_END = (0.585, 0.40)
SAMPLES_BOX = (0.635, 0.18, 0.295, 0.44)

COLOR_OUTER_EDGE = "#375a7f"
COLOR_OUTER_FACE = "#f7fbff"
COLOR_IMAGE_EDGE = "#2f6f73"
COLOR_IMAGE_FACE = "#f2faf8"
COLOR_VIT_EDGE = "#4f5d75"
COLOR_VIT_FACE = "#eef2f6"
COLOR_ARROW = "#263238"
COLOR_GREEN = "#138a36"
COLOR_RED = "#c1121f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unsafe-name", default="obstacle_alpha_overlap.png")
    parser.add_argument("--safe-name", default="safe_alpha_overlap.png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def add_rounded_box(
    ax: plt.Axes,
    xywh: tuple[float, float, float, float],
    *,
    edgecolor: str,
    facecolor: str,
    linewidth: float,
    radius: float,
    zorder: float,
) -> FancyBboxPatch:
    x, y, width, height = xywh
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_image(ax: plt.Axes, path: Path, xywh: tuple[float, float, float, float]) -> None:
    image = mpimg.imread(path)
    x, y, width, height = xywh
    ax.imshow(image, extent=(x, x + width, y, y + height), aspect="auto", zorder=3)


def add_vit_encoder(ax: plt.Axes) -> None:
    cx, cy = VIT_CENTER
    width = 0.105
    height = 0.30
    left = -0.5 * width
    right = 0.5 * width
    bottom = -0.5 * height
    top = 0.5 * height
    slant = 0.018
    points = np.array(
        [
            [left + slant, bottom],
            [right, bottom + 0.018],
            [right - slant, top],
            [left, top - 0.018],
            [left + slant, bottom],
        ]
    )
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    transform = transforms.Affine2D().rotate_deg_around(0.0, 0.0, -2.0).translate(cx, cy) + ax.transAxes
    patch = PathPatch(
        MplPath(points, codes),
        transform=transform,
        facecolor=COLOR_VIT_FACE,
        edgecolor=COLOR_VIT_EDGE,
        linewidth=2.0,
        joinstyle="round",
        capstyle="round",
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        "ViT",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#2f3e46",
        rotation=-2.0,
        transform=ax.transAxes,
        zorder=6,
    )


def add_narrow_arrow(ax: plt.Axes) -> None:
    x0, y0 = ARROW_START
    x1, y1 = ARROW_END
    head = 0.024
    wing = 0.04
    ax.plot([x0, x1], [y0, y1], color=COLOR_ARROW, linewidth=3.0, solid_capstyle="round", zorder=5)
    ax.plot(
        [x1, x1 - head],
        [y1, y1 + wing],
        color=COLOR_ARROW,
        linewidth=3.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    ax.plot(
        [x1, x1 - head],
        [y1, y1 - wing],
        color=COLOR_ARROW,
        linewidth=3.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )


def add_classifier_samples(ax: plt.Axes) -> None:
    x, y, width, height = SAMPLES_BOX
    add_rounded_box(
        ax,
        SAMPLES_BOX,
        edgecolor="#607d8b",
        facecolor="#ffffff",
        linewidth=1.4,
        radius=0.016,
        zorder=2,
    )

    rng = np.random.default_rng(7)
    green = np.array(
        [
            [0.12, 0.70],
            [0.20, 0.58],
            [0.32, 0.73],
            [0.42, 0.62],
            [0.53, 0.77],
            [0.69, 0.66],
            [0.77, 0.79],
            [0.84, 0.61],
            [0.64, 0.52],
        ]
    )
    red = np.array(
        [
            [0.12, 0.28],
            [0.22, 0.40],
            [0.33, 0.24],
            [0.43, 0.36],
            [0.56, 0.22],
            [0.66, 0.39],
            [0.79, 0.30],
            [0.86, 0.45],
            [0.50, 0.48],
        ]
    )
    green += rng.normal(0.0, 0.015, size=green.shape)
    red += rng.normal(0.0, 0.015, size=red.shape)

    def map_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = x + width * points[:, 0]
        py = y + height * points[:, 1]
        return px, py

    gx, gy = map_points(green)
    rx, ry = map_points(red)
    ax.scatter(gx, gy, s=38, color=COLOR_GREEN, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(rx, ry, s=38, color=COLOR_RED, edgecolor="white", linewidth=0.8, zorder=4)

    t = np.linspace(0.10, 0.88, 180)
    curve_x = x + width * t
    curve_y = y + height * (0.49 + 0.13 * np.sin(2.2 * np.pi * (t - 0.10)) - 0.07 * (t - 0.50))
    ax.plot(curve_x, curve_y, color="#202124", linewidth=2.0, solid_capstyle="round", zorder=5)


def draw_diagram(unsafe_path: Path, safe_path: Path, out_path: Path, dpi: int) -> None:
    if not unsafe_path.is_file():
        raise FileNotFoundError(f"Missing unsafe image: {unsafe_path}")
    if not safe_path.is_file():
        raise FileNotFoundError(f"Missing safe image: {safe_path}")

    fig = plt.figure(figsize=FIG_SIZE, dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    add_rounded_box(
        ax,
        OUTER_BOX,
        edgecolor=COLOR_OUTER_EDGE,
        facecolor=COLOR_OUTER_FACE,
        linewidth=2.0,
        radius=0.035,
        zorder=0,
    )
    ax.text(
        *TITLE_POS,
        "Conformal Obstacle Classifier",
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
        zorder=6,
    )

    add_rounded_box(
        ax,
        IMAGE_BOX,
        edgecolor=COLOR_IMAGE_EDGE,
        facecolor=COLOR_IMAGE_FACE,
        linewidth=1.8,
        radius=0.026,
        zorder=1,
    )
    add_image(ax, unsafe_path, UNSAFE_IMAGE)
    add_image(ax, safe_path, SAFE_IMAGE)

    add_vit_encoder(ax)
    add_narrow_arrow(ax)
    add_classifier_samples(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    draw_diagram(
        image_dir / str(args.unsafe_name),
        image_dir / str(args.safe_name),
        out_path,
        int(args.dpi),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
