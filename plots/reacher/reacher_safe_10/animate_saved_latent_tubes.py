#!/usr/bin/env python3
"""Animate saved Reacher SLS latent tubes."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_saved_latent_tubes import GREEN, GREEN_DARK, alpha_for_order, parse_dims, resolve_tube_data

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 300,
    }
)


def fill_tube(ax, horizon_x, lower, upper, *, color, alpha, horizon_alpha_decay):
    if horizon_x.shape[0] < 2:
        ax.fill_between(horizon_x, lower, upper, color=color, alpha=alpha, linewidth=0.0)
        return
    for idx in range(horizon_x.shape[0] - 1):
        seg = slice(idx, idx + 2)
        seg_alpha = float(max(0.01, min(1.0, alpha * (horizon_alpha_decay ** idx))))
        ax.fill_between(horizon_x[seg], lower[seg], upper[seg], color=color, alpha=seg_alpha, linewidth=0.0)


def draw_tube_plan(
    ax,
    *,
    plan_step: int,
    center: np.ndarray,
    width: np.ndarray,
    fill_color: str,
    fill_alpha: float,
    horizon_alpha_decay: float,
    draw_center_line: bool,
    line_color: str,
    clip_after_x: float | None = None,
) -> None:
    horizon_x = int(plan_step) + np.arange(center.shape[0])
    valid = np.isfinite(center) & np.isfinite(width)
    if clip_after_x is not None:
        valid = valid & (horizon_x <= float(clip_after_x))
    if not np.any(valid):
        return
    horizon_x = horizon_x[valid]
    center = center[valid]
    width = np.maximum(width[valid], 0.0)
    fill_tube(
        ax,
        horizon_x,
        center - width,
        center + width,
        color=fill_color,
        alpha=fill_alpha,
        horizon_alpha_decay=horizon_alpha_decay,
    )
    if draw_center_line:
        ax.plot(horizon_x, center, color=line_color, linestyle=":", linewidth=1.2, alpha=0.85)


def selected_plan_indices(plan_steps: np.ndarray, start_step: int, plan_stride: int, max_plans: int | None) -> np.ndarray:
    if plan_stride <= 0:
        raise ValueError("--plan-stride must be positive.")
    selected = np.flatnonzero(plan_steps >= int(start_step))[:: int(plan_stride)]
    if max_plans is not None:
        selected = selected[: int(max_plans)]
    if selected.size == 0:
        raise ValueError("No plans selected for animation.")
    return selected


def data_axis_limits(executed: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state_dim = centers.shape[-1]
    stacked = np.concatenate(
        [
            executed,
            centers.reshape(-1, state_dim),
            (centers - widths).reshape(-1, state_dim),
            (centers + widths).reshape(-1, state_dim),
        ],
        axis=0,
    )
    low = np.nanmin(stacked, axis=0)
    high = np.nanmax(stacked, axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - 0.08 * span, high + 0.08 * span


def render_animation(args: argparse.Namespace, dims: list[int], out_path: Path) -> Path:
    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)
    state_dim = int(np.asarray(data["state_dim"]))

    selected = selected_plan_indices(plan_steps, args.start_step, args.plan_stride, args.max_plans)
    step_to_plan_idx = {int(step): int(idx) for idx, step in enumerate(plan_steps)}
    low, high = data_axis_limits(executed, centers, widths)

    full_markov_dims = dims == list(range(min(10, state_dim)))
    latent_only_dims = dims == list(range(min(5, state_dim)))
    n_cols = 5 if (full_markov_dims or latent_only_dims) else min(5, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig_height = 3.0 if full_markov_dims else 1.75
    exec_x = np.arange(executed.shape[0])

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, quality=args.quality, macro_block_size=1)
    try:
        for frame_step in range(executed.shape[0]):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.45 * n_cols, fig_height), dpi=args.dpi, sharex=True)
            axes = np.atleast_1d(axes).reshape(-1)
            persistent = [int(idx) for idx in selected if int(plan_steps[idx]) <= frame_step]
            current_idx = step_to_plan_idx.get(frame_step)
            show_current = (
                current_idx is not None
                and args.show_current_plan
                and frame_step < executed.shape[0] - 1
            )
            current_start_x = frame_step if show_current else None
            if show_current and current_idx is not None:
                persistent = [idx for idx in persistent if idx != current_idx]

            for panel_idx, dim in enumerate(dims):
                ax = axes[panel_idx]
                ax.plot(
                    exec_x[: frame_step + 1],
                    executed[: frame_step + 1, dim],
                    color=GREEN_DARK,
                    linewidth=2.0,
                    label="executed" if panel_idx == 0 else None,
                )
                for order, plan_idx in enumerate(persistent):
                    draw_tube_plan(
                        ax,
                        plan_step=int(plan_steps[plan_idx]),
                        center=centers[plan_idx, :, dim],
                        width=widths[plan_idx, :, dim],
                        fill_color=GREEN,
                        fill_alpha=alpha_for_order(order, len(selected), args.alpha, args.alpha_decay),
                        horizon_alpha_decay=args.horizon_alpha_decay,
                        draw_center_line=False,
                        line_color=GREEN_DARK,
                        clip_after_x=current_start_x,
                    )
                if show_current and current_idx is not None:
                    draw_tube_plan(
                        ax,
                        plan_step=int(plan_steps[current_idx]),
                        center=centers[current_idx, :, dim],
                        width=widths[current_idx, :, dim],
                        fill_color=args.current_color,
                        fill_alpha=args.current_alpha,
                        horizon_alpha_decay=args.horizon_alpha_decay,
                        draw_center_line=True,
                        line_color=args.current_line_color,
                    )
                ax.text(
                    0.97,
                    0.91,
                    f"Dim. {dim}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=13,
                    color="0.15",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
                )
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.set_ylim(float(low[dim]), float(high[dim]))

            for ax in axes[len(dims) :]:
                ax.axis("off")

            axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
            y_label = args.ylabel
            if y_label is None:
                y_label = "Latent Rollout" if latent_only_dims else "Markovian Latent Rollout"
            fig.supxlabel("MPC step", y=0.005, fontsize=13)
            fig.supylabel(y_label, x=0.004, fontsize=13)
            fig.subplots_adjust(
                left=0.04,
                right=0.995,
                bottom=0.34 if latent_only_dims else 0.22,
                top=0.96 if latent_only_dims else 0.975,
                wspace=0.34,
                hspace=0.30,
            )

            fig.canvas.draw()
            width_px, height_px = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height_px, width_px, 4)
            rgb = frame[:, :, :3]
            pad_h = rgb.shape[0] % 2
            pad_w = rgb.shape[1] % 2
            if pad_h or pad_w:
                rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            writer.append_data(rgb.copy())
            plt.close(fig)
    finally:
        writer.close()

    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tube_data", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--plan-stride", type=int, default=3)
    parser.add_argument("--max-plans", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--alpha-decay", type=float, default=1.00)
    parser.add_argument("--horizon-alpha-decay", type=float, default=0.89)
    parser.add_argument("--dims", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--show-current-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--current-alpha", type=float, default=0.34)
    parser.add_argument("--current-color", type=str, default="#fdae6b")
    parser.add_argument("--current-line-color", type=str, default="#d95f0e")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quality", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    state_dim = int(np.asarray(data["state_dim"]))
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else tube_path.parent

    if args.dims is not None or args.out is not None:
        dims = parse_dims(args.dims, state_dim)
        out_path = args.out
        if out_path is None:
            out_path = out_dir / f"reacher_animated_latent_tubes_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.mp4"
        saved = render_animation(args, dims, out_path)
        print(f"Saved Reacher latent tube animation to {saved}")
        return

    specs = (
        ("2x5", list(range(min(10, state_dim)))),
        ("first_half", list(range(min(5, state_dim)))),
    )
    for suffix, dims in specs:
        out_path = out_dir / f"reacher_animated_latent_tubes_{suffix}_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.mp4"
        saved = render_animation(args, dims, out_path)
        print(f"Saved Reacher latent tube animation to {saved}")


if __name__ == "__main__":
    main()
