#!/usr/bin/env python3
"""Sample Reacher IK configurations whose fingertip lies inside a small workspace circle."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from reacher.plan import obstacle_net_train as obstacle_train
from reacher.plan import plan_ilqr_mpc as planner

DEFAULT_OUT_DIR = "reacher/plan/circle_obstacle_sampling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--bend-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--circle-center-x", type=float, default=0.135)
    parser.add_argument("--circle-center-y", type=float, default=0.135)
    parser.add_argument("--circle-radius", type=float, default=0.035)
    parser.add_argument("--outside-count", type=int, default=2)
    parser.add_argument("--outside-margin", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    return parser.parse_args()


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def get_descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    descendants = {int(root_body_id)}
    parent_ids = np.asarray(model.body_parentid, dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            parent_id = int(parent_ids[body_id])
            if parent_id in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def infer_planar_arm_geometry(env: Any) -> dict[str, Any]:
    physics = env._env.physics
    model = physics.model
    with physics.reset_context():
        physics.data.qpos[:2] = 0.0
        physics.data.qvel[:2] = 0.0

    arm_body_id = int(model.name2id("arm", "body"))
    hand_body_id = int(model.name2id("hand", "body"))
    base_xy = np.asarray(physics.data.xpos[arm_body_id][:2], dtype=np.float64)
    hand_xy = np.asarray(physics.data.xpos[hand_body_id][:2], dtype=np.float64)
    hand_xy_local = hand_xy - base_xy
    link1 = float(np.linalg.norm(hand_xy_local))

    hand_descendants = get_descendant_body_ids(model, hand_body_id)
    candidates: list[tuple[float, np.ndarray, str]] = []

    for site_id in range(int(model.nsite)):
        name = model.id2name(site_id, "site")
        if name is None or "target" in name:
            continue
        if int(model.site_bodyid[site_id]) not in hand_descendants:
            continue
        xy_world = np.asarray(physics.data.site_xpos[site_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"site:{name}"))

    for geom_id in range(int(model.ngeom)):
        name = model.id2name(geom_id, "geom")
        if name is None or name in {"target", "root"}:
            continue
        if int(model.geom_bodyid[geom_id]) not in hand_descendants:
            continue
        xy_world = np.asarray(physics.data.geom_xpos[geom_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"geom:{name}"))

    for body_id in hand_descendants:
        xy_world = np.asarray(physics.data.xpos[body_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"body:{model.id2name(body_id, 'body')}"))

    if not candidates:
        raise RuntimeError("Failed to infer fingertip geometry from the DM Control reacher model.")

    link2, tip_xy, tip_source = max(candidates, key=lambda item: item[0])
    reach_min = abs(link1 - link2)
    reach_max = link1 + link2
    return {
        "base_xy": base_xy,
        "link1": link1,
        "link2": link2,
        "reach_min": reach_min,
        "reach_max": reach_max,
        "tip_source": tip_source,
        "tip_xy_at_zero": tip_xy,
    }


def resolve_circle_spec(
    geom: dict[str, Any],
    *,
    circle_center_x: float | None,
    circle_center_y: float | None,
    circle_radius: float | None,
) -> tuple[np.ndarray, float]:
    reach_min = float(geom["reach_min"])
    reach_max = float(geom["reach_max"])
    annulus_width = max(reach_max - reach_min, 1e-6)
    radius = float(circle_radius) if circle_radius is not None else min(0.03, 0.12 * annulus_width)
    radius = max(radius, 1e-4)

    default_center_radius = reach_min + 0.72 * max(reach_max - reach_min - radius, 1e-6)
    default_center_local = default_center_radius * np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0)
    center_local = np.array(
        [
            default_center_local[0] if circle_center_x is None else float(circle_center_x),
            default_center_local[1] if circle_center_y is None else float(circle_center_y),
        ],
        dtype=np.float64,
    )

    center_norm = float(np.linalg.norm(center_local))
    min_center_norm = reach_min + radius + 1e-4
    max_center_norm = reach_max - radius - 1e-4
    if center_norm < 1e-9:
        center_local = default_center_local.copy()
        center_norm = float(np.linalg.norm(center_local))
    if center_norm < min_center_norm:
        center_local *= min_center_norm / center_norm
    if center_norm > max_center_norm:
        center_local *= max_center_norm / center_norm
    if center_local[0] <= 0.0 or center_local[1] <= 0.0:
        center_local = np.abs(center_local)
        center_norm = float(np.linalg.norm(center_local))
        if center_norm > max_center_norm:
            center_local *= max_center_norm / center_norm
    center_world = np.asarray(geom["base_xy"], dtype=np.float64) + center_local
    return center_world, radius


def sample_points_in_circle(rng: np.random.Generator, center_xy: np.ndarray, radius: float, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=count)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    offsets = np.stack((radial * np.cos(theta), radial * np.sin(theta)), axis=1)
    return center_xy[None, :] + offsets


def sample_points_outside_circle(
    rng: np.random.Generator,
    *,
    center_xy: np.ndarray,
    radius: float,
    reach_min: float,
    reach_max: float,
    base_xy: np.ndarray,
    count: int,
    margin: float,
    max_tries: int = 4096,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    min_radius = radius + max(float(margin), 1e-4)
    max_radius = max(reach_max - 1e-4, min_radius + 1e-4)
    if min_radius >= max_radius:
        raise RuntimeError("Outside sampling region is empty; reduce outside margin or obstacle radius.")

    accepted: list[np.ndarray] = []
    tries = 0
    while len(accepted) < count and tries < max_tries:
        remaining = count - len(accepted)
        batch_size = min(max(remaining * 2, 64), max_tries - tries)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=batch_size)
        radial = np.sqrt(rng.uniform(min_radius * min_radius, max_radius * max_radius, size=batch_size))
        offsets = np.stack((radial * np.cos(theta), radial * np.sin(theta)), axis=1)
        candidates = center_xy[None, :] + offsets
        candidate_norms = np.linalg.norm(candidates - base_xy[None, :], axis=1)
        valid_mask = (candidate_norms >= reach_min + 1e-4) & (candidate_norms <= reach_max - 1e-4)
        accepted.extend(candidates[valid_mask])
        tries += batch_size

    if len(accepted) < count:
        raise RuntimeError(f"Failed to sample {count} outside-circle points after {tries} tries.")
    return np.stack(accepted[:count], axis=0)


def solve_two_link_ik(
    target_xy: np.ndarray,
    *,
    link1: float,
    link2: float,
    bend_sign: int,
) -> np.ndarray:
    x = float(target_xy[0])
    y = float(target_xy[1])
    radius_sq = x * x + y * y
    cos_theta2 = (radius_sq - link1 * link1 - link2 * link2) / (2.0 * link1 * link2)
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    sin_theta2 = float(bend_sign) * math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = math.atan2(y, x) - math.atan2(link2 * sin_theta2, link1 + link2 * cos_theta2)
    return np.array([wrap_angle(theta1), wrap_angle(theta2)], dtype=np.float64)


def qpos_within_limits(qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-6) -> bool:
    if np.any(np.isfinite(lower) & (qpos < lower - tol)):
        return False
    if np.any(np.isfinite(upper) & (qpos > upper + tol)):
        return False
    return True


def joint_limits_with_unbounded_fixed(env: Any) -> tuple[np.ndarray, np.ndarray]:
    model = env._env.physics.model
    raw = np.asarray(model.jnt_range[:2], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[:2], dtype=bool)
    lower = raw[:, 0].copy()
    upper = raw[:, 1].copy()
    lower[~limited] = -np.inf
    upper[~limited] = np.inf
    return lower, upper


def forward_kinematics(qpos: np.ndarray, *, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray]:
    theta1 = float(qpos[0])
    theta2 = float(qpos[1])
    elbow_xy = np.array([link1 * math.cos(theta1), link1 * math.sin(theta1)], dtype=np.float64)
    tip_xy = elbow_xy + np.array(
        [link2 * math.cos(theta1 + theta2), link2 * math.sin(theta1 + theta2)],
        dtype=np.float64,
    )
    return elbow_xy, tip_xy


def solve_ik_batch(
    targets_xy: np.ndarray,
    *,
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_list: list[np.ndarray] = []
    elbow_list: list[np.ndarray] = []
    tip_list: list[np.ndarray] = []
    for target_xy_world in targets_xy:
        target_xy_local = np.asarray(target_xy_world, dtype=np.float64) - base_xy
        qpos = solve_two_link_ik(target_xy_local, link1=link1, link2=link2, bend_sign=bend_sign)
        if not qpos_within_limits(qpos, lower, upper):
            continue
        elbow_xy_local, tip_xy_local = forward_kinematics(qpos, link1=link1, link2=link2)
        if np.linalg.norm(tip_xy_local - target_xy_local) > 1e-5:
            continue
        elbow_xy = base_xy + elbow_xy_local
        tip_xy = base_xy + tip_xy_local
        qpos_list.append(qpos)
        elbow_list.append(elbow_xy)
        tip_list.append(tip_xy)
    if not qpos_list:
        raise RuntimeError("No valid IK solutions survived the joint-limit and reconstruction checks.")
    return np.stack(qpos_list, axis=0), np.stack(elbow_list, axis=0), np.stack(tip_list, axis=0)


def save_workspace_plot(
    *,
    out_path: Path,
    geom: dict[str, Any],
    circle_center_xy: np.ndarray,
    circle_radius: float,
    center_qpos: np.ndarray,
    sampled_elbows_xy: np.ndarray,
    sampled_tips_xy: np.ndarray,
    outside_elbows_xy: np.ndarray,
    outside_tips_xy: np.ndarray,
    bend_sign: int,
) -> None:
    base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)

    reach_max = float(geom["reach_max"])
    reach_min = float(geom["reach_min"])
    outer = plt.Circle(base_xy, reach_max, color="#d9d9d9", fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(outer)
    if reach_min > 1e-4:
        inner = plt.Circle(base_xy, reach_min, color="#ececec", fill=False, linestyle=":", linewidth=1.0)
        ax.add_patch(inner)

    obstacle = plt.Circle(circle_center_xy, circle_radius, color="#d55e00", fill=False, linewidth=2.0)
    ax.add_patch(obstacle)

    for elbow_point, tip_point in zip(sampled_elbows_xy, sampled_tips_xy, strict=True):
        ax.plot(
            [base_xy[0], elbow_point[0]],
            [base_xy[1], elbow_point[1]],
            color="#111111",
            linewidth=0.8,
            alpha=0.12,
        )
        ax.plot(
            [elbow_point[0], tip_point[0]],
            [elbow_point[1], tip_point[1]],
            color="#111111",
            linewidth=0.8,
            alpha=0.12,
        )

    outside_colors = ("#009e73", "#0072b2")
    for idx, (elbow_point, tip_point) in enumerate(zip(outside_elbows_xy, outside_tips_xy, strict=True)):
        color = outside_colors[idx % len(outside_colors)]
        ax.plot(
            [base_xy[0], elbow_point[0]],
            [base_xy[1], elbow_point[1]],
            color=color,
            linewidth=1.2,
            alpha=0.6,
        )
        ax.plot(
            [elbow_point[0], tip_point[0]],
            [elbow_point[1], tip_point[1]],
            color=color,
            linewidth=1.2,
            alpha=0.6,
        )

    ax.scatter(sampled_tips_xy[:, 0], sampled_tips_xy[:, 1], s=10, c="#d55e00", alpha=0.5, label="inside tips")
    ax.scatter(sampled_elbows_xy[:, 0], sampled_elbows_xy[:, 1], s=10, c="#0072b2", alpha=0.35, label="inside elbows")
    if outside_tips_xy.size > 0:
        ax.scatter(
            outside_tips_xy[:, 0],
            outside_tips_xy[:, 1],
            s=48,
            marker="x",
            c="#009e73",
            linewidths=1.5,
            label="outside tips",
        )
    if outside_elbows_xy.size > 0:
        ax.scatter(
            outside_elbows_xy[:, 0],
            outside_elbows_xy[:, 1],
            s=28,
            marker="D",
            c="#56b4e9",
            alpha=0.8,
            label="outside elbows",
        )

    center_elbow_xy_local, center_tip_xy_local = forward_kinematics(
        center_qpos,
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
    )
    center_elbow_xy = base_xy + center_elbow_xy_local
    center_tip_xy = base_xy + center_tip_xy_local
    ax.plot([base_xy[0], center_elbow_xy[0]], [base_xy[1], center_elbow_xy[1]], color="#111111", linewidth=2.2)
    ax.plot(
        [center_elbow_xy[0], center_tip_xy[0]],
        [center_elbow_xy[1], center_tip_xy[1]],
        color="#000000",
        linewidth=2.2,
        label="nominal center IK",
    )
    ax.scatter(
        [base_xy[0], center_elbow_xy[0], center_tip_xy[0]],
        [base_xy[1], center_elbow_xy[1], center_tip_xy[1]],
        c="#111111",
        s=18,
    )

    ax.set_aspect("equal", adjustable="box")
    lim = reach_max + 0.04
    ax.set_xlim(base_xy[0] - lim, base_xy[0] + lim)
    ax.set_ylim(base_xy[1] - lim, base_xy[1] + lim)
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Circle obstacle samples, bend_sign={bend_sign}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = planner.make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        geom = infer_planar_arm_geometry(env)
        lower, upper = joint_limits_with_unbounded_fixed(env)
    finally:
        env.close()

    circle_center_xy, circle_radius = resolve_circle_spec(
        geom,
        circle_center_x=args.circle_center_x,
        circle_center_y=args.circle_center_y,
        circle_radius=args.circle_radius,
    )
    targets_xy = sample_points_in_circle(rng, circle_center_xy, circle_radius, int(args.sample_count))
    qpos_batch, elbow_xy, tip_xy = solve_ik_batch(
        targets_xy,
        base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.bend_sign),
        lower=lower,
        upper=upper,
    )
    outside_targets_xy = sample_points_outside_circle(
        rng,
        center_xy=circle_center_xy,
        radius=float(circle_radius),
        reach_min=float(geom["reach_min"]),
        reach_max=float(geom["reach_max"]),
        base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
        count=int(args.outside_count),
        margin=float(args.outside_margin),
    )
    outside_qpos, outside_elbow_xy, outside_tip_xy = solve_ik_batch(
        outside_targets_xy,
        base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.bend_sign),
        lower=lower,
        upper=upper,
    )
    if outside_qpos.shape[0] < 2:
        raise RuntimeError("Need at least 2 outside configurations to define start and goal.")
    start_qpos = np.asarray(outside_qpos[0], dtype=np.float32)
    goal_qpos = np.asarray(outside_qpos[1], dtype=np.float32)
    start_tip_xy = np.asarray(outside_tip_xy[0], dtype=np.float32)
    goal_tip_xy = np.asarray(outside_tip_xy[1], dtype=np.float32)
    start_qvel = np.zeros_like(start_qpos, dtype=np.float32)
    goal_qvel = np.zeros_like(goal_qpos, dtype=np.float32)
    circle_center_local = circle_center_xy - np.asarray(geom["base_xy"], dtype=np.float64)
    center_qpos = solve_two_link_ik(
        circle_center_local,
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.bend_sign),
    )
    if not qpos_within_limits(center_qpos, lower, upper):
        raise RuntimeError("The center-circle IK solution violates the env joint limits. Pick another bend sign or circle.")

    rollout_stub = {
        "episode_seed": int(args.seed),
        "time_limit": float(args.time_limit),
        "width": int(args.width),
        "height": int(args.height),
        "physics_freq_hz": float(args.physics_freq_hz),
    }
    obstacle_train.save_obstacle_overlay(
        planner_module=planner,
        rollout=rollout_stub,
        center_qpos=center_qpos,
        obstacle_qpos=np.concatenate((qpos_batch, outside_qpos), axis=0),
        out_path=out_dir / "obstacle_overlay_all.png",
    )
    planner.save_torch_payload(
        out_dir / "planner_start_goal_obstacle.pt",
        {
            "metadata": {
                "episode_seed": int(args.seed),
                "time_limit": float(args.time_limit),
                "physics_freq_hz": float(args.physics_freq_hz),
                "height": int(args.height),
                "width": int(args.width),
                "bend_sign": int(args.bend_sign),
                "start_label": "green arm / start pos",
                "goal_label": "blue arm / goal pos",
                "sample_count_valid": int(qpos_batch.shape[0]),
                "outside_count_valid": int(outside_qpos.shape[0]),
            },
            "episode_data": {
                "qpos": np.stack((start_qpos, goal_qpos), axis=0),
                "qvel": np.stack((start_qvel, goal_qvel), axis=0),
            },
            "planner_data": {
                "start_qpos": start_qpos,
                "goal_qpos": goal_qpos,
                "obstacle_center_qpos": np.asarray(center_qpos, dtype=np.float32),
                "obstacle_qpos": np.asarray(qpos_batch, dtype=np.float32),
                "start_tip_xy": start_tip_xy,
                "goal_tip_xy": goal_tip_xy,
                "circle_center_xy": np.asarray(circle_center_xy, dtype=np.float32),
                "circle_radius": float(circle_radius),
            },
        },
    )

    save_workspace_plot(
        out_path=out_dir / "workspace_samples.png",
        geom=geom,
        circle_center_xy=circle_center_xy,
        circle_radius=circle_radius,
        center_qpos=center_qpos,
        sampled_elbows_xy=elbow_xy,
        sampled_tips_xy=tip_xy,
        outside_elbows_xy=outside_elbow_xy,
        outside_tips_xy=outside_tip_xy,
        bend_sign=int(args.bend_sign),
    )
    save_json(
        out_dir / "summary.json",
        {
            "seed": int(args.seed),
            "sample_count_requested": int(args.sample_count),
            "sample_count_valid": int(qpos_batch.shape[0]),
            "outside_count_requested": int(args.outside_count),
            "outside_count_valid": int(outside_qpos.shape[0]),
            "outside_margin": float(args.outside_margin),
            "bend_sign": int(args.bend_sign),
            "circle_center_xy": circle_center_xy.tolist(),
            "circle_radius": float(circle_radius),
            "link1": float(geom["link1"]),
            "link2": float(geom["link2"]),
            "reach_min": float(geom["reach_min"]),
            "reach_max": float(geom["reach_max"]),
            "tip_source": str(geom["tip_source"]),
            "center_qpos": center_qpos.tolist(),
            "start_qpos": start_qpos.tolist(),
            "goal_qpos": goal_qpos.tolist(),
            "start_tip_xy": start_tip_xy.tolist(),
            "goal_tip_xy": goal_tip_xy.tolist(),
            "outside_tip_xy": outside_tip_xy.tolist(),
            "outside_qpos": outside_qpos.tolist(),
        },
    )

    print(f"Saved overlay:   {out_dir / 'obstacle_overlay_all.png'}")
    print(f"Saved workspace: {out_dir / 'workspace_samples.png'}")
    print(f"Saved planner:   {out_dir / 'planner_start_goal_obstacle.pt'}")
    print(f"Saved summary:   {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
