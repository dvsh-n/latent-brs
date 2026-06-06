#!/usr/bin/env python3
"""Re-render Reacher rollout videos from saved qpos/qvel trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from reacher.plan import plan_ilqr_mpc as ilqr_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", default="lpb_guided")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-name", default="reconstructed_rollout.mp4")
    return parser.parse_args()


def render_trajectory(
    *,
    qpos_trajectory: list[list[float]],
    qvel_trajectory: list[list[float]],
    seed: int,
    height: int,
    width: int,
) -> list[np.ndarray]:
    env = ilqr_base.make_render_env(
        seed=seed,
        time_limit=10.0,
        width=width,
        height=height,
        physics_freq_hz=100.0,
    )
    frames: list[np.ndarray] = []
    try:
        physics = env._env.physics
        for qpos_raw, qvel_raw in zip(qpos_trajectory, qvel_trajectory):
            qpos = np.asarray(qpos_raw, dtype=np.float32)
            qvel = np.asarray(qvel_raw, dtype=np.float32)
            with physics.reset_context():
                physics.data.qpos[: qpos.shape[0]] = qpos
                physics.data.qvel[: qvel.shape[0]] = qvel
            frames.append(physics.render(height=height, width=width, camera_id=0))
    finally:
        env.close()
    return frames


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    label_dir = run_dir / args.label
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    summary_paths = sorted(label_dir.glob("case_*/summary.json"))
    if args.limit is not None:
        summary_paths = summary_paths[: int(args.limit)]
    if not summary_paths:
        raise FileNotFoundError(f"No case summaries found under {label_dir}")

    written = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        qpos_trajectory = summary.get("qpos_trajectory")
        qvel_trajectory = summary.get("qvel_trajectory")
        if not qpos_trajectory or not qvel_trajectory:
            raise ValueError(f"Missing qpos/qvel trajectory in {summary_path}")
        frames = render_trajectory(
            qpos_trajectory=qpos_trajectory,
            qvel_trajectory=qvel_trajectory,
            seed=int(summary.get("episode_seed", 0)),
            height=int(args.height),
            width=int(args.width),
        )
        out_path = summary_path.parent / args.out_name
        imageio.mimwrite(out_path, frames, fps=int(args.fps), quality=8, macro_block_size=1)
        written.append(str(out_path))

    manifest_path = run_dir / f"reconstructed_{args.label}_videos.json"
    manifest_path.write_text(json.dumps({"videos": written}, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(manifest_path), "videos": written[:5]}, indent=2))


if __name__ == "__main__":
    main()
