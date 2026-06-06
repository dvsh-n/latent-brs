#!/usr/bin/env python3
"""Replay saved PushT PLDM summaries and render missing rollout videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pusht.plan import plan_ilqr_mpc as pusht_base


DEFAULT_DATASET_PATH = ROOT_DIR / "pusht" / "data" / "train_data" / "pusht_diffusion_train_combined.h5"
DEFAULT_RESULT_DIR = ROOT_DIR / "pusht" / "plan" / "pusht_hard_eval" / "pldm_result_5seeds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def replay_case(summary_path: Path, dataset_path: Path, fps: int, overwrite: bool) -> Path | None:
    case_dir = summary_path.parent
    mp4_path = case_dir / "rollout.mp4"
    gif_path = case_dir / "rollout.gif"
    if not overwrite and (mp4_path.exists() or gif_path.exists()):
        return mp4_path if mp4_path.exists() else gif_path

    summary = load_summary(summary_path)
    actions = np.asarray(summary.get("executed_actions_env", []), dtype=np.float32)
    if actions.size == 0:
        print(f"skip no actions: {summary_path}")
        return None

    episode = pusht_base.load_dataset_episode(dataset_path, int(summary["episode_idx"]))
    state_np = np.asarray(episode["state"], dtype=np.float32)
    proprio_np = np.asarray(episode["proprio"], dtype=np.float32)
    height = int(episode["height"])
    width = int(episode["width"])
    start_step = int(summary.get("start_step", 0))

    state_format = summary.get("dataset_state_format") or pusht_base.infer_dataset_state_format(state_np[start_step])
    env_state_np = np.stack(
        [
            pusht_base.dataset_row_to_env_state(state_row, proprio_row, state_format)
            for state_row, proprio_row in zip(state_np, proprio_np)
        ],
        axis=0,
    )

    goal_pose = summary.get("goal_pose")
    viz_env = pusht_base.make_visualization_env(width=width, height=height)
    frames: list[np.ndarray] = []
    try:
        pusht_base.set_goal_pose(viz_env, None if goal_pose is None else np.asarray(goal_pose, dtype=np.float32))
        frame = pusht_base.reset_env_to_state(viz_env.unwrapped, env_state_np[start_step])
        frames.append(np.asarray(frame, dtype=np.uint8).copy())
        for action in actions.reshape(-1, 2):
            viz_env.step(action.astype(np.float32))
            current_state = pusht_base.extract_full_state(viz_env.unwrapped)
            frame = pusht_base.reset_env_to_state(viz_env.unwrapped, current_state)
            frames.append(np.asarray(frame, dtype=np.uint8).copy())
        video_path = pusht_base.save_rollout_video(frames, case_dir, fps=fps)
    finally:
        viz_env.close()

    summary["video_path"] = str(video_path.resolve())
    save_summary(summary_path, summary)
    print(video_path)
    return video_path


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result dir not found: {result_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    summaries = sorted(result_dir.glob("*/case_*/summary.json"))
    if args.limit is not None:
        summaries = summaries[: int(args.limit)]
    rendered = 0
    for summary_path in summaries:
        if replay_case(summary_path, dataset_path, int(args.fps), bool(args.overwrite)) is not None:
            rendered += 1
    print(f"rendered_or_existing={rendered} summaries={len(summaries)}")


if __name__ == "__main__":
    main()
