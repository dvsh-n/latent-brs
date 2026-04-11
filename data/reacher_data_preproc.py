#!/usr/bin/env python3
"""Preprocess Reacher expert rollouts to match Temporal Straightening image inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision import transforms
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "test_data" / "test_data.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "test_data" / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_video_frames(video_path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[torch.Tensor] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video {video_path} contains no frames.")
    return torch.stack(frames, dim=0)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    obses_dir = output_dir / "obses"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory {output_dir} is not empty. Pass --overwrite to reuse it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    obses_dir.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(dataset_path, map_location="cpu")
    states = dataset["states"].to(torch.float32).contiguous()
    actions = dataset["actions"].to(torch.float32).contiguous()
    video_dir = Path(dataset["video_dir"]).expanduser().resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    num_trajectories = int(dataset["num_trajectories"])
    frames_per_trajectory = int(dataset["steps_per_episode"]) + 1
    transform = build_transform(args.img_size)
    seq_lengths = torch.full((num_trajectories,), frames_per_trajectory, dtype=torch.long)

    for traj_idx in tqdm(range(num_trajectories), desc="Preprocessing trajectories", unit="traj"):
        video_path = video_dir / f"trajectory_{traj_idx:07d}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing trajectory video: {video_path}")
        frames = load_video_frames(video_path)
        if int(frames.shape[0]) != frames_per_trajectory:
            raise RuntimeError(
                f"{video_path.name} yielded {frames.shape[0]} frames, expected {frames_per_trajectory}."
            )
        processed = transform(frames)
        torch.save(processed.contiguous(), obses_dir / f"episode_{traj_idx:05d}.pth")

    padded_actions = torch.cat([actions, torch.zeros_like(actions[:, :1])], dim=1)
    torch.save(states, output_dir / "states.pth")
    torch.save(actions, output_dir / "actions.pth")
    torch.save(padded_actions, output_dir / "actions_padded.pth")
    torch.save(seq_lengths, output_dir / "seq_lengths.pth")

    metadata = {
        "source_dataset_path": str(dataset_path),
        "video_dir": str(video_dir),
        "num_trajectories": num_trajectories,
        "frames_per_trajectory": frames_per_trajectory,
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "actions_padded_shape": list(padded_actions.shape),
        "img_size": args.img_size,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "transform": ["Resize", "CenterCrop", "Normalize"],
    }
    torch.save(metadata, output_dir / "metadata.pt")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
