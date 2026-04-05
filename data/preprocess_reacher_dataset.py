#!/usr/bin/env python3
"""Preprocess streamed Reacher videos into chunked frame tensors for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_streamed_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        header = torch.load(handle, map_location="cpu", weights_only=False)
        meta = torch.load(handle, map_location="cpu", weights_only=False)
        while True:
            try:
                records.append(torch.load(handle, map_location="cpu", weights_only=False))
            except EOFError:
                break
    return header, meta, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/reacher_dataset.pt"),
        help="Input streamed dataset .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reacher_dataset_processed"),
        help="Directory where chunk files will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/reacher_dataset_processed_manifest.pt"),
        help="Manifest file describing the processed dataset.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize all frames to image_size x image_size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="Number of trajectories per processed chunk file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite manifest and chunks if they already exist.",
    )
    return parser.parse_args()


def default_video_path(dataset_path: Path, trajectory_index: int) -> Path:
    video_dir = dataset_path.with_name(f"{dataset_path.stem}_videos")
    return video_dir / f"trajectory_{trajectory_index:07d}.mp4"


def resolve_video_path(dataset_path: Path, record: dict[str, Any]) -> Path:
    video_path_value = record.get("video_path")
    if video_path_value:
        path = (dataset_path.parent / Path(str(video_path_value))).resolve()
        if path.exists():
            return path
    fallback = default_video_path(dataset_path, int(record["trajectory_index"])).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not resolve video for trajectory {int(record['trajectory_index'])} from {dataset_path}"
    )


def read_and_resize_video(video_path: Path, image_size: int) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    try:
        frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in reader]
    finally:
        reader.close()

    video = torch.stack(frames, dim=0).float()
    if video.shape[-2:] != (image_size, image_size):
        video = F.interpolate(
            video,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return video.round().clamp_(0, 255).to(torch.uint8)


def flush_chunk(
    output_dir: Path,
    chunk_index: int,
    chunk_records: list[dict[str, Any]],
    chunk_frames: list[torch.Tensor],
    chunk_actions: list[torch.Tensor],
) -> dict[str, Any]:
    chunk_path = output_dir / f"chunk_{chunk_index:05d}.pt"
    frames_tensor = torch.stack(chunk_frames, dim=0).contiguous()
    actions_tensor = torch.stack(chunk_actions, dim=0).contiguous()
    chunk_data = {
        "trajectory_indices": torch.tensor(
            [int(record["trajectory_index"]) for record in chunk_records],
            dtype=torch.int64,
        ),
        "frames": frames_tensor,
        "actions": actions_tensor,
    }
    torch.save(chunk_data, chunk_path)
    return {
        "chunk_path": str(chunk_path.resolve()),
        "num_trajectories": len(chunk_records),
        "trajectory_indices": [int(record["trajectory_index"]) for record in chunk_records],
        "num_frames": int(frames_tensor.shape[1]),
        "num_actions": int(actions_tensor.shape[1]),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.image_size <= 0:
        raise ValueError("image_size must be positive")

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_manifest = args.output_manifest.resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty; pass --overwrite to replace it")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{output_manifest} already exists; pass --overwrite to replace it")

    output_dir.mkdir(parents=True, exist_ok=True)
    header, meta, records = load_streamed_dataset(input_path)

    chunk_records: list[dict[str, Any]] = []
    chunk_frames: list[torch.Tensor] = []
    chunk_actions: list[torch.Tensor] = []
    chunk_entries: list[dict[str, Any]] = []

    chunk_index = 0
    for record in tqdm(records, desc="Preprocess trajectories", dynamic_ncols=True):
        video_path = resolve_video_path(input_path, record)
        frames = read_and_resize_video(video_path, image_size=args.image_size)
        actions = record["actions"].to(torch.float32).contiguous()

        chunk_records.append(record)
        chunk_frames.append(frames)
        chunk_actions.append(actions)

        if len(chunk_records) == args.chunk_size:
            chunk_entries.append(
                flush_chunk(
                    output_dir=output_dir,
                    chunk_index=chunk_index,
                    chunk_records=chunk_records,
                    chunk_frames=chunk_frames,
                    chunk_actions=chunk_actions,
                )
            )
            chunk_records = []
            chunk_frames = []
            chunk_actions = []
            chunk_index += 1

    if chunk_records:
        chunk_entries.append(
            flush_chunk(
                output_dir=output_dir,
                chunk_index=chunk_index,
                chunk_records=chunk_records,
                chunk_frames=chunk_frames,
                chunk_actions=chunk_actions,
            )
        )

    manifest = {
        "format": "reacher_processed_video_chunks_v1",
        "source_dataset": str(input_path),
        "source_header": header,
        "source_meta": meta,
        "image_size": args.image_size,
        "image_channels": 3,
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunk_entries),
        "num_trajectories": len(records),
        "chunks": chunk_entries,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, output_manifest)

    print(
        {
            "input": str(input_path),
            "output_manifest": str(output_manifest),
            "output_dir": str(output_dir),
            "num_chunks": len(chunk_entries),
            "num_trajectories": len(records),
            "image_size": args.image_size,
            "chunk_size": args.chunk_size,
        }
    )


if __name__ == "__main__":
    main()
