#!/usr/bin/env python3
"""Encode Reacher rollout frames with DINOv2 and save aligned latent tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DINOV2_REPO = REPO_ROOT / "third_party" / "dinov2"
if str(DINOV2_REPO) not in sys.path:
    sys.path.insert(0, str(DINOV2_REPO))

import hubconf  # noqa: E402


PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "expert_data" / "expert_data.pt"

MODEL_CONFIGS = {
    "vits14": {
        "hub_fn": "dinov2_vits14",
        "weights": REPO_ROOT / "models" / "dinov2_vits14_pretrain.pth",
        "embed_dim": 384,
    },
    "vits14_reg": {
        "hub_fn": "dinov2_vits14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vits14_reg4_pretrain.pth",
        "embed_dim": 384,
    },
    "vitb14": {
        "hub_fn": "dinov2_vitb14",
        "weights": REPO_ROOT / "models" / "dinov2_vitb14_pretrain.pth",
        "embed_dim": 768,
    },
    "vitb14_reg": {
        "hub_fn": "dinov2_vitb14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vitb14_reg4_pretrain.pth",
        "embed_dim": 768,
    },
    "vitl14": {
        "hub_fn": "dinov2_vitl14",
        "weights": REPO_ROOT / "models" / "dinov2_vitl14_pretrain.pth",
        "embed_dim": 1024,
    },
    "vitl14_reg": {
        "hub_fn": "dinov2_vitl14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vitl14_reg4_pretrain.pth",
        "embed_dim": 1024,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="vits14",
        help="DINOv2 backbone to use.",
    )
    parser.add_argument(
        "--feature-type",
        choices=("cls", "avg_patch", "cls_and_avg_patch", "all_tokens"),
        default="all_tokens",
        help="Which DINOv2 latent to save for each frame.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of trajectories to buffer on CPU before writing a chunk to disk.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset_dir>/latents/dinov2_<model>_<feature_type>",
    )
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def compute_input_size(video_dir: Path, patch_size: int = PATCH_SIZE) -> int:
    first_video = next(iter(sorted(video_dir.glob("*.mp4"))), None)
    if first_video is None:
        raise FileNotFoundError(f"No .mp4 files found in {video_dir}")
    cap = cv2.VideoCapture(str(first_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    side = min(width, height)
    input_size = (side // patch_size) * patch_size
    if input_size <= 0:
        raise RuntimeError(f"Invalid derived input size from video {first_video}")
    return input_size


def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    cfg = MODEL_CONFIGS[model_name]
    weights_path = Path(cfg["weights"])
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing DINOv2 weights: {weights_path}")
    model = getattr(hubconf, str(cfg["hub_fn"]))(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, cfg


def iter_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Video {video_path} did not yield any frames.")
    return frames


def preprocess_batch(frames: list[np.ndarray], input_size: int, device: torch.device) -> torch.Tensor:
    tensors = []
    for frame in frames:
        resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(img).permute(2, 0, 1))
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def encode_video(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    *,
    input_size: int,
    batch_size: int,
    feature_type: str,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        batch = preprocess_batch(batch_frames, input_size=input_size, device=device)
        features = model.forward_features(batch)
        cls_tokens = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]

        if feature_type == "cls":
            batch_latents = cls_tokens
        elif feature_type == "avg_patch":
            batch_latents = patch_tokens.mean(dim=1)
        elif feature_type == "cls_and_avg_patch":
            batch_latents = torch.cat((cls_tokens, patch_tokens.mean(dim=1)), dim=-1)
        elif feature_type == "all_tokens":
            batch_latents = torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        outputs.append(batch_latents.cpu())

    return torch.cat(outputs, dim=0)


def infer_output_dir(dataset_path: Path, model_name: str, feature_type: str) -> Path:
    return dataset_path.parent / "latents" / f"dinov2_{model_name}_{feature_type}"


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    device = require_device(args.device)
    dataset = torch.load(dataset_path, map_location="cpu")
    video_dir = Path(dataset["video_dir"]).expanduser().resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    num_trajectories = int(dataset["num_trajectories"])
    expected_frames = int(dataset["steps_per_episode"]) + 1
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    video_paths = [video_dir / f"trajectory_{traj_idx:07d}.mp4" for traj_idx in range(num_trajectories)]
    missing_videos = [str(path) for path in video_paths if not path.is_file()]
    if missing_videos:
        raise FileNotFoundError(
            f"Expected {num_trajectories} videos, but {len(missing_videos)} are missing. "
            f"First missing: {missing_videos[0]}"
        )

    input_size = compute_input_size(video_dir)
    model, cfg = load_model(args.model, device)
    tokens_per_side = input_size // PATCH_SIZE
    num_patch_tokens = tokens_per_side * tokens_per_side
    num_tokens = num_patch_tokens + 1 if args.feature_type == "all_tokens" else 1
    latent_dim = int(cfg["embed_dim"]) * (2 if args.feature_type == "cls_and_avg_patch" else 1)
    if args.feature_type == "all_tokens":
        chunk_shape = (expected_frames, num_tokens, int(cfg["embed_dim"]))
    else:
        chunk_shape = (expected_frames, latent_dim)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else infer_output_dir(dataset_path, args.model, args.feature_type)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:      {dataset_path}")
    print(f"Video dir:    {video_dir}")
    print(f"Model:        {args.model} ({cfg['hub_fn']})")
    print(f"Feature type: {args.feature_type}")
    print(f"Input size:   {input_size}px")
    print(f"Chunk size:   {args.chunk_size} trajectories")
    print(f"Chunk shape:  {chunk_shape}")
    print(f"Device:       {device}")
    print(f"Output dir:   {output_dir}")

    chunk_latents: list[torch.Tensor] = []
    chunk_start_idx = 0
    chunk_paths: list[str] = []

    for traj_idx, video_path in enumerate(tqdm(video_paths, desc="Encoding trajectories", unit="traj")):
        frames = iter_video_frames(video_path)
        if len(frames) != expected_frames:
            raise RuntimeError(
                f"{video_path.name} has {len(frames)} frames, expected {expected_frames}."
            )
        chunk_latents.append(
            encode_video(
            model,
            frames,
            input_size=input_size,
            batch_size=args.batch_size,
            feature_type=args.feature_type,
            device=device,
        )
        )

        if len(chunk_latents) == args.chunk_size or traj_idx == num_trajectories - 1:
            chunk_end_idx = traj_idx + 1
            chunk_tensor = torch.stack(chunk_latents, dim=0)
            chunk_path = output_dir / f"latents_{chunk_start_idx:07d}_{chunk_end_idx - 1:07d}.pt"
            torch.save(
                {
                    "latents": chunk_tensor,
                    "trajectory_start_idx": chunk_start_idx,
                    "trajectory_end_idx": chunk_end_idx,
                },
                chunk_path,
            )
            chunk_paths.append(str(chunk_path))
            chunk_latents.clear()
            chunk_start_idx = chunk_end_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata_path = output_dir / "metadata.pt"
    metadata = {
        "latent_type": args.feature_type,
        "latent_dim": latent_dim,
        "num_tokens": num_tokens,
        "num_trajectories": num_trajectories,
        "frames_per_trajectory": expected_frames,
        "model_name": args.model,
        "hub_fn": cfg["hub_fn"],
        "weights_path": str(cfg["weights"]),
        "input_size": input_size,
        "patch_size": PATCH_SIZE,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "chunk_paths": chunk_paths,
        "dataset_path": str(dataset_path),
        "video_dir": str(video_dir),
        "source_states_shape": tuple(dataset["states"].shape),
        "source_actions_shape": tuple(dataset["actions"].shape),
    }
    torch.save(metadata, metadata_path)
    print(f"Saved latent metadata to {metadata_path}")


if __name__ == "__main__":
    main()
