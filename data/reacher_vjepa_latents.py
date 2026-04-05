#!/usr/bin/env python3
"""Encode Reacher trajectory videos into frame-wise V-JEPA latent grids."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.models import load_vjepa2_vitb_encoder_384


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        "--checkpoint",
        type=Path,
        default=Path("models/vjepa2_1_vitb_dist_vitG_384.pt"),
        help="Path to the V-JEPA 2.1 ViT-B checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where chunk files will be written. Defaults to data/reacher_vjepa_latents_<H>step.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Manifest file describing the latent dataset. Defaults to data/reacher_vjepa_latents_<H>step_manifest.pt.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="Number of trajectories per output chunk.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=256,
        help="Number of frames per GPU forward pass.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize frames to resize x resize before encoding.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=1,
        help="Number of frames in the history window encoded for each timestep. Uses left-padding with the first frame.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model inference.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Autocast dtype for inference on CUDA.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
        help="Dtype used when saving latent tensors.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the encoder with torch.compile.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = f"{int(args.history_frames)}step"
    output_dir = args.output_dir if args.output_dir is not None else Path(f"data/reacher_vjepa_latents_{suffix}")
    output_manifest = (
        args.output_manifest
        if args.output_manifest is not None
        else Path(f"data/reacher_vjepa_latents_{suffix}_manifest.pt")
    )
    return output_dir, output_manifest


def default_video_path(dataset_path: Path, trajectory_index: int) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_videos") / f"trajectory_{trajectory_index:07d}.mp4"


def resolve_video_path(dataset_path: Path, record: dict[str, Any]) -> Path:
    video_path_value = record.get("video_path")
    if video_path_value:
        candidate = (dataset_path.parent / Path(str(video_path_value))).resolve()
        if candidate.exists():
            return candidate
    fallback = default_video_path(dataset_path, int(record["trajectory_index"])).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not resolve video for trajectory {int(record['trajectory_index'])} from {dataset_path}"
    )


def read_video_frames(video_path: Path) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    try:
        frames = [torch.from_numpy(frame).permute(2, 0, 1).contiguous() for frame in reader]
    finally:
        reader.close()
    return torch.stack(frames, dim=0)


def get_autocast_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported amp dtype: {name}")


def get_save_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {name}")


def preprocess_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    frames = frames.to(torch.float32).div_(255.0)
    if frames.shape[-2:] != (image_size, image_size):
        frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std


def build_history_batch(
    processed_frames: torch.Tensor,
    target_indices: torch.Tensor,
    history_frames: int,
) -> torch.Tensor:
    history_offsets = torch.arange(history_frames, device=processed_frames.device, dtype=torch.long)
    history_offsets = history_offsets.flip(0)
    gather_indices = target_indices.unsqueeze(1) - history_offsets.unsqueeze(0)
    gather_indices = gather_indices.clamp_min_(0)
    clips = processed_frames.index_select(0, gather_indices.reshape(-1))
    clips = clips.view(target_indices.shape[0], history_frames, *processed_frames.shape[1:])
    return clips.permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)


def encode_frame_batch(
    model: torch.nn.Module,
    clips: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    clips = clips.to(device=device, non_blocking=True, memory_format=torch.channels_last_3d)
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        tokens = model(clips)
    return tokens.float().cpu()


def encode_trajectory_frames(
    model: torch.nn.Module,
    frames: torch.Tensor,
    *,
    frame_batch_size: int,
    history_frames: int,
    image_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> torch.Tensor:
    processed = preprocess_frames(frames, image_size=image_size).contiguous()
    latent_batches: list[torch.Tensor] = []
    temporal_tokens = 1 if history_frames == 1 else history_frames // 2
    timestep_indices = torch.arange(processed.shape[0], dtype=torch.long)

    for start in range(0, processed.shape[0], frame_batch_size):
        batch_indices = timestep_indices[start : start + frame_batch_size]
        clips = build_history_batch(processed, batch_indices, history_frames=history_frames)
        tokens = encode_frame_batch(model, clips, device=device, amp_dtype=amp_dtype)
        latent_batches.append(tokens)

    latents = torch.cat(latent_batches, dim=0)
    h_patches = image_size // 16
    w_patches = image_size // 16
    latents = latents.view(latents.shape[0], temporal_tokens, h_patches, w_patches, latents.shape[-1])
    latents = latents.permute(0, 4, 1, 2, 3).contiguous()
    return latents.to(save_dtype)


def flush_chunk(
    output_dir: Path,
    chunk_index: int,
    chunk_records: list[dict[str, Any]],
    chunk_latents: list[torch.Tensor],
    save_dtype: torch.dtype,
) -> dict[str, Any]:
    chunk_path = output_dir / f"chunk_{chunk_index:05d}.pt"
    latents_tensor = torch.stack(chunk_latents, dim=0).contiguous().to(save_dtype)
    actions_tensor = torch.stack([record["actions"].to(torch.float32) for record in chunk_records], dim=0).contiguous()
    states_tensor = torch.stack([record["states"].to(torch.float32) for record in chunk_records], dim=0).contiguous()
    chunk_data = {
        "trajectory_indices": torch.tensor(
            [int(record["trajectory_index"]) for record in chunk_records],
            dtype=torch.int64,
        ),
        "latents": latents_tensor,
        "actions": actions_tensor,
        "states": states_tensor,
    }
    estimated_bytes = (
        latents_tensor.numel() * latents_tensor.element_size()
        + actions_tensor.numel() * actions_tensor.element_size()
        + states_tensor.numel() * states_tensor.element_size()
        + chunk_data["trajectory_indices"].numel() * chunk_data["trajectory_indices"].element_size()
    )
    free_bytes = shutil.disk_usage(output_dir).free
    safety_margin_bytes = 512 * 1024 * 1024
    if free_bytes < estimated_bytes + safety_margin_bytes:
        estimated_gb = estimated_bytes / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        raise RuntimeError(
            "Insufficient disk space to save the next chunk. "
            f"Need about {estimated_gb:.2f} GiB plus margin, but only {free_gb:.2f} GiB is free in {output_dir}."
        )

    tmp_chunk_path = chunk_path.with_suffix(".pt.tmp")
    try:
        torch.save(chunk_data, tmp_chunk_path)
        os.replace(tmp_chunk_path, chunk_path)
    except Exception:
        if tmp_chunk_path.exists():
            tmp_chunk_path.unlink()
        raise
    return {
        "chunk_path": str(chunk_path.resolve()),
        "num_trajectories": len(chunk_records),
        "trajectory_indices": [int(record["trajectory_index"]) for record in chunk_records],
        "num_frames": int(latents_tensor.shape[1]),
        "latent_channels": int(latents_tensor.shape[2]),
        "latent_grid": [int(latents_tensor.shape[3]), int(latents_tensor.shape[4]), int(latents_tensor.shape[5])],
        "save_dtype": str(latents_tensor.dtype),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    if args.resize <= 0 or args.resize % 16 != 0:
        raise ValueError("resize must be positive and divisible by 16")
    if args.history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if args.history_frames > 1 and args.history_frames % 2 != 0:
        raise ValueError("history_frames must be 1 or an even integer because V-JEPA uses tubelets of size 2")

    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir_arg, output_manifest_arg = resolve_output_paths(args)
    output_dir = output_dir_arg.resolve()
    output_manifest = output_manifest_arg.resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty; pass --overwrite to replace it")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{output_manifest} already exists; pass --overwrite to replace it")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    amp_dtype = get_autocast_dtype(args.amp_dtype)
    save_dtype = get_save_dtype(args.save_dtype)

    model = load_vjepa2_vitb_encoder_384(checkpoint_path, device=device)
    model = model.eval()
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    header, meta, records = load_streamed_dataset(input_path)

    chunk_records: list[dict[str, Any]] = []
    chunk_latents: list[torch.Tensor] = []
    chunk_entries: list[dict[str, Any]] = []
    chunk_index = 0

    with torch.inference_mode():
        for record in tqdm(records, desc="Encode trajectories", dynamic_ncols=True):
            video_path = resolve_video_path(input_path, record)
            frames = read_video_frames(video_path)
            latents = encode_trajectory_frames(
                model,
                frames,
                frame_batch_size=args.frame_batch_size,
                history_frames=args.history_frames,
                image_size=args.resize,
                device=device,
                amp_dtype=amp_dtype,
                save_dtype=save_dtype,
            )

            chunk_records.append(record)
            chunk_latents.append(latents)

            if len(chunk_records) == args.chunk_size:
                chunk_entries.append(
                    flush_chunk(
                        output_dir=output_dir,
                        chunk_index=chunk_index,
                        chunk_records=chunk_records,
                        chunk_latents=chunk_latents,
                        save_dtype=save_dtype,
                    )
                )
                chunk_records = []
                chunk_latents = []
                chunk_index += 1

    if chunk_records:
        chunk_entries.append(
            flush_chunk(
                output_dir=output_dir,
                chunk_index=chunk_index,
                chunk_records=chunk_records,
                chunk_latents=chunk_latents,
                save_dtype=save_dtype,
            )
        )

    manifest = {
        "format": "reacher_vjepa_frame_latents_v1",
        "source_dataset": str(input_path),
        "source_header": header,
        "source_meta": meta,
        "checkpoint": str(checkpoint_path),
        "encoder_name": "vjepa2_1_vitb_dist_vitG_384",
        "encoding_mode": "history_window_video",
        "history_frames": args.history_frames,
        "frame_resolution": [args.resize, args.resize],
        "latent_layout": ["trajectory", "frame", "channel", "time", "height", "width"],
        "latent_channels": 768,
        "latent_grid": [1 if args.history_frames == 1 else args.history_frames // 2, args.resize // 16, args.resize // 16],
        "num_chunks": len(chunk_entries),
        "num_trajectories": len(records),
        "chunk_size": args.chunk_size,
        "frame_batch_size": args.frame_batch_size,
        "amp_dtype": args.amp_dtype,
        "save_dtype": args.save_dtype,
        "chunks": chunk_entries,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, output_manifest)

    print(
        {
            "input": str(input_path),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "output_manifest": str(output_manifest),
            "num_trajectories": len(records),
            "num_chunks": len(chunk_entries),
            "frame_resolution": [args.resize, args.resize],
            "history_frames": args.history_frames,
            "latent_shape_per_timestep": [768, 1 if args.history_frames == 1 else args.history_frames // 2, args.resize // 16, args.resize // 16],
        }
    )


if __name__ == "__main__":
    main()
