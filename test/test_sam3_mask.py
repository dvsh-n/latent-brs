#!/usr/bin/env python3
"""Run SAM3 on a standalone video with a text prompt and save masks plus overlay video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from sam3 import build_sam3_predictor


VIDEO_PATH = Path("test/reacher_texture_test.mp4")
PROMPT = "green ball"
CHECKPOINT_PATH = Path("models/sam3/sam3.pt")
OUTPUT_PATH = Path("test/sam3_mask_output.pt")
OVERLAY_VIDEO_PATH = Path("test/sam3_mask_overlay.mp4")
SAM3_VERSION = "sam3"
PROMPT_FRAME_INDEX = 0
OVERLAY_ALPHA = 0.45
OVERLAY_COLOR = np.array([128, 0, 128], dtype=np.float32)


def collect_masks(predictor: Any, session_id: str) -> dict[int, dict[int, torch.Tensor]]:
    masks_by_frame: dict[int, dict[int, torch.Tensor]] = {}
    for response in predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward"}
    ):
        frame_index = int(response["frame_index"])
        outputs = response.get("outputs", {})
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")

        frame_masks: dict[int, torch.Tensor] = {}
        if binary_masks is not None:
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.cpu().tolist()
            if not isinstance(binary_masks, torch.Tensor):
                binary_masks = torch.as_tensor(binary_masks)
            binary_masks = binary_masks.detach().cpu().to(torch.bool)

            for i, obj_id in enumerate(obj_ids):
                mask = binary_masks[i]
                if mask.ndim == 3:
                    mask = mask[0]
                frame_masks[int(obj_id)] = mask

        masks_by_frame[frame_index] = frame_masks
    return masks_by_frame


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(video_path)
    try:
        return [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()


def render_overlay_frames(
    video_frames: list[np.ndarray],
    masks_by_frame: dict[int, dict[int, torch.Tensor]],
) -> list[np.ndarray]:
    overlay_frames: list[np.ndarray] = []
    for frame_index, frame in enumerate(video_frames):
        composed = frame.astype(np.float32).copy()
        for mask in masks_by_frame.get(frame_index, {}).values():
            mask_np = mask.detach().cpu().numpy().astype(bool)
            composed[mask_np] = (
                (1.0 - OVERLAY_ALPHA) * composed[mask_np] + OVERLAY_ALPHA * OVERLAY_COLOR
            )
        overlay_frames.append(np.clip(composed, 0, 255).astype(np.uint8))
    return overlay_frames


def write_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=fps)


def summarize_masks(masks_by_frame: dict[int, dict[int, torch.Tensor]]) -> tuple[int, int]:
    nonempty_frames = 0
    total_mask_pixels = 0
    for frame_masks in masks_by_frame.values():
        frame_pixels = 0
        for mask in frame_masks.values():
            frame_pixels += int(mask.sum().item())
        if frame_pixels > 0:
            nonempty_frames += 1
            total_mask_pixels += frame_pixels
    return nonempty_frames, total_mask_pixels


def main() -> None:
    video_path = VIDEO_PATH.resolve()
    checkpoint_path = CHECKPOINT_PATH.resolve()
    output_path = OUTPUT_PATH.resolve()
    overlay_video_path = OVERLAY_VIDEO_PATH.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    video_frames = read_video_frames(video_path)
    if not video_frames:
        raise RuntimeError(f"No frames found in {video_path}")

    fps = 24
    try:
        meta = imageio.immeta(video_path)
        if isinstance(meta, dict) and meta.get("fps"):
            fps = int(round(float(meta["fps"])))
    except Exception:
        pass

    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version=SAM3_VERSION,
        compile=False,
        async_loading_frames=False,
    )

    session = predictor.handle_request({"type": "start_session", "resource_path": str(video_path)})
    session_id = session["session_id"]
    try:
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": PROMPT_FRAME_INDEX,
                "text": PROMPT,
            }
        )
        masks_by_frame = collect_masks(predictor, session_id)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    overlay_frames = render_overlay_frames(video_frames, masks_by_frame)
    write_video(overlay_video_path, overlay_frames, fps=fps)
    nonempty_frames, total_mask_pixels = summarize_masks(masks_by_frame)

    output = {
        "prompt_type": "text",
        "prompt": PROMPT,
        "video_path": str(video_path),
        "overlay_video_path": str(overlay_video_path),
        "checkpoint_path": str(checkpoint_path),
        "sam3_version": SAM3_VERSION,
        "prompt_frame_index": PROMPT_FRAME_INDEX,
        "video_fps": fps,
        "num_video_frames": len(video_frames),
        "masks_by_frame": masks_by_frame,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(
        {
            "output_path": str(output_path),
            "overlay_video_path": str(overlay_video_path),
            "prompt_type": "text",
            "prompt": PROMPT,
            "prompt_frame_index": PROMPT_FRAME_INDEX,
            "num_frames_with_outputs": len(masks_by_frame),
            "nonempty_frames": nonempty_frames,
            "total_mask_pixels": total_mask_pixels,
            "video_path": str(video_path),
        }
    )

    if nonempty_frames == 0:
        print(
            "WARNING: SAM3 produced no positive masks for the text prompt. "
            "Try a different prompt or move the prompt frame to one where the object is clearly visible."
        )


if __name__ == "__main__":
    main()
