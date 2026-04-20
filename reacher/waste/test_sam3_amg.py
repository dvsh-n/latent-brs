#!/usr/bin/env python3
"""Run SAM3 on a standalone video with an automatically generated visual box prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from sam3 import build_sam3_predictor


VIDEO_PATH = Path("test/9501826-uhd_2160_4096_24fps.mp4")
CHECKPOINT_PATH = Path("models/sam3/sam3.pt")
OUTPUT_PATH = Path("test/sam3_amg_output.pt")
OVERLAY_VIDEO_PATH = Path("test/sam3_amg_overlay.mp4")
SAM3_VERSION = "sam3"
OVERLAY_ALPHA = 0.45
OVERLAY_COLOR = np.array([128, 0, 128], dtype=np.float32)

AUTO_SEARCH_MAX_FRAMES = 24
MIN_COMPONENT_AREA_RATIO = 0.0005
MAX_COMPONENT_AREA_RATIO = 0.35
MOTION_PERCENTILE = 97.5
MOTION_MIN_THRESHOLD = 20.0
MORPH_KERNEL_SIZE = 7


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


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def choose_seed_frame_index(video_frames: list[np.ndarray]) -> int:
    if len(video_frames) < 3:
        return 0

    max_index = min(len(video_frames) - 2, AUTO_SEARCH_MAX_FRAMES - 1)
    if max_index <= 1:
        return 0

    scores: list[tuple[float, int]] = []
    for frame_index in range(1, max_index + 1):
        prev_gray = to_gray(video_frames[frame_index - 1]).astype(np.float32)
        cur_gray = to_gray(video_frames[frame_index]).astype(np.float32)
        next_gray = to_gray(video_frames[frame_index + 1]).astype(np.float32)
        score = float(
            np.mean(np.abs(cur_gray - prev_gray)) + np.mean(np.abs(next_gray - cur_gray))
        )
        scores.append((score, frame_index))

    if not scores:
        return 0
    return max(scores)[1]


def build_motion_mask(video_frames: list[np.ndarray], frame_index: int) -> np.ndarray:
    frame_count = len(video_frames)
    prev_index = max(frame_index - 1, 0)
    next_index = min(frame_index + 1, frame_count - 1)

    prev_gray = to_gray(video_frames[prev_index]).astype(np.float32)
    cur_gray = to_gray(video_frames[frame_index]).astype(np.float32)
    next_gray = to_gray(video_frames[next_index]).astype(np.float32)

    diff_prev = np.abs(cur_gray - prev_gray)
    diff_next = np.abs(next_gray - cur_gray)
    motion = np.maximum(diff_prev, diff_next)
    motion = cv2.GaussianBlur(motion, (5, 5), 0)

    positive_motion = motion[motion > 0]
    if positive_motion.size == 0:
        return np.zeros_like(motion, dtype=np.uint8)

    threshold = max(
        MOTION_MIN_THRESHOLD,
        float(np.percentile(positive_motion, MOTION_PERCENTILE)),
    )
    mask = (motion >= threshold).astype(np.uint8) * 255

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def generate_auto_seed_box(video_frames: list[np.ndarray]) -> tuple[int, dict[str, Any]]:
    seed_frame_index = choose_seed_frame_index(video_frames)
    motion_mask = build_motion_mask(video_frames, seed_frame_index)
    height, width = motion_mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask, 8)
    seeds: list[dict[str, Any]] = []
    min_area = int(height * width * MIN_COMPONENT_AREA_RATIO)
    max_area = int(height * width * MAX_COMPONENT_AREA_RATIO)

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        cx, cy = centroids[label]
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue

        # Prefer compact, high-motion regions over broad noisy blobs.
        score = float(area / max(w * h, 1))

        seeds.append(
            {
                "point": [float(cx / width), float(cy / height)],
                "box_xywh": [
                    float(x / width),
                    float(y / height),
                    float(w / width),
                    float(h / height),
                ],
                "box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "area": area,
                "score": score,
            }
        )

    seeds.sort(key=lambda seed: (seed["score"], seed["area"]), reverse=True)
    if seeds:
        return seed_frame_index, seeds[0]

    return seed_frame_index, {
        "point": [0.5, 0.5],
        "box_xywh": [0.25, 0.25, 0.5, 0.5],
        "box_xyxy": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
        "area": 0,
        "score": 0.0,
    }


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

    seed_frame_index, auto_seed = generate_auto_seed_box(video_frames)

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
                "frame_index": seed_frame_index,
                "bounding_boxes": [auto_seed["box_xywh"]],
                "bounding_box_labels": [1],
            }
        )
        masks_by_frame = collect_masks(predictor, session_id)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    overlay_frames = render_overlay_frames(video_frames, masks_by_frame)
    write_video(overlay_video_path, overlay_frames, fps=fps)
    nonempty_frames, total_mask_pixels = summarize_masks(masks_by_frame)

    output = {
        "prompt_type": "auto_motion_box",
        "video_path": str(video_path),
        "overlay_video_path": str(overlay_video_path),
        "checkpoint_path": str(checkpoint_path),
        "sam3_version": SAM3_VERSION,
        "auto_seed_frame_index": seed_frame_index,
        "auto_seed": auto_seed,
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
            "prompt_type": "auto_motion_box",
            "auto_seed_frame_index": seed_frame_index,
            "auto_seed_point": auto_seed["point"],
            "auto_seed_box_xywh": auto_seed["box_xywh"],
            "num_frames_with_outputs": len(masks_by_frame),
            "nonempty_frames": nonempty_frames,
            "total_mask_pixels": total_mask_pixels,
            "video_path": str(video_path),
        }
    )

    if nonempty_frames == 0:
        print(
            "WARNING: Automatic box seeding produced no positive masks. "
            "Try loosening the motion thresholds or increasing AUTO_SEARCH_MAX_FRAMES."
        )


if __name__ == "__main__":
    main()
