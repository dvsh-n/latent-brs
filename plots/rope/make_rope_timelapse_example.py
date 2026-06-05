#!/usr/bin/env python3
"""Example: build a 2x5 rollout timelapse with the goal image in the final cell."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


VIDEO_PATH = Path("rope/plan/ilqr_mpc_mlpdyn/1779499040_episode_00121/rollout.mp4")
GOAL_IMAGE_PATH = Path("rope/plan/ilqr_mpc_mlpdyn/1779499040_episode_00121/goal_image.png")
OUT_PATH = Path("plots/rope/episode_00121_ilqr_timelapse_2x5_example.pdf")

ROWS = 2
COLS = 5
NUM_VIDEO_PANELS = ROWS * COLS - 1
END_TIME_S = 1.5


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Image.Image:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError("Video metadata is incomplete.")

    end_frame = min(frame_count - 1, round(END_TIME_S * fps))
    frame_indices = [round(i * end_frame / (NUM_VIDEO_PANELS - 1)) for i in range(NUM_VIDEO_PANELS)]
    frames = [read_frame(cap, frame_idx) for frame_idx in frame_indices]
    cap.release()

    goal = Image.open(GOAL_IMAGE_PATH).convert("RGB")
    width, height = frames[0].size
    if goal.size != (width, height):
        goal = goal.resize((width, height), Image.Resampling.LANCZOS)

    label_height = max(22, height // 12)
    font_size = max(12, label_height // 2)
    font = load_font("DejaVuSans.ttf", font_size)
    goal_font = load_font("DejaVuSans-Bold.ttf", font_size)

    grid = Image.new("RGB", (COLS * width, ROWS * (height + label_height)), "white")
    draw = ImageDraw.Draw(grid)

    panels = frames + [goal]
    labels = [f"t={frame_idx / fps:.2f}s" for frame_idx in frame_indices] + ["Goal"]

    for panel_idx, (image, label) in enumerate(zip(panels, labels, strict=True)):
        row, col = divmod(panel_idx, COLS)
        x = col * width
        y = row * (height + label_height)

        draw.rectangle([x, y, x + width, y + label_height], fill=(245, 245, 245))
        if label == "Goal":
            draw.text((x + 6, y + 4), label, fill=(210, 20, 20), font=goal_font)
        else:
            draw.text((x + 6, y + 4), label, fill=(20, 20, 20), font=font)
        grid.paste(image, (x, y + label_height))

    grid.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")
    print(f"Sampled frames: {frame_indices}")


if __name__ == "__main__":
    main()
