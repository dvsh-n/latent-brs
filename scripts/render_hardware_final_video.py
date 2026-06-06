#!/usr/bin/env python3
"""Render a four-column hardware comparison video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = ROOT / "hardware" / "素材"
DEFAULT_FONT = (
    ROOT
    / "third_party"
    / "stable-pretraining"
    / "assets"
    / "cm-unicode-0.7.0 2"
    / "cmunrm.ttf"
)
DEFAULT_OUTPUT = ROOT / "hardware" / "final_video" / "final_paper_ready_video.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2460)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--crop-left", type=int, default=100)
    parser.add_argument("--stack-views", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def fit_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    return ImageOps.contain(image, box, method=Image.Resampling.LANCZOS)


def centered_paste(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    x = left + (right - left - image.width) // 2
    y = top + (bottom - top - image.height) // 2
    canvas.paste(image, (x, y))


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.last_frame: Image.Image | None = None

    def frame_at(self, seconds: float) -> Image.Image:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = Image.fromarray(frame)
        if self.last_frame is None:
            raise RuntimeError(f"Could not read a frame from: {self.path}")
        return self.last_frame.copy()

    def release(self) -> None:
        self.cap.release()


def render(args: argparse.Namespace) -> Path:
    asset_dir = args.asset_dir
    start_path = asset_dir / "start_image.png"
    goal_path = asset_dir / "goal_image.png"
    video_paths = [
        asset_dir / "stop_motion.mp4",
        asset_dir / "素材_1.mp4",
        asset_dir / "素材_2.mp4",
    ]
    required = [start_path, goal_path, *video_paths]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))

    duration = args.duration or max(video_duration(path) for path in video_paths)
    frame_count = int(round(duration * args.fps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(".tmp.mp4")

    font_path = args.font if args.font.exists() else None
    if font_path is None:
        print(f"Warning: CMU Serif font not found at {args.font}; using PIL default.")
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    else:
        title_font = ImageFont.truetype(str(font_path), 46)
        label_font = ImageFont.truetype(str(font_path), 36)

    canvas_width = args.width + args.crop_left

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp_output}")

    bg = (250, 250, 248)
    ink = (28, 29, 31)
    margin = 18
    gutter = 28
    title_h = 50
    bottom_pad = 18
    content_top = margin
    content_bottom = args.height - bottom_pad

    start_img = Image.open(start_path)
    goal_img = Image.open(goal_path)
    readers = [VideoReader(path) for path in video_paths]
    titles = ["", "Stop Motion", "Front View", "Side View"]

    try:
        for frame_idx in range(frame_count):
            t = frame_idx / args.fps
            canvas = Image.new("RGB", (canvas_width, args.height), bg)
            draw = ImageDraw.Draw(canvas)

            label_h = 44
            gap = 18
            if args.stack_views:
                group_gap = 24
                start_w = 330
                stop_w = 500
                view_w = canvas_width - 2 * margin - start_w - stop_w - 2 * group_gap
                first_left = margin
                first_right = first_left + start_w
                video_lefts = [first_right + group_gap, first_right + group_gap + stop_w + group_gap]
                video_widths = [stop_w, view_w]
            else:
                col_w = (canvas_width - 2 * margin - 3 * gutter) // 4
                col_lefts = [margin + i * (col_w + gutter) for i in range(4)]
                start_goal_shift = 100
                first_left = col_lefts[0] + start_goal_shift
                first_right = col_lefts[0] + col_w + start_goal_shift
                video_lefts = col_lefts[1:]
                video_widths = [col_w, col_w, col_w]

            pair_h = (content_bottom - content_top - gap) // 2
            img_box_h = pair_h - label_h
            start_group_w = first_right - first_left
            img_size = min(int(start_group_w * 0.78), img_box_h)
            for label, image, top in [
                ("Start", start_img, content_top),
                ("Goal", goal_img, content_top + pair_h + gap),
            ]:
                draw_centered_text(
                    draw,
                    label,
                    (first_left, top, first_right, top + label_h),
                    label_font,
                    ink,
                )
                fitted = fit_image(image.copy(), (img_size, img_size))
                centered_paste(
                    canvas,
                    fitted,
                    (
                        first_left,
                        top + label_h,
                        first_right,
                        top + label_h + img_box_h,
                    ),
                )

            if args.stack_views:
                left = video_lefts[0]
                right = left + video_widths[0]
                frame = readers[0].frame_at(t)
                max_w = int(video_widths[0] * 0.9)
                max_h = content_bottom - content_top - title_h
                fitted = fit_image(frame, (max_w, max_h))
                centered_paste(canvas, fitted, (left, content_top, right, content_bottom - title_h))
                draw_centered_text(
                    draw,
                    titles[1],
                    (left, content_bottom - title_h, right, content_bottom),
                    title_font,
                    ink,
                )

                left = video_lefts[1]
                right = left + video_widths[1]
                stack_gap = 14
                caption_h = 44
                slot_h = (content_bottom - content_top - stack_gap) // 2
                for title, reader, top in [
                    (titles[2], readers[1], content_top),
                    (titles[3], readers[2], content_top + slot_h + stack_gap),
                ]:
                    frame = reader.frame_at(t)
                    fitted = fit_image(frame, (video_widths[1], slot_h - caption_h))
                    centered_paste(canvas, fitted, (left, top, right, top + slot_h - caption_h))
                    draw_centered_text(
                        draw,
                        title,
                        (left, top + slot_h - caption_h, right, top + slot_h),
                        label_font,
                        ink,
                    )
            else:
                for i, reader in enumerate(readers, start=1):
                    left = video_lefts[i - 1]
                    width = video_widths[i - 1]
                    right = left + width
                    frame = reader.frame_at(t)
                    max_w = int(width * 0.88) if i == 1 else width
                    max_h = content_bottom - content_top - title_h
                    fitted = fit_image(frame, (max_w, max_h))
                    centered_paste(canvas, fitted, (left, content_top, right, content_bottom - title_h))
                    draw_centered_text(
                        draw,
                        titles[i],
                        (left, content_bottom - title_h, right, content_bottom),
                        title_font,
                        ink,
                    )

            canvas = canvas.crop((args.crop_left, 0, args.crop_left + args.width, args.height))
            frame_bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
        for reader in readers:
            reader.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_output),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    if not args.keep_temp:
        temp_output.unlink(missing_ok=True)
    return args.output


def main() -> None:
    args = parse_args()
    output = render(args)
    print(f"Rendered {output}")


if __name__ == "__main__":
    main()
