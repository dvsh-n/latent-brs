#!/usr/bin/env python3
"""Compose side-by-side planning rollout comparison videos.

Each method should point to either a run directory containing case directories
or directly to a single case directory. Case directories are expected to contain
``rollout.mp4``, ``start_image.png``, ``goal_image.png``, and optionally
``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_DIR = Path("third_party/stable-pretraining/assets/cm-unicode-0.7.0 2")


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must look like LABEL=/path/to/run_or_case")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Method label cannot be empty.")
    return label, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--case", help="Case directory name, e.g. case_0001_episode_00033.")
    parser.add_argument("--case-idx", type=int, help="Resolve case_XXXX_* inside each run directory.")
    parser.add_argument("--episode-idx", type=int, help="Resolve case_*_episode_YYYYY inside each run directory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Planning comparison")
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--fps", type=int, default=None, help="Default: read from first rollout metadata, then fallback to 60.")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional maximum output duration in seconds.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--reference-title", default="Reference")
    parser.add_argument("--note", default="Shorter rollouts hold on their final frame.")
    parser.add_argument("--hide-timestamp", action="store_true")
    parser.add_argument("--hide-captions", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=60)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    parser.add_argument("--regular-font", type=Path, default=None)
    parser.add_argument("--bold-font", type=Path, default=None)
    parser.add_argument("--quality", type=int, default=9)
    return parser.parse_args()


def resolve_case_dir(base: Path, args: argparse.Namespace) -> Path:
    base = base.resolve()
    if (base / "rollout.mp4").is_file():
        return base
    if args.case:
        case_dir = base / args.case
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing case directory: {case_dir}")
        return case_dir
    if args.case_idx is not None:
        matches = sorted(base.glob(f"case_{int(args.case_idx):04d}_episode_*"))
    elif args.episode_idx is not None:
        matches = sorted(base.glob(f"case_*_episode_{int(args.episode_idx):05d}"))
    else:
        raise ValueError("Pass --case, --case-idx, --episode-idx, or method paths that are already case dirs.")
    if not matches:
        raise FileNotFoundError(f"No matching case directory under {base}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous case under {base}: {matches}")
    return matches[0]


def load_summary(case_dir: Path) -> dict:
    summary_path = case_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def count_frames(reader: imageio.Reader) -> int:
    try:
        nframes = reader.count_frames()
        if nframes and math.isfinite(float(nframes)):
            return int(nframes)
    except Exception:
        pass
    count = 0
    for _ in reader:
        count += 1
    return count


def reader_fps(reader: imageio.Reader, fallback: int = 60) -> int:
    try:
        fps = reader.get_meta_data().get("fps")
        if fps:
            return int(round(float(fps)))
    except Exception:
        pass
    return int(fallback)


def choose_fonts(args: argparse.Namespace) -> tuple[Path, Path]:
    regular = args.regular_font or (args.font_dir / "cmunrm.ttf")
    bold = args.bold_font or (args.font_dir / "cmunbx.ttf")
    if regular.is_file() and bold.is_file():
        return regular, bold
    dejavu = Path("/usr/share/fonts/truetype/dejavu")
    return dejavu / "DejaVuSerif.ttf", dejavu / "DejaVuSerif-Bold.ttf"


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=font, fill=fill)


def metric_text(summary: dict) -> str:
    success = summary.get("qpos_success", summary.get("success"))
    status = "reached" if bool(success) else "not reached" if success is not None else ""
    metric_keys = [
        "final_qpos_distance",
        "final_cube_goal_distance",
        "final_block_position_distance",
        "final_block_goal_distance",
        "final_observation_distance",
    ]
    labels = {
        "final_qpos_distance": "final qpos",
        "final_cube_goal_distance": "final cube",
        "final_block_position_distance": "final block",
        "final_block_goal_distance": "final block",
        "final_observation_distance": "final obs",
    }
    parts: list[str] = []
    for key in metric_keys:
        if key in summary:
            parts.append(f"{labels[key]} {float(summary[key]):.3f}")
            break
    if status:
        parts.append(status)
    return " | ".join(parts)


def make_subtitle(case_dirs: list[Path], user_subtitle: str | None) -> str:
    if user_subtitle is not None:
        return user_subtitle
    summary = load_summary(case_dirs[0])
    chunks = []
    if "episode_idx" in summary:
        chunks.append(f"episode {int(summary['episode_idx'])}")
    if "start_step" in summary and "goal_step" in summary:
        chunks.append(f"start step {int(summary['start_step'])} to goal step {int(summary['goal_step'])}")
    return " | ".join(chunks)


def main() -> None:
    args = parse_args()
    if len(args.method) > 4:
        raise ValueError("This layout supports up to 4 methods.")

    method_dirs = [(label, resolve_case_dir(path, args)) for label, path in args.method]
    missing = [case_dir for _, case_dir in method_dirs if not (case_dir / "rollout.mp4").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rollout.mp4 in: {missing}")

    regular_font_path, bold_font_path = choose_fonts(args)
    font_small = ImageFont.truetype(str(regular_font_path), 20)
    font_tiny = ImageFont.truetype(str(regular_font_path), 17)
    font_note = ImageFont.truetype(str(regular_font_path), 18)
    font_bold = ImageFont.truetype(str(bold_font_path), 30)
    font_title = ImageFont.truetype(str(bold_font_path), 40)
    font_panel = ImageFont.truetype(str(bold_font_path), 30)

    readers = [(label, imageio.get_reader(str(case_dir / "rollout.mp4"))) for label, case_dir in method_dirs]
    try:
        fps = int(args.fps or reader_fps(readers[0][1]))
        lengths = [count_frames(reader) for _, reader in readers]
    finally:
        for _, reader in readers:
            reader.close()

    readers = [(label, imageio.get_reader(str(case_dir / "rollout.mp4"))) for label, case_dir in method_dirs]
    n_frames = max(lengths)
    if args.max_duration is not None:
        if float(args.max_duration) <= 0:
            raise ValueError("--max-duration must be positive.")
        n_frames = min(n_frames, max(1, int(round(float(args.max_duration) * fps))))
    summaries = {label: load_summary(case_dir) for label, case_dir in method_dirs}

    reference_case = method_dirs[0][1]
    start_img = Image.open(reference_case / "start_image.png").convert("RGB")
    goal_img = Image.open(reference_case / "goal_image.png").convert("RGB")

    canvas_w = int(args.width)
    canvas_h = int(args.height)
    bg = (246, 247, 249)
    ink = (25, 28, 35)
    muted = (82, 90, 105)
    line = (160, 166, 176)
    accent_start = (30, 118, 210)
    accent_goal = (218, 74, 56)

    left_x = 44
    grid_x = 392
    top_y = 110
    gap = 28
    label_h = 46
    panel_size = min(420, (canvas_w - grid_x - gap - 34) // 2)
    ref_size = min(300, grid_x - left_x - 58)
    start_y = 198
    goal_y = start_y + ref_size + 90
    subtitle = make_subtitle([case_dir for _, case_dir in method_dirs], args.subtitle)

    start_img = start_img.resize((ref_size, ref_size), Image.Resampling.LANCZOS)
    goal_img = goal_img.resize((ref_size, ref_size), Image.Resampling.LANCZOS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preview_path = args.out.with_name(args.out.stem + f"_frame{int(args.preview_frame):03d}.png")

    writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=int(args.quality), macro_block_size=16)
    try:
        for i in range(n_frames):
            canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
            draw = ImageDraw.Draw(canvas)
            draw.text((44, 28), args.title, font=font_title, fill=ink)
            if subtitle:
                draw.text((44, 74), subtitle, font=font_small, fill=muted)
            if not args.hide_timestamp:
                draw.text((canvas_w - 198, 34), f"t = {i / fps:0.2f}s", font=font_bold, fill=ink)
                draw.text((canvas_w - 198, 72), f"frame {i:03d}/{n_frames - 1}", font=font_small, fill=muted)

            draw.text((left_x, 118), args.reference_title, font=font_bold, fill=ink)
            draw.rectangle([left_x - 10, start_y - 44, left_x + ref_size + 10, start_y + ref_size + 10], outline=accent_start, width=4)
            draw_centered(draw, (left_x, start_y - 43, left_x + ref_size, start_y - 8), "START", font_panel, accent_start)
            canvas.paste(start_img, (left_x, start_y))
            draw.rectangle([left_x - 10, goal_y - 44, left_x + ref_size + 10, goal_y + ref_size + 10], outline=accent_goal, width=4)
            draw_centered(draw, (left_x, goal_y - 43, left_x + ref_size, goal_y - 8), "GOAL", font_panel, accent_goal)
            canvas.paste(goal_img, (left_x, goal_y))
            if args.note:
                note_words = args.note.split()
                line_text = ""
                y = goal_y + ref_size + 28
                for word in note_words:
                    candidate = f"{line_text} {word}".strip()
                    if draw.textbbox((0, 0), candidate, font=font_note)[2] > ref_size:
                        draw.text((left_x, y), line_text, font=font_note, fill=muted)
                        y += 22
                        line_text = word
                    else:
                        line_text = candidate
                if line_text:
                    draw.text((left_x, y), line_text, font=font_note, fill=muted)

            for idx, ((label, reader), length) in enumerate(zip(readers, lengths)):
                col = idx % 2
                row = idx // 2
                x = grid_x + col * (panel_size + gap)
                y = top_y + row * (panel_size + label_h + gap)
                draw.rectangle([x, y, x + panel_size, y + label_h], fill=(34, 38, 47))
                draw_centered(draw, (x, y, x + panel_size, y + label_h), label, font_panel, (255, 255, 255))
                frame = reader.get_data(min(i, length - 1))
                frame_img = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
                frame_img = frame_img.resize((panel_size, panel_size), Image.Resampling.NEAREST)
                draw.rectangle([x - 1, y + label_h - 1, x + panel_size, y + label_h + panel_size], fill=(255, 255, 255), outline=line, width=2)
                canvas.paste(frame_img, (x, y + label_h))
                caption = "" if args.hide_captions else metric_text(summaries[label])
                if not args.hide_captions and i >= length and length < n_frames:
                    caption = f"{caption} | held final" if caption else "held final"
                if not args.hide_captions and caption:
                    draw.rectangle([x, y + label_h + panel_size - 32, x + panel_size, y + label_h + panel_size], fill=(255, 255, 255))
                    draw_centered(draw, (x, y + label_h + panel_size - 32, x + panel_size, y + label_h + panel_size), caption, font_tiny, ink)

            writer.append_data(np.asarray(canvas))
            if not args.no_preview and i == min(int(args.preview_frame), n_frames - 1):
                canvas.save(preview_path)
    finally:
        writer.close()
        for _, reader in readers:
            reader.close()

    manifest = {
        "out": str(args.out),
        "preview": None if args.no_preview else str(preview_path),
        "fps": fps,
        "frames": n_frames,
        "methods": [{"label": label, "case_dir": str(case_dir), "frames": length} for (label, case_dir), length in zip(method_dirs, lengths)],
        "regular_font": str(regular_font_path),
        "bold_font": str(bold_font_path),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
