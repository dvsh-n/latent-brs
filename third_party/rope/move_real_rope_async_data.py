#!/usr/bin/env python3
"""Move real-rope async camera side files to external storage as they appear."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "rope" / "data" / "real_data"
DEFAULT_DEST = Path("/media/daniel/My Passport/real_rope_data")
DEFAULT_STABLE_SECONDS = 20.0
DEFAULT_POLL_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--stable-seconds", type=float, default=DEFAULT_STABLE_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Scan once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print moves without copying files.")
    return parser.parse_args()


def is_async_side_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix == ".mp4":
        return True
    return path.name.endswith("_async_camera.h5") or path.name.endswith("_async_camera.csv")


def stable_size(path: Path, wait_seconds: float) -> bool:
    try:
        first = path.stat()
    except FileNotFoundError:
        return False
    if first.st_size <= 0:
        return False
    time.sleep(max(0.0, wait_seconds))
    try:
        second = path.stat()
    except FileNotFoundError:
        return False
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns


def copy_with_progress(source: Path, dest: Path, *, total: int) -> None:
    with source.open("rb") as src, dest.open("wb") as dst:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=source.name,
        ) as progress:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                progress.update(len(chunk))


def move_file(source: Path, dest_dir: Path, *, dry_run: bool) -> bool:
    dest = dest_dir / source.name
    if dest.exists():
        print(f"skip existing destination: {dest}")
        return False
    size = source.stat().st_size
    if dry_run:
        print(f"would move: {source} -> {dest} ({size / 1024**3:.2f} GiB)")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f".{dest.name}.moving.{os.getpid()}")
    if tmp_dest.exists():
        tmp_dest.unlink()
    copy_with_progress(source, tmp_dest, total=size)
    shutil.copystat(source, tmp_dest)
    os.replace(tmp_dest, dest)
    source.unlink()
    print(f"moved: {source} -> {dest}")
    return True


def scan_once(source_dir: Path, dest_dir: Path, *, stable_seconds: float, dry_run: bool) -> int:
    moved = 0
    for path in sorted(source_dir.iterdir()):
        if not is_async_side_file(path):
            continue
        if not stable_size(path, stable_seconds):
            print(f"skip changing file: {path}")
            continue
        moved += int(move_file(path, dest_dir, dry_run=dry_run))
    return moved


def main() -> None:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    dest_dir = args.dest.expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if args.stable_seconds < 0.0:
        raise ValueError("--stable-seconds cannot be negative.")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive.")

    while True:
        scan_once(source_dir, dest_dir, stable_seconds=args.stable_seconds, dry_run=args.dry_run)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
