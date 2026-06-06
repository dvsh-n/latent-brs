#!/usr/bin/env python3
"""Select OGBench cube episodes that cross the analytic half-ellipse obstacle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ogbench.safety.constraints import DEFAULT_OBSTACLE_DATA_PATH, compute_rule_margin, load_obstacle_rule

DEFAULT_DATASET_PATH = "ogbench/data/test_data/ogbench_cube_test.h5"
DEFAULT_OUTPUT = "ogbench/safety/runs/heldout_sets/cube_test_geometry_unsafe_10.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--obstacle-data-path", type=Path, default=Path(DEFAULT_OBSTACLE_DATA_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--min-first-unsafe-step", type=int, default=1)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be positive.")
    dataset_path = args.dataset_path.expanduser().resolve()
    rule = load_obstacle_rule(args.obstacle_data_path)
    selected: list[int] = []
    details: list[dict[str, Any]] = []
    total_unsafe = 0

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        for ep, length in enumerate(ep_len.tolist()):
            if int(length) < 2:
                continue
            start = int(ep_offset[ep])
            stop = start + int(length)
            block_pos = np.asarray(h5["block_pos"][start:stop], dtype=np.float32)
            margins = compute_rule_margin(block_pos, rule)
            unsafe_steps = np.flatnonzero(margins <= 0.0)
            if unsafe_steps.size == 0:
                continue
            total_unsafe += 1
            first_unsafe = int(unsafe_steps[0])
            if first_unsafe < int(args.min_first_unsafe_step):
                continue
            if len(selected) < int(args.num_episodes):
                selected.append(int(ep))
                details.append(
                    {
                        "episode_idx": int(ep),
                        "ep_len": int(length),
                        "first_unsafe_step": first_unsafe,
                        "unsafe_frame_count": int(unsafe_steps.size),
                        "min_margin": float(np.min(margins)),
                    }
                )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "obstacle_rule": jsonable(rule),
        "episode_indices": selected,
        "num_selected": len(selected),
        "requested_num_episodes": int(args.num_episodes),
        "num_total_geometry_unsafe_episodes": int(total_unsafe),
        "min_first_unsafe_step": int(args.min_first_unsafe_step),
        "details": details,
    }
    output.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "num_selected": len(selected), "episode_indices": selected}, indent=2))


if __name__ == "__main__":
    main()
