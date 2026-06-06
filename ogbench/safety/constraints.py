"""Analytic OGBench cube obstacle geometry used for cache diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OBSTACLE_DATA_PATH = "ogbench/experiments/cube_obstacle/obstacle_data/summary.json"

DEFAULT_RULE = {
    "obstacle_profile": "half_ellipse",
    "table_top_z": 0.02,
    "obstacle_base_z": 0.0,
    "obstacle_peak_z": 0.08,
    "obstacle_y_bounds": [-0.06, 0.06],
    "height_margin": 0.0,
}


def load_obstacle_rule(path: Path | str | None = None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_RULE)
    rule_path = Path(path).expanduser()
    if not rule_path.is_file():
        return dict(DEFAULT_RULE)
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", payload)
    rule = dict(DEFAULT_RULE)
    for key in rule:
        if key in metadata:
            rule[key] = metadata[key]
    return rule


def half_ellipse_height(
    y_value: np.ndarray,
    *,
    y_bounds: tuple[float, float],
    base_z: float,
    peak_z: float,
) -> np.ndarray:
    y_values = np.asarray(y_value, dtype=np.float64)
    center_y = 0.5 * (float(y_bounds[0]) + float(y_bounds[1]))
    half_width = 0.5 * (float(y_bounds[1]) - float(y_bounds[0]))
    normalized = (y_values - center_y) / max(half_width, 1e-9)
    profile = np.sqrt(np.clip(1.0 - normalized**2, 0.0, None))
    return float(base_z) + (float(peak_z) - float(base_z)) * profile


def compute_rule_margin(block_pos: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    """Return signed margin: positive outside the obstacle, negative inside."""

    pos = np.asarray(block_pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[-1] < 3:
        raise ValueError(f"Expected block_pos with shape [N, 3], got {pos.shape}.")
    y = pos[:, 1]
    z = pos[:, 2]
    y_min, y_max = [float(item) for item in rule.get("obstacle_y_bounds", DEFAULT_RULE["obstacle_y_bounds"])]
    ceiling = half_ellipse_height(
        y,
        y_bounds=(y_min, y_max),
        base_z=float(rule.get("obstacle_base_z", DEFAULT_RULE["obstacle_base_z"])),
        peak_z=float(rule.get("obstacle_peak_z", DEFAULT_RULE["obstacle_peak_z"])),
    )
    table_z = float(rule.get("table_top_z", DEFAULT_RULE["table_top_z"]))
    height_margin = float(rule.get("height_margin", 0.0))

    outside_y = np.maximum(y_min - y, y - y_max)
    below_table = table_z - z
    above_obstacle = z - ceiling
    outside_margin = np.maximum.reduce((outside_y, below_table, above_obstacle))
    inside_depth = np.minimum.reduce((y - y_min, y_max - y, z - table_z, ceiling - z))
    return np.where(outside_margin > 0.0, outside_margin + height_margin, -inside_depth - height_margin).astype(np.float32)
