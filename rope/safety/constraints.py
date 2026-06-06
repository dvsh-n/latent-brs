"""Analytic rope obstacle-zone margins used before classifier-only runs.

The rule mirrors the obstacle-classifier labels: low rope is unsafe only inside
the configured obstacle reach interval. Low rope outside that interval remains
safe for this obstacle setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_OBSTACLE_DATA_PATH = "rope/safety/obstacle_data/obstacle_classifier_data.pt"


def load_obstacle_rule(obstacle_data_path: Path = Path(DEFAULT_OBSTACLE_DATA_PATH)) -> dict[str, Any]:
    payload = torch.load(obstacle_data_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    sag_profile = payload["sag_profile"]
    return {
        "obstacle_reach": np.asarray(metadata["obstacle_reach"], dtype=np.float32),
        "low_rope_cutoff": float(metadata["low_rope_cutoff"]),
        "obstacle_profile": metadata.get("obstacle_profile", "flat_cutoff"),
        "obstacle_base_height": float(metadata.get("obstacle_base_height", metadata.get("table_top_z", 0.0))),
        "obstacle_height": float(metadata.get("obstacle_height", metadata["low_rope_cutoff"])),
        "width_values": np.asarray(sag_profile["width_values"], dtype=np.float32),
        "sag_drop": np.asarray(sag_profile["sag_drop"], dtype=np.float32),
        "label_rule": metadata.get("label_rule", "obstacle_zone_low_rope"),
        "label_rule_description": metadata.get(
            "label_rule_description",
            "unsafe iff reach is inside obstacle_reach and estimated_low_rope_height <= low_rope_cutoff",
        ),
        "source_path": str(obstacle_data_path),
    }


def compute_rule_margin(task_target: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    """Return positive-safe/negative-unsafe obstacle-zone margins.

    Negative means both conditions are true:
    1. reach is inside the obstacle interval
    2. estimated low-rope height is below the cutoff

    Taking max(reach_outside_margin, height_margin) implements the union of safe
    conditions: outside the obstacle band OR above the height cutoff.
    """

    states = np.asarray(task_target, dtype=np.float32)
    if states.ndim != 2 or states.shape[-1] < 3:
        raise ValueError(f"Expected task_target with shape [N, >=3], got {states.shape}.")
    reach = states[:, 0]
    height = states[:, 1]
    width = states[:, 2]
    reach_low, reach_high = np.asarray(rule["obstacle_reach"], dtype=np.float32)
    sag = np.interp(width, np.asarray(rule["width_values"]), np.asarray(rule["sag_drop"]))
    low_rope_height = height - sag
    reach_outside_margin = np.maximum(reach_low - reach, reach - reach_high)
    if str(rule.get("obstacle_profile", "")) == "half_ellipse":
        center = 0.5 * (float(reach_low) + float(reach_high))
        half_width = 0.5 * (float(reach_high) - float(reach_low))
        if half_width <= 0.0:
            raise ValueError("Obstacle reach interval must have positive width.")
        normalized = (reach - center) / half_width
        profile = np.sqrt(np.clip(1.0 - normalized**2, 0.0, None))
        obstacle_height = float(rule["obstacle_base_height"]) + (
            float(rule["obstacle_height"]) - float(rule["obstacle_base_height"])
        ) * profile
    else:
        obstacle_height = np.full_like(low_rope_height, float(rule["low_rope_cutoff"]))
    height_margin = low_rope_height - obstacle_height
    return np.maximum(reach_outside_margin, height_margin).astype(np.float32)
