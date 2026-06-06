"""Runtime Latent Policy Barrier for rope Markov-state planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class RopeLPBBarrier:
    """Nearest-neighbor latent in-distribution barrier.

    The bank stores whitened Markov-state prototypes. Query states are whitened
    with the saved statistics, then scored by distance to the nearest prototype.
    """

    def __init__(
        self,
        bank_path: Path,
        *,
        device: torch.device,
        threshold_scale: float = 1.0,
    ) -> None:
        self.bank_path = bank_path.expanduser().resolve()
        payload = torch.load(self.bank_path, map_location="cpu", weights_only=False)
        self.payload = payload
        self.prototypes = payload["prototypes"].to(device=device, dtype=torch.float32)
        self.state_mean = payload["state_mean"].to(device=device, dtype=torch.float32)
        self.state_std = payload["state_std"].to(device=device, dtype=torch.float32).clamp_min(1e-6)
        self.threshold = float(payload["threshold"])
        self.threshold_scale = float(threshold_scale)
        self.metadata: dict[str, Any] = dict(payload.get("metadata", {}))
        if self.prototypes.ndim != 2:
            raise ValueError(f"Expected prototypes with shape [N, D], got {tuple(self.prototypes.shape)}.")
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("LPB bank state_mean and state_std shapes do not match.")
        if self.state_mean.numel() != self.prototypes.shape[-1]:
            raise ValueError(
                "LPB bank dimension mismatch: "
                f"mean has {self.state_mean.numel()} values, prototypes have dim {self.prototypes.shape[-1]}."
            )

    @property
    def state_dim(self) -> int:
        return int(self.prototypes.shape[-1])

    @property
    def scaled_threshold(self) -> float:
        return float(self.threshold * self.threshold_scale)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_t = state.to(device=self.prototypes.device, dtype=torch.float32)
        return (state_t - self.state_mean) / self.state_std

    def nearest_distance(self, state: torch.Tensor) -> torch.Tensor:
        query = self.normalize_state(state)
        original_shape = query.shape[:-1]
        query_flat = query.reshape(-1, query.shape[-1])
        diff = query_flat[:, None, :] - self.prototypes[None, :, :]
        distances_sq = diff.square().sum(dim=-1)
        nearest = distances_sq.clamp_min(1e-12).sqrt().min(dim=-1).values
        return nearest.reshape(original_shape)

    def state_penalty(self, state: torch.Tensor) -> torch.Tensor:
        distance = self.nearest_distance(state)
        excess = torch.relu(distance - self.scaled_threshold)
        return excess.square()

    def trajectory_penalty(self, states: torch.Tensor, *, stage_only: bool = True) -> torch.Tensor:
        selected = states[:-1] if stage_only and states.shape[0] > 1 else states
        if selected.numel() == 0:
            return torch.zeros((), dtype=states.dtype, device=states.device)
        return self.state_penalty(selected).mean()

    @torch.no_grad()
    def diagnostics(self, states: torch.Tensor, *, stage_only: bool = True) -> dict[str, float]:
        selected = states[:-1] if stage_only and states.shape[0] > 1 else states
        distances = self.nearest_distance(selected)
        threshold = self.scaled_threshold
        excess = torch.relu(distances - threshold)
        return {
            "lpb_threshold": float(threshold),
            "lpb_distance_mean": float(distances.mean().item()),
            "lpb_distance_min": float(distances.min().item()),
            "lpb_distance_max": float(distances.max().item()),
            "lpb_violation_rate": float((distances > threshold).float().mean().item()),
            "lpb_penalty": float(excess.square().mean().item()),
        }
