"""Latent safety env for rope PyHJ training and smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rope.plan import plan_ilqr_mpc as ilqr_base
from rope.safety.classifier_oracle import RopeObstacleClassifierMargin
from rope.safety.compat import register_legacy_checkpoint_aliases


@dataclass(frozen=True)
class LatentSafetyComponents:
    model: torch.nn.Module
    dynamics: ilqr_base.MarkovDynamicsTorch
    cache: dict[str, Any]
    oracle: "MarginOracle"
    device: torch.device


class MarginOracle(Protocol):
    def __call__(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return signed margin and diagnostics for a latent Markov state."""


class KnnMarginOracle:
    """Temporary lookup oracle over cached margins."""

    def __init__(
        self,
        states: torch.Tensor,
        margins: torch.Tensor,
        *,
        device: torch.device,
        k: int = 5,
        pessimistic: bool = True,
    ) -> None:
        if states.ndim != 2:
            raise ValueError(f"Expected states with shape [N, D], got {tuple(states.shape)}.")
        if margins.ndim != 1 or margins.shape[0] != states.shape[0]:
            raise ValueError("margins must have shape [N] matching states.")
        self.states = states.to(device=device, dtype=torch.float32)
        self.margins = margins.to(device=device, dtype=torch.float32)
        self.device = device
        self.k = max(1, int(k))
        self.pessimistic = bool(pessimistic)

    @torch.no_grad()
    def __call__(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        query = state.unsqueeze(0) if state.ndim == 1 else state
        query = query.to(device=self.device, dtype=torch.float32)
        distances = torch.cdist(query, self.states)
        k = min(self.k, self.states.shape[0])
        nearest_dist, nearest_idx = torch.topk(distances, k=k, largest=False, dim=-1)
        nearest_margin = self.margins[nearest_idx]
        margin = nearest_margin.min(dim=-1).values if self.pessimistic else nearest_margin.mean(dim=-1)
        info = {
            "knn_distance_mean": float(nearest_dist.mean().item()),
            "knn_margin_mean": float(nearest_margin.mean().item()),
            "knn_margin_min": float(nearest_margin.min().item()),
        }
        return margin.squeeze(0), info


class RopeLatentSafetyEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dynamics: ilqr_base.MarkovDynamicsTorch,
        cache: dict[str, Any],
        oracle: MarginOracle,
        device: torch.device,
        max_episode_steps: int = 25,
        action_low: float = -2.0,
        action_high: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.cache = cache
        self.oracle = oracle
        self.device = device
        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")

        markov_state = cache["markov_state"].float()
        action_norm = cache["action_norm"].float()
        valid_transition = cache["valid_transition"].bool()
        valid_indices = torch.nonzero(valid_transition, as_tuple=False).flatten().cpu().numpy()
        if valid_indices.size == 0:
            raise ValueError("Latent cache has no valid transitions.")

        self.states = markov_state
        self.valid_indices = valid_indices
        self.action_dim = int(action_norm.shape[-1])
        self.state_dim = int(markov_state.shape[-1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full((self.action_dim,), float(action_low), dtype=np.float32),
            high=np.full((self.action_dim,), float(action_high), dtype=np.float32),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(seed)
        self.state: torch.Tensor | None = None
        self.start_index: int | None = None
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        del options
        index = int(self.rng.choice(self.valid_indices))
        self.start_index = index
        self.steps = 0
        self.state = self.states[index].to(self.device, dtype=torch.float32)
        margin, oracle_info = self.oracle(self.state)
        info = {
            "cache_index": index,
            "safety_margin": float(margin.item()),
            "is_failure": bool(margin.item() <= 0.0),
            "steps": 0,
            **oracle_info,
        }
        return self.state.detach().cpu().numpy().astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr = np.clip(action_arr, self.action_space.low, self.action_space.high)
        action_t = torch.from_numpy(action_arr).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            next_state = self.dynamics.step(self.state, action_t)
            margin, oracle_info = self.oracle(next_state)
        self.state = next_state.detach()
        self.steps += 1
        reward = float(margin.item())
        terminated = reward <= 0.0
        truncated = self.steps >= self.max_episode_steps
        info = {
            "cache_index": -1,
            "safety_margin": reward,
            "is_failure": bool(terminated),
            "steps": self.steps,
            **oracle_info,
        }
        return self.state.detach().cpu().numpy().astype(np.float32), reward, terminated, truncated, info


def load_latent_safety_components(
    *,
    cache_path: Path,
    model_dir: Path | None = None,
    checkpoint: Path | None = None,
    device_arg: str = "auto",
    oracle_kind: str = "knn",
    classifier_checkpoint: Path | None = None,
    classifier_threshold: str = "conformal",
    margin_transform: str = "auto",
    allow_classifier_latent_slice: bool = False,
    knn_k: int = 5,
    pessimistic: bool = True,
) -> LatentSafetyComponents:
    device = ilqr_base.require_device(device_arg)
    cache = torch.load(cache_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    model_dir_resolved = (
        model_dir.expanduser().resolve()
        if model_dir is not None
        else Path(str(metadata["model_dir"])).expanduser().resolve()
    )
    checkpoint_resolved = (
        checkpoint.expanduser().resolve()
        if checkpoint is not None
        else Path(str(metadata["checkpoint"])).expanduser().resolve()
    )
    if not model_dir_resolved.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir_resolved}")
    register_legacy_checkpoint_aliases()
    model = ilqr_base.load_model(checkpoint_resolved, device)
    dynamics = ilqr_base.MarkovDynamicsTorch(
        model,
        int(metadata["markov_state_dim"]),
        int(metadata["action_dim"]),
        device,
    )

    if oracle_kind == "knn":
        oracle: MarginOracle = KnnMarginOracle(
            cache["markov_state"],
            cache["safety_margin"],
            device=device,
            k=knn_k,
            pessimistic=pessimistic,
        )
    elif oracle_kind == "classifier":
        if classifier_checkpoint is None:
            raise ValueError("--classifier-checkpoint is required for --oracle classifier.")
        resolved_margin_transform = str(margin_transform)
        if resolved_margin_transform == "auto":
            resolved_margin_transform = str(metadata.get("margin_transform", "identity"))
        oracle = RopeObstacleClassifierMargin(
            classifier_checkpoint,
            device=device,
            latent_dim=int(metadata["latent_dim"]),
            threshold=classifier_threshold,
            margin_transform=resolved_margin_transform,
            allow_latent_slice=allow_classifier_latent_slice,
        )
    else:
        raise ValueError(f"Unknown oracle kind: {oracle_kind}")

    return LatentSafetyComponents(model=model, dynamics=dynamics, cache=cache, oracle=oracle, device=device)


RopeLatentSafetySmokeEnv = RopeLatentSafetyEnv
