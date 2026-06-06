"""Obstacle-classifier adapters for the reacher latent-safety pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from reacher.safety.obstacle_classifier import ObstacleMLP


def transform_margin(margin: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the signed-margin transform used by the safety value function."""

    if transform == "identity":
        return margin
    if transform == "tanh":
        return torch.tanh(margin)
    if transform == "tanh2":
        return torch.tanh(2.0 * margin)
    raise ValueError(f"Unknown margin transform: {transform!r}.")


class ReacherObstacleClassifierMargin:
    """Turn the trained reacher obstacle classifier into a signed safety margin.

    The classifier artifact uses the convention:

    - obstacle: negative score
    - non-obstacle: positive score

    Latent-safety/PyHJ wants positive-safe and negative-unsafe rewards, so this
    adapter returns `score - threshold`. When this oracle is selected, the
    classifier margin is the safety value used by latent-safety; analytic
    task-target labels are not consulted.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
        latent_dim: int,
        threshold: str = "conformal",
        margin_transform: str = "identity",
        allow_latent_slice: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        artifact = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.artifact = artifact
        self.input_dim = int(artifact["input_dim"])
        self.latent_dim = int(latent_dim)
        self.device = device
        self.allow_latent_slice = bool(allow_latent_slice)
        if margin_transform not in {"identity", "tanh", "tanh2"}:
            raise ValueError("--margin-transform must be one of: identity, tanh, tanh2.")
        self.margin_transform = str(margin_transform)

        if self.input_dim != self.latent_dim:
            if not self.allow_latent_slice or self.input_dim > self.latent_dim:
                raise ValueError(
                    "Obstacle classifier latent dimension does not match the reacher world-model latent dimension: "
                    f"classifier input_dim={self.input_dim}, cache latent_dim={self.latent_dim}. "
                    "Retrain the classifier on the same world model, or pass --allow-classifier-latent-slice "
                    "for a temporary compatibility check."
                )

        self.model = ObstacleMLP(
            self.input_dim,
            int(artifact["hidden_dim"]),
            int(artifact["depth"]),
            float(artifact["dropout"]),
            head_style=str(artifact.get("head_style", "postnorm-gelu")),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.model.requires_grad_(False)

        self.feature_mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device)
        self.feature_std = torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
        if threshold == "conformal":
            self.threshold = float(artifact.get("conformal_safe_score_threshold", artifact.get("base_decision_threshold", 0.0)))
        elif threshold == "base":
            self.threshold = float(artifact.get("base_decision_threshold", 0.0))
        else:
            self.threshold = float(threshold)

    def _extract_classifier_input(self, state_or_latent: torch.Tensor) -> torch.Tensor:
        x = state_or_latent.to(device=self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        latent = x[..., : self.latent_dim]
        if self.input_dim != self.latent_dim:
            latent = latent[..., : self.input_dim]
        return (latent - self.feature_mean) / self.feature_std

    @torch.no_grad()
    def score(self, state_or_latent: torch.Tensor) -> torch.Tensor:
        return self.model(self._extract_classifier_input(state_or_latent))

    @torch.no_grad()
    def margin(self, state_or_latent: torch.Tensor) -> torch.Tensor:
        return transform_margin(self.raw_margin(state_or_latent), self.margin_transform)

    @torch.no_grad()
    def raw_margin(self, state_or_latent: torch.Tensor) -> torch.Tensor:
        return self.score(state_or_latent) - float(self.threshold)

    @torch.no_grad()
    def __call__(self, state_or_latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        score = self.score(state_or_latent)
        raw_margin = score - float(self.threshold)
        margin = transform_margin(raw_margin, self.margin_transform)
        info = {
            "classifier_score_mean": float(score.mean().item()),
            "classifier_raw_margin_mean": float(raw_margin.mean().item()),
            "classifier_margin_mean": float(margin.mean().item()),
            "classifier_threshold": float(self.threshold),
            "classifier_margin_transform": self.margin_transform,
            "classifier_input_dim": float(self.input_dim),
            "classifier_used_latent_slice": float(self.input_dim != self.latent_dim),
        }
        return margin.squeeze(0), info

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "threshold": self.threshold,
            "margin_transform": self.margin_transform,
            "allow_latent_slice": self.allow_latent_slice,
            "score_sign_convention": self.artifact.get("score_sign_convention", {}),
        }
