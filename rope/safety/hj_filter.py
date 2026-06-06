"""Closed-loop latent HJ safety filter for rope experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import numpy as np
import torch
from gymnasium import spaces

ROOT_DIR = Path(__file__).resolve().parents[2]
LATENT_SAFETY_DIR = ROOT_DIR / "third_party" / "latent-safety"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LATENT_SAFETY_DIR) not in sys.path:
    sys.path.insert(0, str(LATENT_SAFETY_DIR))

from rope.plan import plan_ilqr_mpc as ilqr_base
from rope.safety.latent_env import load_latent_safety_components
from rope.safety.train_pyhj_rope import build_policy
from rope.train.mlpdyn_train import required_markov_history

from PyHJ.data import Batch


def parse_hidden_sizes(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    sizes = [int(item) for item in str(value).replace(",", " ").split() if item]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"Invalid hidden sizes: {value!r}.")
    return sizes


@dataclass(frozen=True)
class HJFilterDecision:
    action_raw: np.ndarray
    action_norm: np.ndarray
    record: dict[str, Any]


class RopeHJSafetyFilter:
    """Paper-style least-restrictive LatentSafe filter.

    A nominal action is converted to the normalized action space used by the
    world model. The filter predicts the next latent state, evaluates
    ``B = min(l, V)`` there, and switches to the PyHJ safety actor when
    ``B <= epsilon``.
    """

    def __init__(
        self,
        *,
        cache_path: Path,
        policy_path: Path,
        classifier_checkpoint: Path,
        device_arg: str = "auto",
        model_dir: Path | None = None,
        checkpoint: Path | None = None,
        classifier_threshold: str = "conformal",
        margin_transform: str = "auto",
        actor_hidden: str | list[int] = "512 512 512 512",
        critic_hidden: str | list[int] = "512 512 512 512",
        action_low: float = -2.0,
        action_high: float = 2.0,
        epsilon: float = 0.0,
    ) -> None:
        self.components = load_latent_safety_components(
            cache_path=cache_path,
            model_dir=model_dir,
            checkpoint=checkpoint,
            device_arg=device_arg,
            oracle_kind="classifier",
            classifier_checkpoint=classifier_checkpoint,
            classifier_threshold=classifier_threshold,
            margin_transform=margin_transform,
        )
        self.device = self.components.device
        self.cache = self.components.cache
        self.metadata = self.cache["metadata"]
        self.state_dim = int(self.metadata["markov_state_dim"])
        self.action_dim = int(self.metadata["action_dim"])
        self.latent_dim = int(self.metadata["latent_dim"])
        self.markov_deriv = int(self.metadata.get("markov_deriv", 1))
        self.history_len = required_markov_history(self.markov_deriv)
        self.img_size = int(self.metadata.get("model_config", {}).get("img_size", 224))
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.epsilon = float(epsilon)
        self.action_mean = np.asarray(self.metadata["action_mean"], dtype=np.float32).reshape(-1)
        self.action_std = np.asarray(self.metadata["action_std"], dtype=np.float32).reshape(-1)

        action_space = spaces.Box(
            low=np.full((self.action_dim,), self.action_low, dtype=np.float32),
            high=np.full((self.action_dim,), self.action_high, dtype=np.float32),
            dtype=np.float32,
        )
        policy_args = SimpleNamespace(
            actor_hidden=parse_hidden_sizes(actor_hidden),
            critic_hidden=parse_hidden_sizes(critic_hidden),
            actor_lr=1e-4,
            critic_lr=1e-3,
            weight_decay=1e-3,
            tau=0.005,
            gamma=0.9999,
            exploration_noise=0.1,
            actor_gradient_steps=1,
            n_step=1,
            optimizer="adam",
            policy_variant="dinowm",
        )
        self.policy = build_policy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_space=action_space,
            device=self.device,
            args=policy_args,
        )
        self.policy.load_state_dict(torch.load(policy_path.expanduser().resolve(), map_location=self.device))
        self.policy.eval()

        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.latent_history: list[torch.Tensor] = []

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        latent = ilqr_base.encode_single_frame(
            self.components.model,
            frame,
            device=self.device,
            img_size=self.img_size,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
        )
        return latent[: self.latent_dim].detach()

    def reset(self, frame: np.ndarray) -> torch.Tensor:
        latent = self.encode_frame(frame)
        self.latent_history = [latent] * self.history_len
        return self.current_state()

    def append_frame(self, frame: np.ndarray) -> torch.Tensor:
        latent = self.encode_frame(frame)
        self.latent_history.append(latent)
        self.latent_history = self.latent_history[-self.history_len :]
        return self.current_state()

    def current_state(self) -> torch.Tensor:
        if not self.latent_history:
            raise RuntimeError("Call reset(frame) before using the HJ safety filter.")
        state = ilqr_base.make_markov_state(self.latent_history, self.markov_deriv).to(self.device, dtype=torch.float32)
        if int(state.numel()) != self.state_dim:
            raise ValueError(f"Built state dim {state.numel()}, expected {self.state_dim}.")
        return state

    def raw_to_norm(self, action_raw: np.ndarray) -> np.ndarray:
        action = np.asarray(action_raw, dtype=np.float32).reshape(self.action_dim)
        norm = (action - self.action_mean) / self.action_std
        return np.clip(norm, self.action_low, self.action_high).astype(np.float32)

    def norm_to_raw(self, action_norm: np.ndarray) -> np.ndarray:
        action = np.asarray(action_norm, dtype=np.float32).reshape(self.action_dim)
        return (action * self.action_std + self.action_mean).astype(np.float32)

    @torch.no_grad()
    def safety_action_norm(self, state: torch.Tensor) -> np.ndarray:
        batch = Batch(obs=state.detach().cpu().numpy().astype(np.float32)[None, :], info=Batch())
        raw = self.policy(batch, model="actor").act
        raw_np = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        mapped = self.policy.map_action(raw_np)
        return np.asarray(mapped[0], dtype=np.float32)

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> float:
        obs = state.to(self.device, dtype=torch.float32).reshape(1, -1)
        batch = Batch(obs=obs, info=Batch())
        act = self.policy(batch, model="actor").act
        q = self.policy.critic(obs, act).flatten()[0]
        return float(q.item())

    @torch.no_grad()
    def margin(self, state: torch.Tensor) -> float:
        margin, _ = self.components.oracle(state.to(self.device, dtype=torch.float32))
        return float(margin.item())

    @torch.no_grad()
    def evaluate_state(self, state: torch.Tensor, prefix: str) -> dict[str, float]:
        margin = self.margin(state)
        value = self.value(state)
        barrier = min(margin, value)
        return {
            f"{prefix}_l": float(margin),
            f"{prefix}_V": float(value),
            f"{prefix}_B": float(barrier),
        }

    @torch.no_grad()
    def predict_next(self, state: torch.Tensor, action_norm: np.ndarray) -> torch.Tensor:
        action = torch.as_tensor(action_norm, dtype=torch.float32, device=self.device)
        return self.components.dynamics.step(state.to(self.device, dtype=torch.float32), action).detach()

    @torch.no_grad()
    def filter_action(self, nominal_action_raw: np.ndarray) -> HJFilterDecision:
        state = self.current_state()
        current = self.evaluate_state(state, "current")

        nominal_norm = self.raw_to_norm(nominal_action_raw)
        nominal_next = self.predict_next(state, nominal_norm)
        nominal_next_eval = self.evaluate_state(nominal_next, "nominal_next")

        safe_norm = self.safety_action_norm(state)
        safe_raw = self.norm_to_raw(safe_norm)
        safe_next = self.predict_next(state, safe_norm)
        safe_next_eval = self.evaluate_state(safe_next, "safe_next")

        override_reasons: list[str] = []
        if current["current_l"] <= 0.0:
            override_reasons.append("current_classifier_failure")
        if nominal_next_eval["nominal_next_B"] <= self.epsilon:
            override_reasons.append("nominal_next_barrier_below_epsilon")

        override = bool(override_reasons)
        executed_norm = safe_norm if override else nominal_norm
        executed_raw = safe_raw if override else np.asarray(nominal_action_raw, dtype=np.float32).reshape(self.action_dim)

        record: dict[str, Any] = {
            **current,
            **nominal_next_eval,
            **safe_next_eval,
            "epsilon": float(self.epsilon),
            "override": override,
            "override_reason": ",".join(override_reasons) if override_reasons else "nominal_safe",
            "classifier_unsafe": bool(current["current_l"] <= 0.0),
            "nominal_action_raw": np.asarray(nominal_action_raw, dtype=np.float32).reshape(self.action_dim).tolist(),
            "nominal_action_norm": nominal_norm.tolist(),
            "safety_action_raw": safe_raw.tolist(),
            "safety_action_norm": safe_norm.tolist(),
            "executed_action_raw": executed_raw.tolist(),
            "executed_action_norm": executed_norm.tolist(),
        }
        return HJFilterDecision(action_raw=executed_raw, action_norm=executed_norm, record=record)
