from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    depth: int,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be at least 1.")

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.GELU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ResNetBackbone(nn.Module):
    def __init__(self, block_counts: tuple[int, int, int, int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(64, block_counts[0], stride=1)
        self.layer2 = self._make_layer(128, block_counts[1], stride=2)
        self.layer3 = self._make_layer(256, block_counts[2], stride=2)
        self.layer4 = self._make_layer(512, block_counts[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ResNet18LatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 24,
        proj_hidden_dim: int = 512,
        proj_depth: int = 1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.backbone = ResNetBackbone(block_counts=(2, 2, 2, 2))
        self.projector = build_mlp(
            in_dim=self.backbone.out_dim,
            out_dim=latent_dim,
            hidden_dim=proj_hidden_dim,
            depth=proj_depth,
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.latent_dim = int(latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        latents = self.projector(features)
        return self.output_norm(latents)


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if state_dim % latent_dim != 0:
            raise ValueError("state_dim must be divisible by latent_dim.")

        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.net = build_mlp(
            in_dim=state_dim + action_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


def make_history_states(frame_latents: torch.Tensor, history: int) -> torch.Tensor:
    if frame_latents.ndim != 3:
        raise ValueError("frame_latents must have shape [batch, time, latent_dim].")
    batch, total_frames, latent_dim = frame_latents.shape
    if total_frames < history:
        raise ValueError("Not enough frames to build history states.")

    windows = []
    for start in range(total_frames - history + 1):
        window = frame_latents[:, start : start + history].reshape(batch, history * latent_dim)
        windows.append(window)
    return torch.stack(windows, dim=1)


def rollout_dynamics(
    dynamics: nn.Module,
    initial_state: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, horizon, action_dim].")

    predictions = []
    state = initial_state
    latent_dim = int(dynamics.latent_dim)
    for step in range(actions.shape[1]):
        next_latent = dynamics(state, actions[:, step])
        state = torch.cat((state[:, latent_dim:], next_latent), dim=-1)
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def symmetric_stopgrad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = F.mse_loss(pred, target.detach())
    backward = F.mse_loss(pred.detach(), target)
    return 0.5 * (forward + backward)


def curvature_loss(frame_latents: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if frame_latents.shape[1] < 3:
        return frame_latents.new_zeros(())

    velocities = frame_latents[:, 1:] - frame_latents[:, :-1]
    v_t = velocities[:, :-1]
    v_next = velocities[:, 1:]
    numerator = (v_t * v_next).sum(dim=-1)
    denominator = v_t.norm(dim=-1).clamp_min(eps) * v_next.norm(dim=-1).clamp_min(eps)
    return (1.0 - numerator / denominator).mean()


@dataclass
class LatentDynamicsOutput:
    loss: torch.Tensor
    state_loss: torch.Tensor
    dynamics_loss: torch.Tensor
    curvature_loss: torch.Tensor
    encoded_latents: torch.Tensor
    predicted_states: torch.Tensor
    target_states: torch.Tensor


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 24,
        history: int = 3,
        action_dim: int = 2,
        encoder_proj_hidden_dim: int = 512,
        encoder_proj_depth: int = 1,
        dynamics_hidden_dim: int = 1024,
        dynamics_depth: int = 3,
        curvature_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if history < 1:
            raise ValueError("history must be at least 1.")

        self.encoder = ResNet18LatentEncoder(
            latent_dim=latent_dim,
            proj_hidden_dim=encoder_proj_hidden_dim,
            proj_depth=encoder_proj_depth,
        )
        self.history = int(history)
        self.latent_dim = int(latent_dim)
        self.state_dim = int(history * latent_dim)
        self.action_dim = int(action_dim)
        self.dynamics = LatentDynamicsMLP(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=dynamics_hidden_dim,
            depth=dynamics_depth,
        )
        self.curvature_weight = float(curvature_weight)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        latents = self.encoder(frames.reshape(batch * time, channels, height, width))
        return latents.reshape(batch, time, -1)

    def compute_losses(
        self,
        latents_a: torch.Tensor,
        latents_b: torch.Tensor,
        actions: torch.Tensor,
    ) -> LatentDynamicsOutput:
        states_a = make_history_states(latents_a, self.history)
        states_b = make_history_states(latents_b, self.history)
        initial_a = states_a[:, 0]
        initial_b = states_b[:, 0]
        target_a = states_a[:, 1:]
        target_b = states_b[:, 1:]

        pred_a = rollout_dynamics(self.dynamics, initial_a, actions)
        pred_b = rollout_dynamics(self.dynamics, initial_b, actions)
        pred_next_a = pred_a[..., -self.latent_dim :]
        pred_next_b = pred_b[..., -self.latent_dim :]
        target_next_a = target_a[..., -self.latent_dim :]
        target_next_b = target_b[..., -self.latent_dim :]

        state_loss = symmetric_stopgrad_mse(initial_a, initial_b)
        dyn_loss_a = symmetric_stopgrad_mse(pred_next_a, target_next_b)
        dyn_loss_b = symmetric_stopgrad_mse(pred_next_b, target_next_a)
        dynamics_loss = 0.5 * (dyn_loss_a + dyn_loss_b)

        curvature = latents_a.new_zeros(())
        if self.curvature_weight > 0.0:
            curvature = 0.5 * (curvature_loss(latents_a) + curvature_loss(latents_b))

        total_loss = state_loss + dynamics_loss + self.curvature_weight * curvature
        return LatentDynamicsOutput(
            loss=total_loss,
            state_loss=state_loss,
            dynamics_loss=dynamics_loss,
            curvature_loss=curvature,
            encoded_latents=0.5 * (latents_a + latents_b),
            predicted_states=0.5 * (pred_a + pred_b),
            target_states=0.5 * (target_a + target_b),
        )

    def forward(
        self,
        frames_a: torch.Tensor,
        frames_b: torch.Tensor,
        actions: torch.Tensor,
    ) -> LatentDynamicsOutput:
        latents_a = self.encode_frames(frames_a)
        latents_b = self.encode_frames(frames_b)
        return self.compute_losses(latents_a=latents_a, latents_b=latents_b, actions=actions)
