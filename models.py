import torch
import torch.nn as nn
from typing import Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    depth: int,
    activation_fn: Type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    if depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width), activation_fn()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_width, hidden_width), activation_fn()])
    layers.append(nn.Linear(hidden_width, output_dim))
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    """Small CNN that maps images or frame stacks into a latent vector."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_width: int = 256,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        act = activation_fn
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            act(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 4 * 4, hidden_width),
            act(),
            nn.Linear(hidden_width, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class SnapshotEncoder(nn.Module):
    """Encodes one image into the directly propagated state-like block."""

    def __init__(
        self,
        image_channels: int,
        state_dim: int,
        hidden_width: int,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.net = ConvEncoder(
            in_channels=image_channels,
            latent_dim=state_dim,
            hidden_width=hidden_width,
            activation_fn=activation_fn,
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.net(frame)


class HistoryEncoder(nn.Module):
    """Encodes a short image history into auxiliary Koopman observables."""

    def __init__(
        self,
        history_length: int,
        image_channels: int,
        observable_dim: int,
        hidden_width: int,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.history_length = history_length
        self.image_channels = image_channels
        self.net = ConvEncoder(
            in_channels=history_length * image_channels,
            latent_dim=observable_dim,
            hidden_width=hidden_width,
            activation_fn=activation_fn,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        b, h, c, height, width = history.shape
        if h != self.history_length or c != self.image_channels:
            raise ValueError(
                f"Expected history shape [B, {self.history_length}, {self.image_channels}, H, W], "
                f"got {tuple(history.shape)}"
            )
        stacked = history.reshape(b, h * c, height, width)
        return self.net(stacked)


class DeepKoopmanImageNoDec(nn.Module):
    """
    Image-only decoder-free Koopman model with a split lifted state:
    z = [s(I_t), o(I_{t-H+1:t}, u_{t-H:t-1})].
    """

    def __init__(
        self,
        image_channels: int,
        control_dim: int,
        history_length: int,
        state_dim: int,
        observable_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.control_dim = control_dim
        self.history_length = history_length
        self.state_dim = state_dim
        self.observable_dim = observable_dim
        self.latent_dim = state_dim + observable_dim

        self.snapshot_encoder = SnapshotEncoder(
            image_channels=image_channels,
            state_dim=state_dim,
            hidden_width=hidden_width,
            activation_fn=activation_fn,
        )
        self.history_encoder = HistoryEncoder(
            history_length=history_length,
            image_channels=image_channels,
            observable_dim=observable_dim,
            hidden_width=hidden_width,
            activation_fn=activation_fn,
        )

        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.latent_dim, bias=False)

    def encode_snapshot(self, frame: torch.Tensor) -> torch.Tensor:
        return self.snapshot_encoder(frame)

    def encode_observables(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:
        return self.history_encoder(history)

    def lift_state(
        self,
        frame: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        s = self.encode_snapshot(frame)
        o = self.encode_observables(history)
        return torch.cat([s, o], dim=-1)

    def _encode_targets(
        self,
        frame_seq: torch.Tensor,
        history_seq: torch.Tensor,
        *,
        chunk_size: int | None = None,
        requires_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, m, c, h, w = frame_seq.shape
        frame_flat = frame_seq.reshape(b * m, c, h, w)
        history_flat = history_seq.reshape(b * m, self.history_length, c, h, w)

        if chunk_size is None or chunk_size <= 0:
            chunk_size = b * m

        s_target_chunks = []
        z_target_chunks = []
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            for start in range(0, b * m, chunk_size):
                end = min(start + chunk_size, b * m)
                s_target_chunk = self.encode_snapshot(frame_flat[start:end])
                o_target_chunk = self.encode_observables(history_flat[start:end])
                z_target_chunk = torch.cat([s_target_chunk, o_target_chunk], dim=-1)
                s_target_chunks.append(s_target_chunk)
                z_target_chunks.append(z_target_chunk)

        s_target_flat = torch.cat(s_target_chunks, dim=0)
        z_target_flat = torch.cat(z_target_chunks, dim=0)
        s_target_seq = s_target_flat.view(b, m, self.state_dim)
        z_target_seq = z_target_flat.view(b, m, self.latent_dim)
        return z_target_seq, s_target_seq

    def forward(
        self,
        frame_k: torch.Tensor,
        history_k: torch.Tensor,
        u_seq: torch.Tensor,
        frame_next_seq: torch.Tensor,
        history_next_seq: torch.Tensor,
        *,
        target_encode_chunk_size: int | None = None,
        target_requires_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m = u_seq.shape[1]
        z_k = self.lift_state(frame_k, history_k)

        z_target_seq, s_target_seq = self._encode_targets(
            frame_seq=frame_next_seq,
            history_seq=history_next_seq,
            chunk_size=target_encode_chunk_size,
            requires_grad=target_requires_grad,
        )

        z_pred_list = []
        z_current = z_k
        for i in range(m):
            u_i = u_seq[:, i, :]
            z_next = self.A(z_current) + self.B(u_i)
            z_pred_list.append(z_next)
            z_current = z_next

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        s_pred_seq = z_pred_seq[..., :self.state_dim]
        return z_pred_seq, s_pred_seq, z_target_seq, s_target_seq


class ResidualConvBlock(nn.Module):
    """Pre-activation residual block with optional spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        activation_fn: Type[nn.Module] = nn.GELU,
        groups: int = 8,
    ):
        super().__init__()
        norm_groups_in = min(groups, in_channels)
        norm_groups_out = min(groups, out_channels)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_channels)
        self.act1 = activation_fn()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups_out, out_channels)
        self.act2 = activation_fn()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class ExpressiveConvEncoder(nn.Module):
    """Residual CNN backbone that extracts a high-capacity feature vector from a single frame."""

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        base_channels: int = 64,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        widths = [
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, widths[0]), widths[0]),
            activation_fn(),
        )
        self.stages = nn.Sequential(
            ResidualConvBlock(widths[0], widths[1], stride=1, activation_fn=activation_fn),
            ResidualConvBlock(widths[1], widths[2], stride=2, activation_fn=activation_fn),
            ResidualConvBlock(widths[2], widths[2], stride=1, activation_fn=activation_fn),
            ResidualConvBlock(widths[2], widths[3], stride=2, activation_fn=activation_fn),
            ResidualConvBlock(widths[3], widths[3], stride=1, activation_fn=activation_fn),
            ResidualConvBlock(widths[3], widths[4], stride=2, activation_fn=activation_fn),
            ResidualConvBlock(widths[4], widths[4], stride=1, activation_fn=activation_fn),
        )
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, widths[4]), widths[4]),
            activation_fn(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(widths[4], feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


class StateAdapter(nn.Module):
    """Maps encoder features to the compact state vector used by dynamics heads."""

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        hidden_width: int,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_width),
            activation_fn(),
            nn.Linear(hidden_width, state_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class HistoryConditionedMLP(nn.Module):
    """MLP that consumes flattened latent histories plus optional side inputs."""

    def __init__(
        self,
        history_length: int,
        state_dim: int,
        extra_input_dim: int,
        output_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.history_length = history_length
        self.state_dim = state_dim
        self.extra_input_dim = extra_input_dim
        input_dim = history_length * state_dim + extra_input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            create_mlp(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_width=hidden_width,
                depth=depth,
                activation_fn=activation_fn,
            ),
        )

    def forward(self, history: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        history_flat = history.reshape(history.shape[0], self.history_length * self.state_dim)
        if extra is not None:
            history_flat = torch.cat([history_flat, extra], dim=-1)
        return self.net(history_flat)


class SingleFrameLatentDynamicsModel(nn.Module):
    """
    Single-frame visual encoder plus latent-history inverse/forward dynamics heads.

    The encoder only processes individual frames. Temporal context is formed by stacking
    encoded state vectors over the last H timesteps.
    """

    def __init__(
        self,
        image_channels: int,
        control_dim: int,
        state_dim: int,
        history_length: int,
        encoder_feature_dim: int = 512,
        encoder_base_channels: int = 64,
        adapter_hidden_width: int = 512,
        dynamics_hidden_width: int = 512,
        dynamics_depth: int = 2,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.history_length = history_length

        self.encoder = ExpressiveConvEncoder(
            in_channels=image_channels,
            feature_dim=encoder_feature_dim,
            base_channels=encoder_base_channels,
            activation_fn=activation_fn,
        )
        self.adapter = StateAdapter(
            feature_dim=encoder_feature_dim,
            state_dim=state_dim,
            hidden_width=adapter_hidden_width,
            activation_fn=activation_fn,
        )
        self.inverse_model = HistoryConditionedMLP(
            history_length=history_length * 2,
            state_dim=state_dim,
            extra_input_dim=0,
            output_dim=control_dim,
            hidden_width=dynamics_hidden_width,
            depth=dynamics_depth,
            activation_fn=activation_fn,
        )
        self.forward_model = HistoryConditionedMLP(
            history_length=history_length,
            state_dim=state_dim,
            extra_input_dim=control_dim,
            output_dim=state_dim,
            hidden_width=dynamics_hidden_width,
            depth=dynamics_depth,
            activation_fn=activation_fn,
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.encoder(frames)
        return self.adapter(features)

    def predict_action(self, history_t: torch.Tensor, history_tp1: torch.Tensor) -> torch.Tensor:
        pair_history = torch.cat([history_t, history_tp1], dim=1)
        return self.inverse_model(pair_history)

    def predict_next_state(self, history_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        return self.forward_model(history_t, extra=action_t)
