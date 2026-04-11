from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
DINOV2_REPO = REPO_ROOT / "third_party" / "dinov2"
if str(DINOV2_REPO) not in sys.path:
    sys.path.insert(0, str(DINOV2_REPO))

import hubconf  # noqa: E402


DINO_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2_vits14": {
        "hub_fn": "dinov2_vits14",
        "weights": REPO_ROOT / "models" / "dinov2_vits14_pretrain.pth",
        "embed_dim": 384,
        "patch_size": 14,
    },
    "dinov2_vits14_reg": {
        "hub_fn": "dinov2_vits14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vits14_reg4_pretrain.pth",
        "embed_dim": 384,
        "patch_size": 14,
    },
    "dinov2_vitb14": {
        "hub_fn": "dinov2_vitb14",
        "weights": REPO_ROOT / "models" / "dinov2_vitb14_pretrain.pth",
        "embed_dim": 768,
        "patch_size": 14,
    },
    "dinov2_vitb14_reg": {
        "hub_fn": "dinov2_vitb14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vitb14_reg4_pretrain.pth",
        "embed_dim": 768,
        "patch_size": 14,
    },
    "dinov2_vitl14": {
        "hub_fn": "dinov2_vitl14",
        "weights": REPO_ROOT / "models" / "dinov2_vitl14_pretrain.pth",
        "embed_dim": 1024,
        "patch_size": 14,
    },
    "dinov2_vitl14_reg": {
        "hub_fn": "dinov2_vitl14_reg",
        "weights": REPO_ROOT / "models" / "dinov2_vitl14_reg4_pretrain.pth",
        "embed_dim": 1024,
        "patch_size": 14,
    },
}


class GlobalProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 384,
        hidden: int = 384,
        pool_hw: int = 1,
        gn_groups: int = 8,
        conv_layers: list[dict[str, int]] | None = None,
    ) -> None:
        super().__init__()
        self.pool_hw = int(pool_hw)
        self.mix = nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(num_groups=max(1, min(gn_groups, hidden)), num_channels=hidden)

        blocks: list[nn.Module] = []
        conv_layers = conv_layers or []
        for cfg in conv_layers:
            out_ch = int(cfg["out_dim"])
            blocks.append(
                nn.Conv2d(
                    int(cfg["in_dim"]),
                    out_ch,
                    kernel_size=int(cfg["kernel_size"]),
                    stride=int(cfg["stride"]),
                    padding=int(cfg["padding"]),
                    bias=False,
                )
            )
            blocks.append(
                nn.GroupNorm(
                    num_groups=max(1, min(gn_groups, out_ch)),
                    num_channels=out_ch,
                )
            )
            blocks.append(nn.GELU())
        last_dim = int(conv_layers[-1]["out_dim"]) if conv_layers else hidden
        self.down_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.head = nn.Conv2d(last_dim, out_dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.gn0(self.mix(x)))
        x = self.down_blocks(x)
        x = self.pool(x)
        x = self.head(x)
        if self.pool_hw == 1:
            return self.ln(x.flatten(1))
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.ln(x)


class ChannelProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 384,
        norm_type: str = "layer",
        conv_layers: list[dict[str, int]] | None = None,
    ) -> None:
        super().__init__()
        del in_dim, out_dim
        self.norm_type = norm_type
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.layer_norm_layers = nn.ModuleList()

        for cfg in (conv_layers or []):
            out_ch = int(cfg["out_dim"])
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=int(cfg["in_dim"]),
                    out_channels=out_ch,
                    kernel_size=int(cfg["kernel_size"]),
                    stride=int(cfg["stride"]),
                    padding=int(cfg["padding"]),
                )
            )
            self.batch_norm_layers.append(nn.BatchNorm2d(out_ch))
            self.layer_norm_layers.append(nn.LayerNorm(out_ch))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.norm_type == "batch":
                x = self.batch_norm_layers[idx](x)
            elif self.norm_type == "layer":
                x = x.permute(0, 2, 3, 1)
                x = self.layer_norm_layers[idx](x)
                x = x.permute(0, 3, 1, 2)
            if len(self.conv_layers) > 1 and idx != len(self.conv_layers) - 1:
                x = self.activation(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class LocalDinoV2Encoder(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vits14",
        feature_key: str = "x_norm_patchtokens",
        projector: str = "none",
        projector_kwargs: dict[str, Any] | None = None,
        agg_type: str = "flatten",
        agg_out_dim: int | None = None,
        agg_mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if name not in DINO_MODEL_CONFIGS:
            raise ValueError(f"Unsupported DINO model '{name}'.")
        cfg = DINO_MODEL_CONFIGS[name]
        weights_path = Path(cfg["weights"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing DINOv2 weights: {weights_path}")

        model = getattr(hubconf, cfg["hub_fn"])(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        self.base_model = model
        self.name = name
        self.feature_key = feature_key
        self.projector_name = projector
        self.emb_dim = int(cfg["embed_dim"])
        self.patch_size = int(cfg["patch_size"])
        self.agg_type = agg_type
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim

        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
            if projector == "global":
                self.projector = GlobalProjector(**(projector_kwargs or {}))
                self.emb_dim = self.projector.head.out_channels
            elif projector == "channel":
                self.projector = ChannelProjector(**(projector_kwargs or {}))
                if len(self.projector.conv_layers) > 0:
                    self.emb_dim = self.projector.conv_layers[-1].out_channels
            elif projector not in {"none", None}:
                raise ValueError(f"Unsupported projector '{projector}'.")
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature_key '{feature_key}'.")

        if self.agg_type == "mlp":
            self._agg_out_dim = int(self.agg_out_dim) if self.agg_out_dim is not None else self.emb_dim
            hidden_dim = (
                int(self.agg_mlp_hidden_dim)
                if self.agg_mlp_hidden_dim is not None
                else 4 * self._agg_out_dim
            )
            # 196 is the fixed patch count after 196x196 DINO resize with patch size 14.
            self.agg_mlp = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self._agg_out_dim),
            )
            self.agg_post_norm = nn.LayerNorm(self._agg_out_dim)

    def freeze_backbone(self) -> None:
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def agg(self, x: torch.Tensor) -> torch.Tensor:
        if self.agg_type == "mean":
            return x.mean(dim=1)
        x = x.contiguous().view(x.shape[0], -1)
        if self.agg_type == "flatten":
            return x
        if self.agg_type == "mlp":
            return self.agg_post_norm(self.agg_mlp(x))
        raise ValueError(f"Unsupported agg_type '{self.agg_type}'.")

    def forward(self, x: torch.Tensor, return_agg: bool = False) -> torch.Tensor:
        with torch.no_grad():
            emb = self.base_model.forward_features(x)[self.feature_key]
        if hasattr(self, "projector"):
            bsz, n_tokens, n_dim = emb.shape
            side = int(math.sqrt(n_tokens))
            if side * side != n_tokens:
                raise ValueError(f"Expected square patch grid, got {n_tokens} tokens.")
            emb = emb.view(bsz, side, side, n_dim).permute(0, 3, 1, 2).contiguous()
            if self.projector_name == "channel":
                emb = self.projector(emb)
                self.latent_ndim = 2
            elif self.projector_name == "global":
                emb = self.projector(emb)
                self.latent_ndim = 2 if emb.dim() == 3 else 1
        if return_agg and emb.dim() == 3:
            emb = self.agg(emb)
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
        return emb


class ProprioActionEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_mask_matrix(npatch: int, nwindow: int) -> torch.Tensor:
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for idx in range(nwindow):
        rows.append(torch.cat([ones] * (idx + 1) + [zeros] * (nwindow - idx - 1), dim=1))
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_patches: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("bias", generate_mask_matrix(num_patches, num_frames), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_tokens, _ = x.size()
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3).contiguous()
            for t in qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.bias[:, :, :n_tokens, :n_tokens] == 0, float("-inf"))
        attn = self.dropout(torch.softmax(dots, dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_patches: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            num_frames=num_frames,
                            num_patches=num_patches,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim=dim,
            num_frames=num_frames,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(x)
        return self.transformer(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel: int, channel: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        channel: int,
        n_res_block: int,
        n_res_channel: int,
        stride: int,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                ]
            )
        else:
            raise ValueError(f"Unsupported stride {stride}.")
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ProjectorDecoder(nn.Module):
    def __init__(self, projector_conv_layers: list[dict[str, int]]) -> None:
        super().__init__()
        self.deconv_layers = nn.ModuleList()
        for cfg in reversed(projector_conv_layers):
            self.deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=int(cfg["out_dim"]),
                    out_channels=int(cfg["in_dim"]),
                    kernel_size=int(cfg["kernel_size"]),
                    stride=int(cfg["stride"]),
                    padding=int(cfg["padding"]),
                    output_padding=int(cfg.get("output_padding", 0)),
                )
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if idx != len(self.deconv_layers) - 1:
                x = self.activation(x)
        return x


class VQVAEDecoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        channel: int = 384,
        n_res_block: int = 4,
        n_res_channel: int = 128,
        projector_conv_layers: list[dict[str, int]] | None = None,
    ) -> None:
        super().__init__()
        decoder_emb_dim = emb_dim
        self.proj_decoder = None
        if projector_conv_layers:
            decoder_emb_dim = int(projector_conv_layers[0]["in_dim"])
            self.proj_decoder = ProjectorDecoder(projector_conv_layers)
        self.dec = Decoder(
            in_channel=decoder_emb_dim,
            out_channel=3,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=4,
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_patches = tokens.shape[2]
        num_side_patches = int(math.sqrt(num_patches))
        if num_side_patches * num_side_patches != num_patches:
            raise ValueError(f"Decoder expects square tokens, got {num_patches}.")
        x = (
            tokens.view(tokens.shape[0], tokens.shape[1], num_side_patches, num_side_patches, tokens.shape[3])
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .view(tokens.shape[0] * tokens.shape[1], tokens.shape[3], num_side_patches, num_side_patches)
        )
        if self.proj_decoder is not None:
            x = self.proj_decoder(x)
        if x.shape[-2:] != (14, 14):
            x = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        return self.dec(x), x.new_zeros(1)


class StraighteningWorldModel(nn.Module):
    def __init__(
        self,
        encoder: LocalDinoV2Encoder,
        action_encoder: ProprioActionEncoder,
        proprio_encoder: ProprioActionEncoder,
        predictor: ViTPredictor,
        image_size: int = 224,
        num_hist: int = 3,
        num_pred: int = 1,
        concat_dim: int = 1,
        num_action_repeat: int = 1,
        num_proprio_repeat: int = 1,
        stop_grad: bool = True,
        straighten: str | bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        self.predictor = predictor
        self.image_size = image_size
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.concat_dim = concat_dim
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.stop_grad = bool(stop_grad)
        self.emb_criterion = nn.MSELoss()
        self.straighten = False
        self.curvature_mode: str | None = None
        self.straighten_scale = 0.0
        if isinstance(straighten, str):
            if straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "")
                self.curvature_mode = "aggcos"
                self.straighten_scale = float(suffix) if suffix else 1.0
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "")
                self.curvature_mode = "cos"
                self.straighten_scale = float(suffix) if suffix else 1.0
        self.straighten = self.curvature_mode is not None and self.straighten_scale > 0

        decoder_scale = 16
        num_side_patches = image_size // decoder_scale
        self.encoder_image_size = num_side_patches * encoder.patch_size

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = obs["visual"]
        batch_size = visual.shape[0]
        visual = visual.view(visual.shape[0] * visual.shape[1], visual.shape[2], visual.shape[3], visual.shape[4])
        if visual.shape[-1] != self.encoder_image_size or visual.shape[-2] != self.encoder_image_size:
            visual = F.interpolate(
                visual,
                size=(self.encoder_image_size, self.encoder_image_size),
                mode="bilinear",
                align_corners=False,
            )
        visual_emb = self.encoder(visual)
        visual_emb = visual_emb.view(batch_size, -1, visual_emb.shape[1], visual_emb.shape[2])
        proprio_emb = self.proprio_encoder(obs["proprio"])
        return {"visual": visual_emb, "proprio": proprio_emb}

    def encode(self, obs: dict[str, torch.Tensor], act: torch.Tensor) -> torch.Tensor:
        z_dct = self.encode_obs(obs)
        act_emb = self.action_encoder(act)
        if self.concat_dim != 1:
            raise ValueError("Only concat_dim=1 is implemented for this trainer.")
        token_count = z_dct["visual"].shape[2]
        proprio_tiled = (
            z_dct["proprio"].unsqueeze(2).expand(-1, -1, token_count, -1).repeat(1, 1, 1, self.num_proprio_repeat)
        )
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, token_count, -1).repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([z_dct["visual"], proprio_tiled, act_tiled], dim=3)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        frames = z.shape[1]
        z = z.view(z.shape[0], z.shape[1] * z.shape[2], z.shape[3])
        z = self.predictor(z)
        return z.view(z.shape[0], frames, -1, z.shape[-1])

    def separate_emb(self, z: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        proprio_dim = self.proprio_encoder.emb_dim * self.num_proprio_repeat
        action_dim = self.action_encoder.emb_dim * self.num_action_repeat
        z_visual = z[..., : -(proprio_dim + action_dim)]
        z_proprio = z[..., -(proprio_dim + action_dim) : -action_dim]
        z_act = z[..., -action_dim:]
        z_proprio = z_proprio[:, :, 0, : proprio_dim // self.num_proprio_repeat]
        z_act = z_act[:, :, 0, : action_dim // self.num_action_repeat]
        return {"visual": z_visual, "proprio": z_proprio}, z_act

    def _cos_curvature(self, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        return (1.0 - cos).mean()

    def total_curvature(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] < 3:
            return features.new_zeros(())
        if self.curvature_mode == "aggcos":
            tokens = features.reshape(features.shape[0] * features.shape[1], features.shape[2], features.shape[3])
            z = self.encoder.agg(tokens).reshape(features.shape[0], features.shape[1], -1)
            v1 = z[:, 1:-1] - z[:, :-2]
            v2 = z[:, 2:] - z[:, 1:-1]
            return self._cos_curvature(v1, v2)
        v1 = features[:, 1:-1] - features[:, :-2]
        v2 = features[:, 2:] - features[:, 1:-1]
        return self._cos_curvature(v1, v2)

    def forward(self, obs: dict[str, torch.Tensor], act: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = obs["visual"].new_zeros(())
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist]
        z_tgt = z[:, self.num_pred :]
        z_pred = self.predict(z_src)

        action_dim = self.action_encoder.emb_dim * self.num_action_repeat
        proprio_dim = self.proprio_encoder.emb_dim * self.num_proprio_repeat
        z_tgt_for_loss = z_tgt.detach() if self.stop_grad else z_tgt
        z_visual_loss = self.emb_criterion(
            z_pred[:, :, :, : -(proprio_dim + action_dim)],
            z_tgt_for_loss[:, :, :, : -(proprio_dim + action_dim)],
        )
        z_proprio_loss = self.emb_criterion(
            z_pred[:, :, :, -(proprio_dim + action_dim) : -action_dim],
            z_tgt_for_loss[:, :, :, -(proprio_dim + action_dim) : -action_dim],
        )
        z_loss = self.emb_criterion(z_pred[:, :, :, :-action_dim], z_tgt_for_loss[:, :, :, :-action_dim])
        loss = loss + z_loss

        outputs: dict[str, torch.Tensor] = {
            "loss": loss,
            "z_loss": z_loss.detach(),
            "z_visual_loss": z_visual_loss.detach(),
            "z_proprio_loss": z_proprio_loss.detach(),
        }

        if self.straighten:
            curvature_loss = self.total_curvature(z[:, :, :, : -(proprio_dim + action_dim)])
            loss = loss + self.straighten_scale * curvature_loss
            outputs["loss"] = loss
            outputs["curvature_loss"] = curvature_loss.detach()

        outputs["z_pred"] = z_pred
        return outputs
