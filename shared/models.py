from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def rotate_queries_or_keys(
    x: torch.Tensor,
    pos: torch.Tensor,
    n_registers: int,
    has_cls_first: bool,
) -> torch.Tensor:
    _, _, n_tokens, head_dim = x.size()
    if head_dim % 2 != 0:
        raise ValueError("Embedding dimension must be divisible by 2 for RoPE")

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = n_tokens - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(head_dim // 2, dtype=x.dtype, device=x.device)
    omega /= head_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        drop: float = 0.0,
        wide_silu: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        grid_size: int = 14,
        is_causal: bool = False,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)
        else:
            self.pretrained_grid_size = grid_size

    def _get_frame_pos(
        self,
        ids: torch.Tensor,
        h_patches: int | None = None,
        w_patches: int | None = None,
    ) -> torch.Tensor:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        return ids // tokens_per_frame

    def _get_height_pos(
        self,
        ids: torch.Tensor,
        h_patches: int | None = None,
        w_patches: int | None = None,
    ) -> torch.Tensor:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        tokens_per_row = w_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, h_patches, w_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(
        self,
        ids: torch.Tensor,
        h_patches: int | None = None,
        w_patches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        tokens_per_row = w_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, h_patches, w_patches)
        height_ids = self._get_height_pos(ids, h_patches, w_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        t_patches: int | None = None,
        h_patches: int | None = None,
        w_patches: int | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, n_tokens, channels = x.size()
        n_ctx = n_tokens - self.n_registers
        grid_depth = int(n_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, h_patches, w_patches)
        else:
            if t_patches is None or h_patches is None or w_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(t_patches * h_patches * w_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, h_patches, w_patches)

        if self.interpolate_rope:
            h_patches = h_patches or self.grid_size
            w_patches = w_patches or self.grid_size
            if h_patches > 1:
                h_mask = h_mask * (self.pretrained_grid_size - 1) / (h_patches - 1)
            if w_patches > 1:
                w_mask = w_mask * (self.pretrained_grid_size - 1) / (w_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        s += self.w_dim

        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.proj_drop_prob,
                is_causal=self.is_causal,
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, n_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn if return_attn else None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.num_heads, channels // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.proj_drop_prob,
                is_causal=self.is_causal,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, n_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_value: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        wide_silu: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
        use_rope: bool = False,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
            )
        self.drop_path = DropPath(drop_path_value) if drop_path_value > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        t_patches: int | None = None,
        h_patches: int | None = None,
        w_patches: int | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x),
                mask=mask,
                t_patches=t_patches,
                h_patches=h_patches,
                w_patches=w_patches,
                return_attn=return_attn,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn if return_attn else None


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        use_silu: bool = False,
        wide_silu: bool = True,
        use_sdpa: bool = True,
        is_causal: bool = False,
        use_rope: bool = False,
        init_type: str = "default",
        handle_nonsquare_inputs: bool = True,
        img_temporal_dim_size: int | None = None,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        modality_embedding: bool = True,
        n_output_distillation: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.uniform_power = uniform_power
        self.use_rope = use_rope

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.patch_embed_img = None
        if self.img_temporal_dim_size is not None:
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_value=dpr[i],
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    grid_size=img_size[0] // patch_size,
                    use_rope=use_rope,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            self.out_layers_distillation = [11] if n_output_distillation == 1 else [2, 5, 8, 11]
        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            self.out_layers_distillation = [23] if n_output_distillation == 1 else [5, 11, 17, 23]
        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            self.out_layers_distillation = [39] if n_output_distillation == 1 else [9, 19, 29, 39]
        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]
            self.out_layers_distillation = [47] if n_output_distillation == 1 else [11, 23, 37, 47]
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        self.attn_out = False
        self.init_std = init_std
        self.return_hierarchical = False

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=self.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                trunc_normal_(module.weight, std=self.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            return
        raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data if isinstance(layer.mlp, MLP) else layer.mlp.fc3.weight.data, layer_id + 1)

    def check_temporal_dim(self, shape: torch.Size) -> bool:
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if x.ndim == 4:
            _, _, h, w = x.shape
            t_patches = 1
        elif x.ndim == 5:
            _, _, t, h, w = x.shape
            t_patches = t if self.check_temporal_dim(x.shape) else t // self.tubelet_size
        else:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")

        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        if not self.handle_nonsquare_inputs:
            t_patches = h_patches = w_patches = None

        if self.check_temporal_dim(x.shape):
            if self.patch_embed_img is None:
                raise RuntimeError("Image patch embedding is not initialized")
            x = self.patch_embed_img(x)
            if self.modality_embedding:
                x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            if self.modality_embedding:
                x = x + self.video_mod_embed.repeat(x.shape[0], 1, 1)

        hier = []
        for i, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                t_patches=t_patches,
                h_patches=h_patches,
                w_patches=w_patches,
                return_attn=self.attn_out,
            )
            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)


def vit_base(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def clean_backbone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def build_vjepa2_vitb_encoder_384() -> VisionTransformer:
    return vit_base(
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        n_output_distillation=1,
        modality_embedding=True,
    )


def load_vjepa2_vitb_encoder_384(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> VisionTransformer:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if "ema_encoder" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain `ema_encoder`")

    model = build_vjepa2_vitb_encoder_384()
    state_dict = clean_backbone_state_dict(checkpoint["ema_encoder"])
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint load result: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model.to(device)
def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor
