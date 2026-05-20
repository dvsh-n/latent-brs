#!/usr/bin/env python3
"""Cache single-frame JEPA Markov states and train an unconditional DDPM on them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm.auto import tqdm


DEFAULT_DATASET_PATHS = [
    Path("reacher/data/train_data_noisy/reacher_train.h5"),
]
DEFAULT_SOURCE_MODEL_DIR = Path("reacher/models/mlpdyn_ft_7")
DEFAULT_RUN_DIR = Path("reacher/models/markov_state_ddpm")
DEFAULT_OUTPUT_MODEL_NAME = "markov_state_ddpm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, default=DEFAULT_SOURCE_MODEL_DIR)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-model-name", default=DEFAULT_OUTPUT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=3072)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--markov-deriv", type=int, default=None)
    parser.add_argument("--cache-batch-size", type=int, default=256)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--cache-prefetch-factor", type=int, default=2)

    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=10)
    return parser.parse_args()


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_source_checkpoint(args: argparse.Namespace) -> Path:
    if args.source_checkpoint is not None:
        return args.source_checkpoint.expanduser().resolve()
    return latest_object_checkpoint(args.source_model_dir.expanduser().resolve())


def load_source_config(source_model_dir: Path, checkpoint_path: Path) -> dict[str, object]:
    candidates = [
        source_model_dir.expanduser().resolve() / "config.json",
        checkpoint_path.expanduser().resolve().parent / "config.json",
    ]
    for config_path in candidates:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        "Could not find source model config.json. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def required_markov_history(markov_deriv: int) -> int:
    if markov_deriv < 0:
        raise ValueError("markov_deriv must be non-negative.")
    return markov_deriv + 1


def build_markov_state(history_emb: torch.Tensor, markov_deriv: int) -> torch.Tensor:
    squeeze = False
    if history_emb.ndim == 2:
        history_emb = history_emb.unsqueeze(0)
        squeeze = True
    context_len = required_markov_history(markov_deriv)
    if history_emb.ndim != 3 or history_emb.shape[1] < context_len:
        raise ValueError(
            f"Expected history_emb with shape [batch, >= {context_len}, dim], got {tuple(history_emb.shape)}."
        )
    deriv_seq = history_emb[:, -context_len:]
    components = [deriv_seq[:, -1]]
    for _ in range(markov_deriv):
        deriv_seq = deriv_seq[:, 1:] - deriv_seq[:, :-1]
        components.append(deriv_seq[:, -1])
    state = torch.cat(components, dim=-1)
    return state[0] if squeeze else state


class PixelFrameDataset(Dataset):
    def __init__(self, dataset_path: Path, img_size: int) -> None:
        self.dataset_path = dataset_path
        self.img_size = int(img_size)
        self._h5: h5py.File | None = None
        with h5py.File(self.dataset_path, "r") as h5:
            self.length = int(h5["pixels"].shape[0])

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> torch.Tensor:
        h5 = self._file()
        pixel = torch.from_numpy(np.asarray(h5["pixels"][index], dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)
        if tuple(pixel.shape[-2:]) != (self.img_size, self.img_size):
            pixel = F.interpolate(
                pixel.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (pixel - mean) / std

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
            self._h5 = None


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0 and persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def sanitize_hparams(args: argparse.Namespace) -> dict[str, object]:
    hparams = vars(args).copy()
    for key, value in hparams.items():
        if isinstance(value, Path):
            hparams[key] = str(value)
        elif isinstance(value, list):
            hparams[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return hparams


def dataset_cache_key(dataset_path: Path, checkpoint_path: Path, img_size: int, markov_deriv: int) -> str:
    dataset_stat = dataset_path.stat()
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "dataset_path": str(dataset_path),
        "dataset_size": int(dataset_stat.st_size),
        "dataset_mtime_ns": int(dataset_stat.st_mtime_ns),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "img_size": int(img_size),
        "markov_deriv": int(markov_deriv),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", dataset_path.stem).strip("_").lower() or "dataset"
    return f"{stem}_{digest[:12]}"


@torch.no_grad()
def encode_dataset_to_cache(
    *,
    model: torch.nn.Module,
    dataset_path: Path,
    cache_path: Path,
    checkpoint_path: Path,
    img_size: int,
    markov_deriv: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> dict[str, object]:
    dataset = PixelFrameDataset(dataset_path, img_size)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        pin_memory=device.type == "cuda",
    )
    latents = []
    for pixels in tqdm(
        loader,
        desc=f"Encoding {dataset_path.name}",
        total=len(loader),
        leave=False,
    ):
        pixels = pixels.to(device, non_blocking=True)
        output = model.encoder(pixels, interpolate_pos_encoding=True)
        batch_latents = model.projector(output.last_hidden_state[:, 0]).detach().cpu()
        latents.append(batch_latents)

    all_latents = torch.cat(latents, dim=0).contiguous()
    context_len = required_markov_history(markov_deriv)
    markov_states = []
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    for episode_len, episode_offset in tqdm(
        zip(ep_len.tolist(), ep_offset.tolist()),
        desc=f"Markov states {dataset_path.name}",
        total=len(ep_len),
        leave=False,
    ):
        start = int(episode_offset)
        stop = start + int(episode_len)
        episode_latents = all_latents[start:stop]
        if episode_latents.numel() == 0:
            continue
        history_states = []
        for step in range(int(episode_len)):
            history = []
            for prev in range(markov_deriv, 0, -1):
                hist_step = max(step - prev, 0)
                history.append(episode_latents[hist_step])
            history.append(episode_latents[step])
            history_states.append(build_markov_state(torch.stack(history, dim=0), markov_deriv))
        markov_states.append(torch.stack(history_states, dim=0))
    all_markov_states = torch.cat(markov_states, dim=0).contiguous()
    payload = {
        "states": all_markov_states,
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "img_size": int(img_size),
        "markov_deriv": int(markov_deriv),
        "num_samples": int(all_markov_states.shape[0]),
        "state_dim": int(all_markov_states.shape[1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return {
        "cache_path": str(cache_path),
        "dataset_path": str(dataset_path),
        "num_samples": int(all_markov_states.shape[0]),
        "state_dim": int(all_markov_states.shape[1]),
        "cache_hit": False,
    }


def load_or_create_state_cache(
    *,
    model: torch.nn.Module,
    dataset_paths: list[Path],
    checkpoint_path: Path,
    cache_dir: Path,
    img_size: int,
    markov_deriv: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    refresh_cache: bool,
) -> tuple[list[torch.Tensor], list[dict[str, object]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "source_checkpoint.txt").write_text(str(checkpoint_path), encoding="utf-8")

    state_tensors: list[torch.Tensor] = []
    manifest: list[dict[str, object]] = []
    for dataset_path in dataset_paths:
        cache_key = dataset_cache_key(dataset_path, checkpoint_path, img_size, markov_deriv)
        cache_path = cache_dir / f"{cache_key}.pt"
        if cache_path.is_file() and not refresh_cache:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            states = payload["states"].float().contiguous()
            state_tensors.append(states)
            manifest.append(
                {
                    "cache_path": str(cache_path),
                    "dataset_path": str(dataset_path),
                    "num_samples": int(states.shape[0]),
                    "state_dim": int(states.shape[1]),
                    "cache_hit": True,
                }
            )
            continue

        info = encode_dataset_to_cache(
            model=model,
            dataset_path=dataset_path,
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            img_size=img_size,
            markov_deriv=markov_deriv,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            device=device,
        )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        state_tensors.append(payload["states"].float().contiguous())
        manifest.append(info)
    return state_tensors, manifest


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat((args.sin(), args.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_embed_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = h + self.time_proj(t_emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class LatentStateDDPM(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        time_embed_dim: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.time_embed_dim = int(time_embed_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("latent_mean", latent_mean.float().view(1, -1))
        self.register_buffer("latent_std", latent_std.float().view(1, -1))

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, time_embed_dim, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.latent_mean) / self.latent_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.latent_std + self.latent_mean

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)
        t_emb = self.time_mlp(self.time_embed(timesteps))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(self.output_norm(F.silu(h)))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale_x = self.sqrt_alpha_bars[timesteps].unsqueeze(-1)
        scale_n = self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(-1)
        return scale_x * x0 + scale_n * noise

    def score(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        eps = self.forward(self.normalize(x_t), timesteps)
        denom = self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(-1).clamp_min(1e-6)
        return -(eps / denom) / self.latent_std


class LatentStateDDPMTrainer(L.LightningModule):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        time_embed_dim: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        lr: float,
        weight_decay: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["latent_mean", "latent_std"])
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.model = LatentStateDDPM(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            time_embed_dim=time_embed_dim,
            diffusion_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            latent_mean=latent_mean,
            latent_std=latent_std,
            dropout=dropout,
        )

    def _step(self, batch: tuple[torch.Tensor], stage: str) -> torch.Tensor:
        x0 = self.model.normalize(batch[0].float())
        batch_size = x0.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.model.diffusion_steps,
            size=(batch_size,),
            device=x0.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, timesteps, noise)
        pred_noise = self.model(x_t, timesteps)
        loss = F.mse_loss(pred_noise, noise)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class ModelObjectCallback(Callback):
    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, self.dirpath / f"{self.filename}_epoch_{epoch}_object.pt")


def build_train_and_val_sets(states: torch.Tensor, train_split: float, seed: int) -> tuple[Dataset, Dataset | None]:
    dataset = TensorDataset(states)
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 Markov states for train/val split, got {len(dataset)}.")
    train_len = int(len(dataset) * train_split)
    train_len = min(max(train_len, 1), len(dataset) - 1)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    return train_set, val_set


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    dataset_paths = [dataset_path.expanduser().resolve() for dataset_path in args.dataset_path]
    missing_dataset_paths = [dataset_path for dataset_path in dataset_paths if not dataset_path.is_file()]
    if missing_dataset_paths:
        missing_list = ", ".join(str(dataset_path) for dataset_path in missing_dataset_paths)
        raise FileNotFoundError(f"Dataset file not found: {missing_list}")

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    cache_dir = (args.cache_dir.expanduser().resolve() if args.cache_dir is not None else run_dir.parent / "markov_state_ddpm_cache")
    checkpoint_path = resolve_source_checkpoint(args)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")
    source_config = load_source_config(args.source_model_dir, checkpoint_path)
    markov_deriv = int(args.markov_deriv) if args.markov_deriv is not None else int(source_config["markov_deriv"])

    device_arg = args.accelerator
    if device_arg == "gpu" and not torch.cuda.is_available():
        encode_device = torch.device("cpu")
    else:
        encode_device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "gpu" else "cpu")

    source_model = torch.load(checkpoint_path, map_location=encode_device, weights_only=False)
    source_model = source_model.to(encode_device)
    source_model.eval()
    source_model.requires_grad_(False)

    state_tensors, cache_manifest = load_or_create_state_cache(
        model=source_model,
        dataset_paths=dataset_paths,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        img_size=args.img_size,
        markov_deriv=markov_deriv,
        batch_size=args.cache_batch_size,
        num_workers=args.cache_num_workers,
        prefetch_factor=args.cache_prefetch_factor,
        device=encode_device,
        refresh_cache=args.refresh_cache,
    )
    del source_model
    if encode_device.type == "cuda":
        torch.cuda.empty_cache()

    states = torch.cat(state_tensors, dim=0).float().contiguous()
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0).clamp_min(1e-6)

    train_set, val_set = build_train_and_val_sets(states, args.train_split, args.seed)
    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.accelerator == "gpu",
    )
    val_loader = make_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=False,
        pin_memory=args.accelerator == "gpu",
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    config = sanitize_hparams(args)
    config["dataset_path"] = [str(path) for path in dataset_paths]
    config["source_checkpoint"] = str(checkpoint_path)
    config["run_dir"] = str(run_dir)
    config["cache_dir"] = str(cache_dir)
    config["markov_deriv"] = markov_deriv
    config["state_dim"] = int(states.shape[1])
    config["num_states"] = int(states.shape[0])
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (run_dir / "cache_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(cache_manifest, handle, indent=2)

    module = LatentStateDDPMTrainer(
        latent_dim=int(states.shape[1]),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_embed_dim=args.time_embed_dim,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        latent_mean=state_mean,
        latent_std=state_std,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename=f"{args.output_model_name}" + "_{epoch:03d}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        ModelObjectCallback(run_dir, args.output_model_name, epoch_interval=args.save_object_every),
    ]
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=run_dir,
        logger=False,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
    )
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(module.model, run_dir / f"{args.output_model_name}_final_object.pt")


if __name__ == "__main__":
    main()
