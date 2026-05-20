#!/usr/bin/env python3
"""Train a small MLP that maps one JEPA embedding space into another."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm.auto import tqdm


DEFAULT_DATASET_PATHS = [
    Path("reacher/data/train_data_noisy/reacher_train.h5"),
]
DEFAULT_SOURCE_MODEL_DIR = Path("reacher/models/mlpdyn_ft_5")
DEFAULT_TARGET_MODEL_DIR = Path("reacher/models/mlpdyn_ft_7")
DEFAULT_RUN_DIR = Path("reacher/models/latent_translate_5_to_7")
DEFAULT_OUTPUT_MODEL_NAME = "latent_translate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, default=DEFAULT_SOURCE_MODEL_DIR)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--target-model-dir", type=Path, default=DEFAULT_TARGET_MODEL_DIR)
    parser.add_argument("--target-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-model-name", default=DEFAULT_OUTPUT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=3072)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--cache-batch-size", type=int, default=256)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--cache-prefetch-factor", type=int, default=2)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=2048)
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


def resolve_checkpoint(model_dir: Path, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is None:
        checkpoint_path = latest_object_checkpoint(model_dir.expanduser().resolve())
    else:
        checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_model_config(model_dir: Path, checkpoint_path: Path) -> dict[str, object]:
    candidates = [
        model_dir.expanduser().resolve() / "config.json",
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


def maybe_add_safe_globals() -> None:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])


def dataset_cache_key(
    dataset_paths: list[Path],
    source_checkpoint: Path,
    target_checkpoint: Path,
    img_size: int,
) -> str:
    payload = {
        "dataset_paths": [],
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "img_size": int(img_size),
    }
    for dataset_path in dataset_paths:
        stat = dataset_path.stat()
        payload["dataset_paths"].append(
            {
                "path": str(dataset_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    for key, path in [("source_checkpoint", source_checkpoint), ("target_checkpoint", target_checkpoint)]:
        stat = path.stat()
        payload[f"{key}_size"] = int(stat.st_size)
        payload[f"{key}_mtime_ns"] = int(stat.st_mtime_ns)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def load_jepa_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    maybe_add_safe_globals()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_projected_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    for pixels in tqdm(loader, desc=desc):
        pixels = pixels.to(device, non_blocking=True)
        output = model.encoder(pixels, interpolate_pos_encoding=True)
        batch_latents = model.projector(output.last_hidden_state[:, 0]).detach().cpu()
        latents.append(batch_latents)
    return torch.cat(latents, dim=0).float().contiguous()


def build_latent_pair_cache(
    dataset_paths: list[Path],
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    img_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    src_parts: list[torch.Tensor] = []
    tgt_parts: list[torch.Tensor] = []
    manifest: list[dict[str, object]] = []

    for dataset_path in dataset_paths:
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
        src = encode_projected_embeddings(
            source_model,
            loader,
            device=device,
            desc=f"Encoding source {dataset_path.name}",
        )
        tgt = encode_projected_embeddings(
            target_model,
            loader,
            device=device,
            desc=f"Encoding target {dataset_path.name}",
        )
        src_parts.append(src)
        tgt_parts.append(tgt)
        manifest.append(
            {
                "dataset_path": str(dataset_path),
                "num_frames": int(len(dataset)),
                "source_dim": int(src.shape[1]),
                "target_dim": int(tgt.shape[1]),
            }
        )

    return (
        torch.cat(src_parts, dim=0).contiguous(),
        torch.cat(tgt_parts, dim=0).contiguous(),
        manifest,
    )


def load_or_create_latent_pairs(
    *,
    dataset_paths: list[Path],
    source_checkpoint: Path,
    target_checkpoint: Path,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    img_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    cache_dir: Path,
    refresh_cache: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Path, list[dict[str, object]]]:
    cache_key = dataset_cache_key(dataset_paths, source_checkpoint, target_checkpoint, img_size)
    cache_path = cache_dir / f"latent_pairs_{cache_key}.pt"
    if cache_path.is_file() and not refresh_cache:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["source_latents"], payload["target_latents"], cache_path, payload["manifest"]

    source_latents, target_latents, manifest = build_latent_pair_cache(
        dataset_paths,
        source_model=source_model,
        target_model=target_model,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_latents": source_latents,
            "target_latents": target_latents,
            "manifest": manifest,
            "source_checkpoint": str(source_checkpoint),
            "target_checkpoint": str(target_checkpoint),
        },
        cache_path,
    )
    return source_latents, target_latents, cache_path, manifest


class LatentTranslator(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        source_mean: torch.Tensor,
        source_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        layers: list[nn.Module] = []
        current_dim = int(source_dim)
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, int(target_dim)))
        self.net = nn.Sequential(*layers)

        self.register_buffer("source_mean", source_mean.float().view(1, -1))
        self.register_buffer("source_std", source_std.float().view(1, -1).clamp_min(1e-6))
        self.register_buffer("target_mean", target_mean.float().view(1, -1))
        self.register_buffer("target_std", target_std.float().view(1, -1).clamp_min(1e-6))

    def forward_normalized(self, source_latent: torch.Tensor) -> torch.Tensor:
        normalized = (source_latent - self.source_mean) / self.source_std
        return self.net(normalized)

    def forward(self, source_latent: torch.Tensor) -> torch.Tensor:
        pred = self.forward_normalized(source_latent)
        return pred * self.target_std + self.target_mean


def normalized_loss(model: LatentTranslator, source_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    pred_norm = model.forward_normalized(source_latent)
    target_norm = (target_latent - model.target_mean) / model.target_std
    return F.mse_loss(pred_norm, target_norm)


@torch.no_grad()
def evaluate(
    model: LatentTranslator,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_norm_loss = 0.0
    total_raw_mse = 0.0
    total_count = 0
    for source_latent, target_latent in loader:
        source_latent = source_latent.to(device, non_blocking=True)
        target_latent = target_latent.to(device, non_blocking=True)
        batch_size = source_latent.shape[0]
        loss = normalized_loss(model, source_latent, target_latent)
        pred = model(source_latent)
        raw_mse = F.mse_loss(pred, target_latent)
        total_norm_loss += float(loss.item()) * batch_size
        total_raw_mse += float(raw_mse.item()) * batch_size
        total_count += batch_size
    return total_norm_loss / total_count, total_raw_mse / total_count


def train_epoch(
    model: LatentTranslator,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_val: float,
    precision: str,
) -> float:
    model.train()
    use_autocast = device.type == "cuda" and precision in {"16-mixed", "bf16-mixed"}
    autocast_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "16-mixed")
    total_loss = 0.0
    total_count = 0

    for source_latent, target_latent in tqdm(loader, desc="Training", leave=False):
        source_latent = source_latent.to(device, non_blocking=True)
        target_latent = target_latent.to(device, non_blocking=True)
        batch_size = source_latent.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = normalized_loss(model, source_latent, target_latent)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / total_count


def save_training_state(
    path: Path,
    *,
    model: LatentTranslator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_raw_mse: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_raw_mse": float(val_raw_mse),
        },
        path,
    )


def build_train_and_val_sets(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    train_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    dataset = TensorDataset(source_latents, target_latents)
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 latent pairs for train/val splitting, got {len(dataset)}.")
    train_len = int(len(dataset) * train_split)
    val_len = len(dataset) - train_len
    if train_len < 1:
        train_len = 1
        val_len = len(dataset) - 1
    if val_len < 1:
        val_len = 1
        train_len = len(dataset) - 1
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def require_device(device_arg: str) -> torch.device:
    if device_arg in {"auto", "gpu"}:
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_paths = [path.expanduser().resolve() for path in args.dataset_path]
    missing_dataset_paths = [path for path in dataset_paths if not path.is_file()]
    if missing_dataset_paths:
        missing_list = ", ".join(str(path) for path in missing_dataset_paths)
        raise FileNotFoundError(f"Dataset file not found: {missing_list}")

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else (run_dir.parent / f"{run_dir.name}_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_model_dir = args.source_model_dir.expanduser().resolve()
    target_model_dir = args.target_model_dir.expanduser().resolve()
    source_checkpoint = resolve_checkpoint(source_model_dir, args.source_checkpoint)
    target_checkpoint = resolve_checkpoint(target_model_dir, args.target_checkpoint)
    source_config = load_model_config(source_model_dir, source_checkpoint)
    target_config = load_model_config(target_model_dir, target_checkpoint)

    device = require_device(args.accelerator)
    source_model = load_jepa_model(source_checkpoint, device)
    target_model = load_jepa_model(target_checkpoint, device)

    source_latents, target_latents, cache_path, cache_manifest = load_or_create_latent_pairs(
        dataset_paths=dataset_paths,
        source_checkpoint=source_checkpoint,
        target_checkpoint=target_checkpoint,
        source_model=source_model,
        target_model=target_model,
        img_size=args.img_size,
        batch_size=args.cache_batch_size,
        num_workers=args.cache_num_workers,
        prefetch_factor=args.cache_prefetch_factor,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
        device=device,
    )

    if source_latents.shape[0] != target_latents.shape[0]:
        raise RuntimeError("Source and target latent counts do not match.")

    source_mean = source_latents.mean(dim=0)
    source_std = source_latents.std(dim=0).clamp_min(1e-6)
    target_mean = target_latents.mean(dim=0)
    target_std = target_latents.std(dim=0).clamp_min(1e-6)

    train_set, val_set = build_train_and_val_sets(source_latents, target_latents, args.train_split, args.seed)
    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=False,
        pin_memory=device.type == "cuda",
    )

    translator = LatentTranslator(
        source_dim=int(source_latents.shape[1]),
        target_dim=int(target_latents.shape[1]),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    translator = translator.to(device)
    optimizer = torch.optim.AdamW(translator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir.mkdir(parents=True, exist_ok=False)
    config = sanitize_hparams(args)
    config["dataset_path"] = [str(path) for path in dataset_paths]
    config["run_dir"] = str(run_dir)
    config["cache_dir"] = str(cache_dir)
    config["cache_path"] = str(cache_path)
    config["source_checkpoint"] = str(source_checkpoint)
    config["target_checkpoint"] = str(target_checkpoint)
    config["source_model_dir"] = str(source_model_dir)
    config["target_model_dir"] = str(target_model_dir)
    config["source_embed_dim"] = int(source_latents.shape[1])
    config["target_embed_dim"] = int(target_latents.shape[1])
    config["num_pairs"] = int(source_latents.shape[0])
    config["source_model_config"] = source_config
    config["target_model_config"] = target_config
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (run_dir / "cache_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(cache_manifest, handle, indent=2)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            translator,
            train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_val=args.gradient_clip_val,
            precision=args.precision,
        )
        val_loss, val_raw_mse = evaluate(translator, val_loader, device=device)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_raw_mse={val_raw_mse:.6f}",
            flush=True,
        )

        save_training_state(
            run_dir / "last.ckpt",
            model=translator,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_raw_mse=val_raw_mse,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_training_state(
                run_dir / f"{args.output_model_name}_best.ckpt",
                model=translator,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_raw_mse=val_raw_mse,
            )
        if epoch % args.save_object_every == 0 or epoch == args.epochs:
            torch.save(translator, run_dir / f"{args.output_model_name}_epoch_{epoch}_object.pt")

    torch.save(translator, run_dir / f"{args.output_model_name}_final_object.pt")


if __name__ == "__main__":
    main()
