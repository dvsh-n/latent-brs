#!/usr/bin/env python3
"""Finetune a real-rope LE-WM-style JEPA on measured task-space actions."""

from __future__ import annotations

import argparse
import glob
import json
import os
from functools import partial
from pathlib import Path, PosixPath
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split

from rope.shared.models import JEPA, MLP, MLPDynamicsPredictor, SIGReg


DEFAULT_DATASET_GLOB = "rope/data/real_data/*.h5"
DEFAULT_INIT_RUN_DIR = "rope/models/mlpdyn_ft_real"
DEFAULT_RUN_DIR = "rope/models/mlpdyn_ft_real_1"
FIXED_FRAMESKIP = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-run-dir", type=Path, default=DEFAULT_INIT_RUN_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume training from a Lightning checkpoint, restoring optimizer/scheduler state.",
    )
    parser.add_argument("--dataset-glob", default=DEFAULT_DATASET_GLOB)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--action-key",
        default="action",
        help="Which action tensor to train on from the real-data shards, for example measured_action, action, or control.",
    )
    parser.add_argument("--output-model-name", default="lewm")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=1.0)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-scale", default="tiny")
    parser.add_argument("--embed-dim", type=int, default=12)
    parser.add_argument("--markov-deriv", type=int, default=1)
    parser.add_argument("--num-preds", type=int, default=5, help="Autoregressive rollout horizon.")
    parser.add_argument("--action-dim", type=int, default=3)

    parser.add_argument("--predictor-hidden-width", type=int, default=512)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)

    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--straighten", action="store_true", default=True, help="Apply temporal straightening to encoder latents.")
    parser.add_argument("--straighten-weight", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--load-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load projector weights from the init checkpoint.",
    )
    parser.add_argument(
        "--load-mlpdyn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load MLP dynamics weights from the init checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep training dataloader workers alive across epochs. Validation workers stay non-persistent to avoid doubled RAM usage.",
    )
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=2)
    args = parser.parse_args()
    args.frameskip = FIXED_FRAMESKIP
    args.markov_state_dim = (args.markov_deriv + 1) * args.embed_dim
    return args


class LeWMRopeRealDataset(Dataset):
    def __init__(
        self,
        dataset_glob: str,
        *,
        markov_deriv: int,
        num_preds: int,
        frameskip: int,
        img_size: int,
        action_dim: int,
        action_key: str,
    ) -> None:
        self.dataset_glob = dataset_glob
        self.markov_deriv = int(markov_deriv)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        self.num_steps = 1 + self.num_preds
        self.action_steps = self.num_preds
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.action_key = str(action_key)
        self.effective_action_dim = self.frameskip * self.action_dim
        self._h5_files: dict[int, h5py.File] = {}

        if self.markov_deriv < 0:
            raise ValueError("markov_deriv must be non-negative.")
        if self.num_preds < 1:
            raise ValueError("num_preds must be positive.")
        if self.frameskip < 1:
            raise ValueError("frameskip must be positive.")

        shard_paths = [Path(p).expanduser().resolve() for p in sorted(glob.glob(dataset_glob))]
        if not shard_paths:
            raise FileNotFoundError(f"No shard files matched: {dataset_glob}")

        self.shard_infos: list[dict[str, object]] = []
        self.skipped_shards: list[dict[str, str]] = []
        action_sum = np.zeros((1, self.action_dim), dtype=np.float64)
        action_sq_sum = np.zeros((1, self.action_dim), dtype=np.float64)
        action_count = 0

        for shard_path in shard_paths:
            try:
                with h5py.File(shard_path, "r") as h5:
                    if self.action_key not in h5:
                        raise KeyError(f"Action key not found in shard: {self.action_key}")
                    if int(h5[self.action_key].shape[-1]) != self.action_dim:
                        raise ValueError(f"Expected action_dim={self.action_dim}, got {h5[self.action_key].shape[-1]}.")
                    ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
                    ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
                    actions = np.asarray(h5[self.action_key][:], dtype=np.float32)
            except Exception as exc:
                self.skipped_shards.append({"path": str(shard_path), "error": repr(exc)})
                continue

            finite_mask = ~np.isnan(actions).any(axis=1)
            finite_actions = actions[finite_mask]
            if finite_actions.size == 0:
                raise ValueError(f"No finite action rows found in shard: {shard_path}")
            action_sum += finite_actions.sum(axis=0, keepdims=True, dtype=np.float64)
            action_sq_sum += np.square(finite_actions, dtype=np.float64).sum(axis=0, keepdims=True)
            action_count += int(finite_actions.shape[0])

            self.shard_infos.append(
                {
                    "path": shard_path,
                    "ep_len": ep_len,
                    "ep_offset": ep_offset,
                }
            )

        if not self.shard_infos:
            raise ValueError("No readable shards found.")

        action_mean = action_sum / action_count
        action_var = np.maximum(action_sq_sum / action_count - np.square(action_mean), 0.0)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = np.maximum(np.sqrt(action_var), 1e-6).astype(np.float32)

        self.samples: list[tuple[int, int, int]] = []
        required_last_frame_offset = (self.num_steps - 1) * self.frameskip
        required_action_end_offset = self.action_steps * self.frameskip
        required_offset = max(required_last_frame_offset, required_action_end_offset)
        for shard_idx, shard_info in enumerate(self.shard_infos):
            ep_len = shard_info["ep_len"]
            for ep_idx, ep_len_i in enumerate(ep_len.tolist()):
                max_start = ep_len_i - 1 - required_offset
                for start in range(max_start + 1):
                    self.samples.append((shard_idx, ep_idx, start))
        if not self.samples:
            raise ValueError("No valid training windows found. Check frameskip/markov_deriv/num_preds.")

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, shard_idx: int) -> h5py.File:
        h5 = self._h5_files.get(shard_idx)
        if h5 is None:
            shard_path = self.shard_infos[shard_idx]["path"]
            h5 = h5py.File(shard_path, "r")
            self._h5_files[shard_idx] = h5
        return h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_idx, ep_idx, start = self.samples[index]
        shard_info = self.shard_infos[shard_idx]
        h5 = self._file(shard_idx)
        ep_offset = shard_info["ep_offset"]
        base = int(ep_offset[ep_idx]) + start

        frame_offsets = np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixel_rows = base + frame_offsets
        pixels_np = np.asarray(h5["pixels"][pixel_rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        if self.markov_deriv > 0:
            prev_offsets = np.arange(self.markov_deriv, 0, -1, dtype=np.int64) * self.frameskip
            prev_rows = int(ep_offset[ep_idx]) + np.maximum(start - prev_offsets, 0)
            prev_pixels_np = np.stack([np.asarray(h5["pixels"][int(row)], dtype=np.uint8) for row in prev_rows], axis=0)
            prev_pixels = torch.from_numpy(prev_pixels_np).permute(0, 3, 1, 2).contiguous()
        else:
            prev_pixels = pixels[:0].clone()

        action_blocks = []
        for step in range(self.action_steps):
            action_start = base + step * self.frameskip
            action_stop = action_start + self.frameskip
            block = np.asarray(h5[self.action_key][action_start:action_stop], dtype=np.float32)
            block = (np.nan_to_num(block, nan=0.0) - self.action_mean) / self.action_std
            action_blocks.append(torch.from_numpy(block.reshape(-1)))

        return {
            "pixels": pixels,
            "prev_pixels": prev_pixels,
            "action": torch.stack(action_blocks, dim=0).float(),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_files"] = {}
        return state

    def __del__(self) -> None:
        h5_files = getattr(self, "_h5_files", {})
        for h5 in h5_files.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5_files = {}


class ModelObjectCallback(Callback):
    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt")


def temporal_straightening_loss(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if emb.shape[1] < 3:
        return emb.new_zeros(())
    vel_prev = emb[:, 1:-1] - emb[:, :-2]
    vel_next = emb[:, 2:] - emb[:, 1:-1]
    cosine = F.cosine_similarity(vel_prev, vel_next, dim=-1, eps=eps)
    return (1.0 - cosine).mean()


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


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        batch_size, time_steps = pixels.shape[:2]
        pixels = F.interpolate(
            pixels.view(batch_size * time_steps, *pixels.shape[2:]),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, time_steps, *pixels.shape[2:3], img_size, img_size)
    pixel_mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    pixel_std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (pixels - pixel_mean) / pixel_std


def lewm_forward(self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace):
    n_preds = args.num_preds
    lambd = args.sigreg_weight
    embed_dim = args.embed_dim
    markov_context_len = required_markov_history(args.markov_deriv)

    all_pixels = torch.cat((batch["prev_pixels"], batch["pixels"]), dim=1)
    all_pixels = preprocess_pixels(all_pixels, args.img_size)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode({"pixels": all_pixels, "action": batch["action"]})
    history_emb = output["emb"][:, :markov_context_len]
    emb = output["emb"][:, args.markov_deriv :]
    output["emb"] = emb
    act_emb = output["act_emb"]
    rollout_state = build_markov_state(history_emb, args.markov_deriv)
    pred_losses = []
    pred_embs = []
    for step in range(n_preds):
        ctx_state = rollout_state.unsqueeze(1)
        ctx_act = act_emb[:, step : step + 1]

        pred_state = self.model.predict(ctx_state, ctx_act)[:, 0]
        pred_next = pred_state[..., :embed_dim]
        tgt_next = emb[:, step + 1]
        if args.markov_deriv > 0:
            tgt_history = torch.cat((emb[:, step + 1 - args.markov_deriv : step + 1], tgt_next.unsqueeze(1)), dim=1)
        else:
            tgt_history = tgt_next.unsqueeze(1)
        tgt_state = build_markov_state(tgt_history, args.markov_deriv)
        pred_losses.append((pred_state - tgt_state).pow(2).mean())
        pred_embs.append(pred_next)
        rollout_state = pred_state

    output["pred_emb"] = torch.stack(pred_embs, dim=1)
    output["pred_loss"] = torch.stack(pred_losses).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["straighten_loss"] = temporal_straightening_loss(emb) if args.straighten else emb.new_zeros(())
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + args.straighten_weight * output["straighten_loss"]
    )

    log_prefix = "train" if stage == "fit" else stage
    losses = {f"{log_prefix}/{key}": value.detach() for key, value in output.items() if "loss" in key}
    self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return output


def build_model(args: argparse.Namespace) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        args.encoder_scale,
        patch_size=args.patch_size,
        image_size=args.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = args.embed_dim
    markov_state_dim = args.markov_state_dim
    effective_act_dim = args.frameskip * args.action_dim

    predictor = MLPDynamicsPredictor(
        embed_dim=markov_state_dim,
        action_dim=effective_act_dim,
        history_size=1,
        action_history_size=1,
        num_preds=1,
        hidden_width=args.predictor_hidden_width,
        depth=args.predictor_depth,
        dropout=args.predictor_dropout,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=torch.nn.Identity(),
        projector=projector,
        pred_proj=torch.nn.Identity(),
    )


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator == "gpu",
        "persistent_workers": args.num_workers > 0 and persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def sanitize_hparams(args: argparse.Namespace) -> dict[str, object]:
    hparams = vars(args).copy()
    for key, value in hparams.items():
        if isinstance(value, Path):
            hparams[key] = str(value)
    return hparams


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


def resolve_init_checkpoint(args: argparse.Namespace) -> Path:
    if args.init_checkpoint is not None:
        checkpoint_path = args.init_checkpoint.expanduser().resolve()
    else:
        checkpoint_path = latest_object_checkpoint(args.init_run_dir.expanduser().resolve()).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_checkpoint is None:
        return None
    checkpoint_path = args.resume_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_pretrained_model(checkpoint_path: Path) -> JEPA:
    source_model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = getattr(source_model, "encoder", None)
    predictor = getattr(source_model, "predictor", None)
    action_encoder = getattr(source_model, "action_encoder", None)
    if encoder is None or predictor is None or action_encoder is None:
        source_type = f"{type(source_model).__module__}.{type(source_model).__name__}"
        raise TypeError(
            "Expected a JEPA-compatible object checkpoint with "
            f"`encoder`, `predictor`, and `action_encoder`, got {source_type}."
        )
    return source_model


def load_module_state(target: torch.nn.Module, source: torch.nn.Module, module_name: str) -> None:
    try:
        target.load_state_dict(source.state_dict(), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load {module_name} weights from the init checkpoint. "
            "Check architecture compatibility for this fine-tune run."
        ) from exc


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    run_dir = args.run_dir.expanduser().resolve()
    resume_checkpoint_path = resolve_resume_checkpoint(args)
    init_checkpoint_path = None if resume_checkpoint_path is not None else resolve_init_checkpoint(args)
    if run_dir.exists() and resume_checkpoint_path is None:
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    if not run_dir.exists() and resume_checkpoint_path is not None:
        raise FileNotFoundError(f"Run dir does not exist for resume: {run_dir}")

    dataset = LeWMRopeRealDataset(
        args.dataset_glob,
        markov_deriv=args.markov_deriv,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        action_dim=args.action_dim,
        action_key=args.action_key,
    )
    if len(dataset) < 1:
        raise ValueError("Need at least 1 valid training window.")
    train_len = int(len(dataset) * args.train_split)
    val_len = len(dataset) - train_len
    if train_len < 1:
        train_len = 1
        val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = make_loader(
        train_set,
        args,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.persistent_workers,
    )
    val_loader = (
        make_loader(
            val_set,
            args,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )
        if val_len
        else None
    )

    world_model = build_model(args)
    if resume_checkpoint_path is None:
        pretrained_model = load_pretrained_model(init_checkpoint_path)
        pretrained_encoder = getattr(pretrained_model, "encoder", None)
        if pretrained_encoder is None:
            raise AttributeError("Source checkpoint does not contain model.encoder.")
        load_module_state(world_model.encoder, pretrained_encoder, "encoder")
        if args.load_projector:
            pretrained_projector = getattr(pretrained_model, "projector", None)
            if pretrained_projector is None:
                raise AttributeError("Source checkpoint does not contain model.projector.")
            load_module_state(world_model.projector, pretrained_projector, "projector")
        if args.load_mlpdyn:
            pretrained_predictor = getattr(pretrained_model, "predictor", None)
            if pretrained_predictor is None:
                raise AttributeError("Source checkpoint does not contain model.predictor.")
            load_module_state(world_model.predictor, pretrained_predictor, "predictor")

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj),
        forward=partial(lewm_forward, args=args),
        optim=optimizers,
        hparams=sanitize_hparams(args),
    )

    spt.set(cache_dir=str(run_dir))
    if resume_checkpoint_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = vars(args).copy()
        config["run_dir"] = str(run_dir)
        config["dataset_glob"] = args.dataset_glob
        config["init_checkpoint"] = str(init_checkpoint_path)
        config["readable_shards"] = [str(info["path"]) for info in dataset.shard_infos]
        config["skipped_shards"] = dataset.skipped_shards
        with (run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, default=str)

        loaded_modules = ["encoder"]
        if args.load_projector:
            loaded_modules.append("projector")
        if args.load_mlpdyn:
            loaded_modules.append("predictor")

        reinitialized_modules = ["action_encoder", "pred_proj"]
        if not args.load_projector:
            reinitialized_modules.insert(0, "projector")
        if not args.load_mlpdyn:
            reinitialized_modules.insert(1 if not args.load_projector else 0, "predictor")

        init_report = {
            "init_checkpoint": str(init_checkpoint_path),
            "loaded_modules": loaded_modules,
            "reinitialized_modules": reinitialized_modules,
            "load_projector": bool(args.load_projector),
            "load_mlpdyn": bool(args.load_mlpdyn),
            "action_key": str(args.action_key),
            "action_dim": int(args.action_dim),
        }
        with (run_dir / "init_report.json").open("w") as f:
            json.dump(init_report, f, indent=2)
    else:
        resume_report = {
            "resume_checkpoint": str(resume_checkpoint_path),
            "run_dir": str(run_dir),
            "sigreg_weight": float(args.sigreg_weight),
            "action_key": str(args.action_key),
            "action_dim": int(args.action_dim),
        }
        with (run_dir / "resume_report.json").open("w") as f:
            json.dump(resume_report, f, indent=2)

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename=f"{args.output_model_name}" + "_{epoch:03d}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        ),
        ModelObjectCallback(run_dir, args.output_model_name, epoch_interval=args.save_object_every),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=run_dir,
        logger=False,
        num_sanity_val_steps=1 if val_loader is not None else 0,
        enable_checkpointing=True,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_checkpoint_path) if resume_checkpoint_path else None)


if __name__ == "__main__":
    main()
