#!/usr/bin/env python3
"""Train an LE-WM-style JEPA with a Markov latent-state MLP dynamics predictor on PushT."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    hdf5plugin = None

import h5py
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split

from reacher.shared.models import JEPA, MLP, MLPDynamicsPredictor, SIGReg


DEFAULT_DATASET_PATH = "pusht/data/pusht_expert_train.h5"
DEFAULT_RUN_DIR = "pusht/models/lewm_pusht_mlpdyn_markov"
DEFAULT_FRAMESKIP = 1
REQUIRED_DATASET_KEYS = ("ep_len", "ep_offset", "pixels", "action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-model-name", default="lewm")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=0.9)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-scale", default="tiny")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--num-preds", type=int, default=4, help="Autoregressive rollout horizon.")
    parser.add_argument("--frameskip", type=int, default=DEFAULT_FRAMESKIP)
    parser.add_argument("--action-dim", type=int, default=2)

    parser.add_argument("--predictor-hidden-width", type=int, default=512)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)

    parser.add_argument("--sigreg-weight", type=float, default=0.009)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--straighten", action="store_true", help="Apply temporal straightening to encoder latents.")
    parser.add_argument("--straighten-weight", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=1)
    args = parser.parse_args()
    args.markov_state_dim = 2 * args.embed_dim
    return args


class LeWMPushTDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        *,
        history_size: int,
        num_preds: int,
        frameskip: int,
        img_size: int,
        action_dim: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        self.num_steps = self.history_size + self.num_preds
        self.action_steps = self.num_preds
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.effective_action_dim = self.frameskip * self.action_dim
        self._h5: h5py.File | None = None

        if self.history_size < 1:
            raise ValueError("history_size must be positive.")
        if self.num_preds < 1:
            raise ValueError("num_preds must be positive.")
        if self.frameskip < 1:
            raise ValueError("frameskip must be positive.")

        with h5py.File(self.dataset_path, "r") as h5:
            missing = [key for key in REQUIRED_DATASET_KEYS if key not in h5]
            if missing:
                raise KeyError(f"PushT dataset is missing required keys: {missing}")
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if int(h5["action"].shape[-1]) != self.action_dim:
                raise ValueError(f"Expected action_dim={self.action_dim}, got {h5['action'].shape[-1]}.")
            finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
            finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
            if finite_actions.size == 0:
                raise ValueError("No finite actions found in PushT dataset.")
            self.action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
            self.action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
            self.action_std = np.maximum(self.action_std, 1e-6)

        self.samples: list[tuple[int, int]] = []
        required_last_frame_offset = (self.num_steps - 1) * self.frameskip
        action_start_step = self.history_size - 1
        required_action_end_offset = (action_start_step + self.action_steps) * self.frameskip
        required_offset = max(required_last_frame_offset, required_action_end_offset)
        for ep_idx, ep_len in enumerate(self.ep_len.tolist()):
            max_start = ep_len - 1 - required_offset
            for start in range(max_start + 1):
                self.samples.append((ep_idx, start))
        if not self.samples:
            raise ValueError("No valid PushT training windows found. Check frameskip/history/num_preds.")
        self.num_valid_episodes = len({ep_idx for ep_idx, _ in self.samples})

        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        ep_idx, start = self.samples[index]
        base = int(self.ep_offset[ep_idx]) + start

        frame_offsets = np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixel_rows = base + frame_offsets
        pixels_np = np.asarray(h5["pixels"][pixel_rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).float().div_(255.0)
        if pixels.shape[-2:] != (self.img_size, self.img_size):
            pixels = F.interpolate(pixels, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        pixels = (pixels - self.pixel_mean) / self.pixel_std

        action_blocks = []
        first_action_step = self.history_size - 1
        for step in range(first_action_step, first_action_step + self.action_steps):
            action_start = base + step * self.frameskip
            action_stop = action_start + self.frameskip
            block = np.asarray(h5["action"][action_start:action_stop], dtype=np.float32)
            block = (np.nan_to_num(block, nan=0.0) - self.action_mean) / self.action_std
            action_blocks.append(torch.from_numpy(block.reshape(-1)))

        return {
            "pixels": pixels,
            "action": torch.stack(action_blocks, dim=0).float(),
        }

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


def lewm_forward(self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace):
    ctx_len = args.history_size
    n_preds = args.num_preds
    lambd = args.sigreg_weight
    embed_dim = args.embed_dim

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    rollout_emb = emb[:, :ctx_len]
    pred_losses = []
    pred_embs = []
    for step in range(n_preds):
        current_emb = rollout_emb[:, -1]
        if rollout_emb.shape[1] >= 2:
            delta_emb = current_emb - rollout_emb[:, -2]
        else:
            delta_emb = torch.zeros_like(current_emb)
        ctx_state = torch.cat((current_emb, delta_emb), dim=-1).unsqueeze(1)
        ctx_act = act_emb[:, step : step + 1]

        pred_state = self.model.predict(ctx_state, ctx_act)[:, 0]
        pred_next = pred_state[..., :embed_dim]
        tgt_next = emb[:, ctx_len + step]
        tgt_delta = tgt_next - current_emb.detach()
        tgt_state = torch.cat((tgt_next, tgt_delta), dim=-1)
        pred_losses.append((pred_state - tgt_state).pow(2).mean())
        pred_embs.append(pred_next)
        rollout_emb = torch.cat((rollout_emb, pred_next.unsqueeze(1)), dim=1)

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
    markov_state_dim = 2 * embed_dim
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


def make_loader(dataset: Dataset, args: argparse.Namespace, *, shuffle: bool, drop_last: bool) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator == "gpu",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dataset_path = args.dataset_path.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    spt.set(cache_dir=str(run_dir))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not 0.0 < args.train_split < 1.0:
        raise ValueError("--train-split must be between 0 and 1.")

    dataset = LeWMPushTDataset(
        dataset_path,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        action_dim=args.action_dim,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 valid training windows for train/val splitting, got {len(dataset)}.")
    train_len = int(len(dataset) * args.train_split)
    val_len = len(dataset) - train_len
    if train_len < 1:
        train_len = 1
        val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = make_loader(train_set, args, shuffle=True, drop_last=True)
    val_loader = make_loader(val_set, args, shuffle=False, drop_last=False) if val_len else None

    world_model = build_model(args)
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
        hparams=vars(args),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

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
    ckpt_path = str(run_dir / "last.ckpt") if (run_dir / "last.ckpt").is_file() else None
    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
