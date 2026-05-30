#!/usr/bin/env python3
"""Standalone generator for 1-step prediction errors for the Reacher model."""

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/latent_brs_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# Imports from the reacher training stack matching the Markov dynamics model.
from reacher.train.mlpdyn_ft import build_markov_state, preprocess_pixels, required_markov_history

DEFAULT_DATASET_PATH = "reacher/data/test_data_noisy.h5"
DEFAULT_ACTION_STATS_DATASET_PATH = "reacher/data/train_data_noisy.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_embd_5"

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def load_action_stats(stats_dataset_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(stats_dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != int(args.action_dim):
            raise ValueError(f"Expected action_dim={args.action_dim}, got {h5['action'].shape[-1]} in {stats_dataset_path}.")
        finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
    if finite_actions.size == 0:
        raise ValueError(f"No finite actions found in {stats_dataset_path}.")
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(finite_actions.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
    return action_mean, action_std

def load_episode_standalone(
    dataset_path: Path,
    episode_idx: int,
    args: argparse.Namespace,
    action_mean: np.ndarray,
    action_std: np.ndarray,
):
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()

        # Match the image preprocessing used during reacher dynamics training.
        pixels = preprocess_pixels(pixels.unsqueeze(0), args.img_size)[0]

        actions = np.asarray(h5["action"][rows], dtype=np.float32)
        actions = (np.nan_to_num(actions, nan=0.0) - action_mean) / action_std

    return pixels, torch.from_numpy(actions).float()

@torch.no_grad()
def extract_errors(model, pixels, actions, args, device):
    # Process images through the encoder
    latents = []
    for start in range(0, pixels.shape[0], args.frame_batch_size):
        chunk = pixels[start : start + args.frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    true_latents = torch.cat(latents, dim=0)

    rollout_steps = (true_latents.shape[0] - 1) // args.frameskip
    states, acts, targets = [], [], []
    
    embed_dim = true_latents.shape[-1]
    history_len = required_markov_history(args.markov_deriv)

    for step in range(rollout_steps):
        # Gather the history frames required up to this point
        t_curr = step * args.frameskip
        
        # Emulate the evaluation script's warm-start / history padding logic
        if t_curr == 0:
            history = true_latents[:1]
            if history_len > 1:
                history = torch.cat((history[:1].repeat(history_len - 1, 1), history), dim=0)
        else:
            # Gather available historical context back to history_len
            start_idx = max(0, t_curr - history_len + 1)
            history = true_latents[start_idx : t_curr + 1]
            if history.shape[0] < history_len:
                padding_amt = history_len - history.shape[0]
                history = torch.cat((history[:1].repeat(padding_amt, 1), history), dim=0)

        # Build current Markov State [z_t, delta_z_t, ...]
        curr_state = build_markov_state(history.unsqueeze(0), args.markov_deriv)[0]
        states.append(curr_state)
        
        # Grab actions across frameskip window
        action_start = step * args.frameskip
        action_stop = action_start + args.frameskip
        acts.append(actions[action_start:action_stop].flatten())
        
        # Build TARGET Markov State for t + frameskip
        t_next = t_curr + args.frameskip
        if t_next < true_latents.shape[0]:
            start_idx_next = max(0, t_next - history_len + 1)
            history_next = true_latents[start_idx_next : t_next + 1]
            if history_next.shape[0] < history_len:
                padding_amt = history_len - history_next.shape[0]
                history_next = torch.cat((history_next[:1].repeat(padding_amt, 1), history_next), dim=0)
            
            next_state = build_markov_state(history_next.unsqueeze(0), args.markov_deriv)[0]
            targets.append(next_state)
        else:
            # Handle edge case where target exceeds encoded timeline sequence
            states.pop()
            acts.pop()
            break

    if not states:
        return None

    s_tsr = torch.stack(states).to(device)
    a_tsr = torch.stack(acts).to(device)
    target_tsr = torch.stack(targets).to(device)
    
    # Dynamics prediction f(s, a)
    act_emb = model.action_encoder(a_tsr.unsqueeze(1))
    pred_s = model.predict(s_tsr.unsqueeze(1), act_emb)[:, 0] 
    
    return {"x_t": s_tsr.cpu(), "a_t": a_tsr.cpu(), "error": (target_tsr - pred_s).cpu()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--action-stats-dataset-path", type=Path, default=DEFAULT_ACTION_STATS_DATASET_PATH)
    parser.add_argument("--out-file", type=Path, default="reacher/eval/reacher_one_step_error_data.pt")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    
    # Command line overrides mimicking the config fields
    parser.add_argument("--markov-deriv", type=int, default=1) 
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    
    for k in ["markov_deriv", "frameskip", "img_size", "action_dim"]:
        val = config.get(k)
        if val is not None:
            setattr(args, k, val)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()
    action_mean, action_std = load_action_stats(args.action_stats_dataset_path, args)
    
    with h5py.File(args.dataset_path, "r") as h5:
        ep_len = h5["ep_len"][:]
        
    num_steps = 1 + int(args.num_preds)
    required_offset = max((num_steps - 1) * int(args.frameskip), int(args.num_preds) * int(args.frameskip))
    valid_indices = np.flatnonzero(ep_len - 1 - required_offset >= 0)
    if valid_indices.size == 0:
        raise ValueError(
            "No valid one-step windows found. "
            f"min_ep_len={int(np.min(ep_len)) if ep_len.size else 'n/a'}, "
            f"max_ep_len={int(np.max(ep_len)) if ep_len.size else 'n/a'}, "
            f"num_preds={args.num_preds}, frameskip={args.frameskip}, required_offset={required_offset}. "
            "For a noisy one-step file with two-frame episodes, use --num-preds=1."
        )

    all_x, all_a, all_e = [], [], []
    for idx in tqdm(valid_indices, desc="Generating Errors"):
        px, act = load_episode_standalone(args.dataset_path, idx, args, action_mean, action_std)
        data = extract_errors(model, px, act, args, device)
        if data is not None:
            all_x.append(data["x_t"])
            all_a.append(data["a_t"])
            all_e.append(data["error"])

    if not all_x:
        raise ValueError(
            "No prediction errors were generated after processing valid episodes. "
            "Check that each episode has at least two frames after frameskip and that action dimensions match the model."
        )

    torch.save({"x_t": torch.cat(all_x), "a_t": torch.cat(all_a), "error": torch.cat(all_e)}, args.out_file)
    print(f"Saved {len(torch.cat(all_x))} transitions to {args.out_file}")


if __name__ == "__main__":
    main()
