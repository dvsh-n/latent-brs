#!/usr/bin/env python3
"""Standalone generator for 1-step prediction errors for OGBench MLP predictor."""

import argparse
import json
import torch
import h5py
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Keep only external imports for OGBench
from ogbench_cube.train.mlpdyn_train import (
    LeWMOGBenchCubeDataset,
    build_markov_state,
    preprocess_pixels,
    required_markov_history,
)

# --- Re-defined Constants from mlpdyn_eval to break circular import ---
DEFAULT_DATASET_PATH = "ogbench_cube/data/test_data/ogbench_cube_test.h5"
DEFAULT_MODEL_DIR = "ogbench_cube/models/mlpdyn"

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

def load_episode_standalone(dataset_path, episode_idx, args):
    dataset = LeWMOGBenchCubeDataset(
        dataset_path,
        markov_deriv=args.markov_deriv,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        action_dim=args.action_dim,
    )

    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        pixels = preprocess_pixels(pixels.unsqueeze(0), args.img_size)[0]

        actions = np.asarray(h5["action"][rows], dtype=np.float32)
        actions = (np.nan_to_num(actions, nan=0.0) - dataset.action_mean) / dataset.action_std
        actions = torch.from_numpy(actions).float()
        
    return pixels, actions

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

    history_len = required_markov_history(args.markov_deriv)
    rollout_steps = (true_latents.shape[0] - 1 - (history_len - 1) * args.frameskip) // args.frameskip
    
    if rollout_steps < 1:
        return None

    states, acts, targets = [], [], []

    for step in range(rollout_steps):
        t_curr = step * args.frameskip + (history_len - 1) * args.frameskip
        
        # Current State: build markov state dynamically using native OGBench logic
        hist_indices = [t_curr - i * args.frameskip for i in range(history_len - 1, -1, -1)]
        hist_z = true_latents[hist_indices] 
        curr_state = build_markov_state(hist_z.unsqueeze(0), args.markov_deriv)[0]
        states.append(curr_state)
        
        a_start = t_curr
        acts.append(actions[a_start : a_start + args.frameskip].flatten())
        
        # Target: Full Markov State
        t_next = t_curr + args.frameskip
        next_hist_indices = [t_next - i * args.frameskip for i in range(history_len - 1, -1, -1)]
        next_hist_z = true_latents[next_hist_indices]
        next_state = build_markov_state(next_hist_z.unsqueeze(0), args.markov_deriv)[0]
        targets.append(next_state)

    s_tsr = torch.stack(states).to(device)
    a_tsr = torch.stack(acts).to(device)
    target_tsr = torch.stack(targets).to(device)
    
    # Dynamics prediction f(s, a)
    act_emb = model.action_encoder(a_tsr.unsqueeze(1))
    
    # Extract the full dimensional state prediction
    pred_s = model.predict(s_tsr.unsqueeze(1), act_emb)[:, 0] 
    
    # Error calculated directly in the n-dimensional derivative space
    return {"x_t": s_tsr.cpu(), "a_t": a_tsr.cpu(), "error": (target_tsr - pred_s).cpu()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-file", type=Path, default="lewm_one_step_error_data_ogbench.pt")
    parser.add_argument("--frame-batch-size", type=int, default=128)
    args = parser.parse_args()

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    
    # --- FIXED: Config injection with robust fallbacks ---
    defaults = {
        "markov_deriv": 1,
        "num_preds": 1,
        "frameskip": 1,
        "img_size": 224,
        "action_dim": 5,
    }
    for k, fallback in defaults.items():
        val = config.get(k)
        setattr(args, k, val if val is not None else fallback)
    # -----------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()
    
    with h5py.File(args.dataset_path, "r") as h5:
        ep_len = h5["ep_len"][:]
        
    history_len = required_markov_history(args.markov_deriv)
    valid_indices = np.flatnonzero(ep_len - 1 - (history_len - 1 + args.num_preds) * args.frameskip >= 0)

    all_x, all_a, all_e = [], [], []
    for idx in tqdm(valid_indices, desc="Generating Errors"):
        px, act = load_episode_standalone(args.dataset_path, idx, args)
        data = extract_errors(model, px, act, args, device)
        if data is not None:
            all_x.append(data["x_t"])
            all_a.append(data["a_t"])
            all_e.append(data["error"])

    torch.save({"x_t": torch.cat(all_x), "a_t": torch.cat(all_a), "error": torch.cat(all_e)}, args.out_file)
    print(f"Saved {len(torch.cat(all_x))} transitions to {args.out_file}")

if __name__ == "__main__":
    main()