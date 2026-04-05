# koop_eval.py
import sys
import argparse
from pathlib import Path

# --- Root Path Setup ---
def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path

ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))
# -----------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model and dataset base classes
from models import DeepKoopman
from utils import resolve_path, print_config_table, print_rmse_table, print_table

# ====== CONFIGURATION & HYPERPARAMETERS ======
DATA_PATH = "data/random_data.pt"
# MODEL_SAVE_PATH = "models/sweep/koop_quad3d_h50.pt"
MODEL_SAVE_PATH = "models/koop_quad3d_dagger_1.pt"
PLOT_SAVE_DIR = "eval/eval_plots" 
N_EVAL_EPISODES = 25
PURE_LATENT_ROLLOUT = True
OVERALL_RMSE_STEPS = [1, 50, 100, 150]

def parse_eval_args():
    """Parse command-line args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Deep Koopman model")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=MODEL_SAVE_PATH,
        help="Path to trained model checkpoint",
    )
    return parser.parse_args()

# ====== EVALUATION HELPERS ======
class QuadrotorDatasetEval(nn.Module):
    """
    Evaluation dataset for 3D data.
    Loads data and uses pre-computed normalization stats from the trained model.
    """
    def __init__(self, data_path):
        super().__init__()
        # Load the list of demo dictionaries
        demos = torch.load(data_path, weights_only=False)
        
        self.states_data = []
        self.controls_data = []

        # Iterate through each demo, transpose, and append
        for demo in demos:
            # demo['states'] is [12, N+1], demo['controls'] is [4, N]
            # Transpose to [N+1, 12] and [N, 4]
            self.states_data.append(torch.tensor(demo['states'].T, dtype=torch.float32))
            self.controls_data.append(torch.tensor(demo['controls'].T, dtype=torch.float32))
        
        self.min = None
        self.max = None
        self.range = None

    def set_normalization_stats(self, stats: dict):
        """Sets the normalization stats from the loaded model file."""
        self.min = stats['min']
        self.max = stats['max']
        # Use 'range' if available, otherwise compute it.
        self.range = stats.get('range', self.max - self.min)
        # Add robustness for any state dim that was constant
        self.range[self.range == 0] = 1e-6 

    def normalize(self, x_12d: torch.Tensor) -> torch.Tensor:
        """Applies min-max normalization to the full 12D state vector."""
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        
        x_norm = x_12d.clone()
        min_dev = self.min.to(x_12d.device)
        range_dev = self.range.to(x_12d.device)
        
        # Apply normalization: (x - min) / (max - min) -> [0, 1] -> * 2 - 1 -> [-1, 1]
        x_norm = 2 * (x_norm - min_dev) / range_dev - 1
        return x_norm

    def denormalize(self, x_12d_norm: torch.Tensor) -> torch.Tensor:
        """Reverses min-max normalization for the full 12D state vector."""
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        
        out = x_12d_norm.clone()
        min_dev = self.min.to(out.device)
        range_dev = self.range.to(out.device)

        # Reverse normalization: (x_norm + 1) / 2 -> [0, 1] -> * range + min -> original scale
        out = (out + 1) / 2 * range_dev + min_dev
        return out

    def get_raw_episode(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the full state and control trajectory for one episode."""
        return self.states_data[idx], self.controls_data[idx]

def rollout_latent(model, x0_12d, u_raw, dataset, device):
    """
    Performs a rollout purely in the latent space and decodes the 
    entire trajectory only at the end.
    """
    model.eval()
    with torch.no_grad():
        H = len(u_raw)
        x0_12d, u_raw = x0_12d.to(device), u_raw.to(device)
        
        # 1. Lift the initial state into the latent space (z0)
        # Add batch dim [1, 12]
        x0_norm = dataset.normalize(x0_12d.unsqueeze(0)) 
        z_k = model.encoder(x0_norm) # Initial latent state [1, latent_dim]

        # 2. Simulate the trajectory purely in the latent space
        z_hat_list = [z_k]
        for k in range(H):
            u_k = u_raw[k].unsqueeze(0) # Add batch dim [1, 4] for B matrix
            z_kp1 = model.A(z_k) + model.B(u_k)
            z_hat_list.append(z_kp1)
            z_k = z_kp1 

        # 3. Decode the entire latent state trajectory in one batch
        z_hat_traj = torch.cat(z_hat_list, dim=0) # [H+1, latent_dim]
        
        x_hat_norm_traj = model.decoder(z_hat_traj) # [H+1, 12]

        # 4. Denormalize the entire trajectory at once
        x_hat_traj = dataset.denormalize(x_hat_norm_traj)
        
        return x_hat_traj.cpu().numpy()
    
def rollout_recurrent(model, x0_12d, u_raw, dataset, device):
    """
    Performs a rollout by repeatedly decoding, re-encoding, and propagating
    at each time step.
    """
    model.eval()
    with torch.no_grad():
        H = len(u_raw)
        x0_12d, u_raw = x0_12d.to(device), u_raw.to(device)
        
        # Store the trajectory of predicted states (denormalized)
        x_hat_denorm_list = [x0_12d] # Store 12D states

        # 3. Simulation loop with encode-propagate-decode at each step
        for k in range(H):
            # A. Normalize and encode the *last predicted state*
            x_k_norm = dataset.normalize(x_hat_denorm_list[-1].unsqueeze(0))
            z_k = model.encoder(x_k_norm)
            
            # B. Propagate one step forward in the latent space
            u_k = u_raw[k].unsqueeze(0)
            z_kp1 = model.A(z_k) + model.B(u_k)
            
            # C. Decode the next latent vector to get the next normalized state
            x_kp1_norm = model.decoder(z_kp1)

            # D. Denormalize the predicted state to use for the next step
            x_kp1_denorm = dataset.denormalize(x_kp1_norm).squeeze(0) # Remove batch dim
            
            # E. Store the denormalized result
            x_hat_denorm_list.append(x_kp1_denorm)
            
        # 4. Stack the full (denormalized) trajectory
        x_hat_traj = torch.stack(x_hat_denorm_list)
        
        return x_hat_traj.cpu().numpy()
    
def calculate_rmse(x_12d_traj: np.ndarray, x_hat_12d_traj: np.ndarray) -> np.ndarray:
    """
    Calculates the Root Mean Squared Error between the 12D ground truth
    and 12D predicted trajectories.
    """
    # Ensure trajectories are the same length
    T = min(len(x_12d_traj), len(x_hat_12d_traj))
    err = x_12d_traj[:T] - x_hat_12d_traj[:T]
    
    # Calculate RMSE for each of the 12 dimensions
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return rmse

def plot_rollout(x_12d_traj: np.ndarray, x_hat_12d_traj: np.ndarray, episode_idx: int, plot_save_dir: str):
    """
    Plots the ground truth vs. the predicted Koopman rollout using a
    2x2 grid, each containing 3 subplots.
    """
    # 12D state labels: [px, py, pz, psi, th, phi, vx, vy, vz, p, q, r]
    labels = ['Pos X (m)', 'Pos Y (m)', 'Pos Z (m)',
              'Angle Ψ (rad)', 'Angle θ (rad)', 'Angle φ (rad)',
              'Vel X (m/s)', 'Vel Y (m/s)', 'Vel Z (m/s)',
              'Ang Vel p (rad/s)', 'Ang Vel q (rad/s)', 'Ang Vel r (rad/s)']

    # Create the figure and the 2x2 outer grid
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    fig.suptitle(f'Koopman Rollout vs. Ground Truth (Episode {episode_idx})', fontsize=20)
    outer_grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.4)

    # --- Create inner 3x1 grids ---
    # Top-left: Position (px, py, pz)
    axs_pos = outer_grid[0, 0].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_pos[0].set_title("Position", fontsize=16)
    
    # Top-right: Euler Angles (psi, th, phi)
    axs_ang = outer_grid[0, 1].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_ang[0].set_title("Euler Angles", fontsize=16)
    
    # Bottom-left: Linear Velocity (vx, vy, vz)
    axs_vel = outer_grid[1, 0].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_vel[0].set_title("Linear Velocity", fontsize=16)

    # Bottom-right: Angular Velocity (p, q, r)
    axs_ang_vel = outer_grid[1, 1].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_ang_vel[0].set_title("Angular Velocity", fontsize=16)

    # Group all axes and corresponding labels
    all_axs = [*axs_pos, *axs_ang, *axs_vel, *axs_ang_vel]
    
    T = min(len(x_12d_traj), len(x_hat_12d_traj))
    time_ax = np.arange(T)

    # Plot all 12 dimensions
    for i, ax in enumerate(all_axs):
        ax.plot(time_ax, x_12d_traj[:T, i], label='Ground Truth', color='blue', linewidth=2)
        ax.plot(time_ax, x_hat_12d_traj[:T, i], label='Koopman Rollout', color='red', linestyle='--', linewidth=2)
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle=':')

    # Add shared x-labels to the bottom plots of each column
    for ax in [axs_vel[-1], axs_ang_vel[-1]]:
        ax.set_xlabel('Time Step', fontsize=12)
    
    # Add a single, shared legend to the figure
    handles, leg_labels = all_axs[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='upper right', fontsize=14, bbox_to_anchor=(0.95, 0.98))

    Path(plot_save_dir).mkdir(exist_ok=True)
    save_path = Path(plot_save_dir) / f"episode_{episode_idx}.png"
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to save memory

def evaluate(model_save_path_arg=None):
    # Resolve paths relative to embodiment root
    data_path = resolve_path(DATA_PATH)
    model_save_path = resolve_path(model_save_path_arg or MODEL_SAVE_PATH)
    plot_save_dir = resolve_path(PLOT_SAVE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_data = torch.load(model_save_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_save_path}.")
        print("Please run 'koop_train.py' for the 3D model first.")
        return

    # 1. Initialize the 3D Evaluation Dataset
    dataset = QuadrotorDatasetEval(data_path)
    
    if 'normalization_stats' not in model_data:
        print("ERROR: 'normalization_stats' not found in model file. Please re-train and save.")
        return
        
    dataset.set_normalization_stats(model_data['normalization_stats'])

    # 2. Load the 3D Model
    # Re-create the model architecture from the saved config
    # Use 'model_config' if available (new format), otherwise fall back to 'config'
    model_config = model_data.get('model_config', model_data.get('config'))
    
    # Print the configuration
    print_config_table(model_config, title="Model Configuration")
    if 'training_config' in model_data:
        print_config_table(model_data['training_config'], title="Training Configuration")

    # Activation might be a string in the new format
    if isinstance(model_config.get('activation_fn'), str):
        from utils import get_activation_fn
        model_config['activation_fn'] = get_activation_fn(model_config['activation_fn'])

    # Filter out extra keys for DeepKoopman constructor
    model_params = {k: v for k, v in model_config.items() 
                    if k in ['state_dim', 'control_dim', 'latent_dim', 'hidden_width', 'depth', 'activation_fn']}
    model = DeepKoopman(**model_params).to(device)
    model.load_state_dict(model_data['state_dict'])
    
    state_errs = []
    step_mse = {step: [] for step in OVERALL_RMSE_STEPS}
    
    pbar = tqdm(range(N_EVAL_EPISODES), desc="Evaluating episodes")
    for ep_id in pbar:
        # Get the full 12D state trajectory and control trajectory
        x_12d, u = dataset.get_raw_episode(ep_id)
        x0_12d = x_12d[0] # Get the first state [12]
        
        if PURE_LATENT_ROLLOUT:
            x_hat_12d_traj = rollout_latent(model, x0_12d, u, dataset, device)
        else:
            x_hat_12d_traj = rollout_recurrent(model, x0_12d, u, dataset, device)
        
        x_12d_np = x_12d.numpy()

        # Plot the 12D rollout
        plot_rollout(x_12d_np, x_hat_12d_traj, ep_id, plot_save_dir)
        
        # Calculate RMSE over all 12 states
        rmse_state = calculate_rmse(x_12d_np, x_hat_12d_traj)
        state_errs.append(rmse_state)

        # Per-horizon overall RMSE (aggregated over all 12 dims at a specific step)
        T = min(len(x_12d_np), len(x_hat_12d_traj))
        err = x_12d_np[:T] - x_hat_12d_traj[:T]
        for step in OVERALL_RMSE_STEPS:
            if step < T:
                step_mse[step].append(float(np.mean(err[step] ** 2)))
        
        # Update pbar with mean position error (px, py, pz)
        pbar.set_postfix(mean_pos_err=f"{np.mean(np.array(state_errs)[:, :3]):.4f}")
        
    # --- Print Final Results ---
    state_errs_np = np.array(state_errs)
    rmse_state_mean = state_errs_np.mean(axis=0)

    grouped_labels = ["Position", "Angles", "Velocity", "Angular Velocity"]
    grouped_rmse = np.array([
        rmse_state_mean[0:3].mean(),
        rmse_state_mean[3:6].mean(),
        rmse_state_mean[6:9].mean(),
        rmse_state_mean[9:12].mean(),
    ])

    print_rmse_table(grouped_labels, grouped_rmse)

    horizon_rows = []
    for step in OVERALL_RMSE_STEPS:
        count = len(step_mse[step])
        rmse_val = np.sqrt(np.mean(step_mse[step])) if count > 0 else np.nan
        rmse_str = f"{rmse_val:.6f}" if count > 0 else "N/A"
        horizon_rows.append((f"{step}-step", rmse_str, f"{count}/{N_EVAL_EPISODES}"))
    print_table("Overall Horizon RMSE", ["Horizon", "RMSE", "Episodes Used"], horizon_rows)

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate(args.model_save_path)
