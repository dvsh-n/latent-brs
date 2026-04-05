import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Root Path Setup ---
def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path

ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))
# -----------------------

from models import DeepKoopman
from utils import resolve_path, print_config
from eval.koop_eval import QuadrotorDatasetEval

# ====== CONFIGURATION ======
DATA_PATH = "data/random_data.pt"
MODEL_SAVE_PATH = "models/koop_quad3d_random.pt"
PLOT_SAVE_PATH = "test/latent_evolution_rand.png"
N_EVAL_EPISODES = 250
ROLLOUT = False

def get_latent_trajectories(model, dataset, n_episodes, device):
    """
    Collects latent trajectories for multiple episodes.
    """
    model.eval()
    all_latent_trajs = []
    
    with torch.no_grad():
        for i in range(min(n_episodes, len(dataset.states_data))):
            x_12d, u = dataset.get_raw_episode(i)
            x0_12d = x_12d[0].to(device)
            u = u.to(device)
            H = len(u)
            
            # 1. Lift initial state
            x0_norm = dataset.normalize(x0_12d.unsqueeze(0))
            z_k = model.encoder(x0_norm)
            
            # 2. Rollout in latent space
            z_traj = [z_k.cpu().numpy().flatten()]
            for k in range(H):
                u_k = u[k].unsqueeze(0)
                # Propagate latent state
                z_kp1 = model.A(z_k) + model.B(u_k)
                z_traj.append(z_kp1.cpu().numpy().flatten())
                z_k = z_kp1
            
            all_latent_trajs.append(np.array(z_traj))
            
    return all_latent_trajs

def get_encoded_latent_trajectories(model, dataset, n_episodes, device):
    """
    Encodes raw state trajectories directly into latent space (no rollout).
    """
    model.eval()
    all_latent_trajs = []
    
    with torch.no_grad():
        for i in range(min(n_episodes, len(dataset.states_data))):
            x_12d, _ = dataset.get_raw_episode(i)
            x_12d = x_12d.to(device)
            
            x_norm = dataset.normalize(x_12d)
            z_traj = model.encoder(x_norm)
            all_latent_trajs.append(z_traj.cpu().numpy())
            
    return all_latent_trajs

def plot_latent_evolution(all_trajs, latent_dim, save_path):
    """
    Plots all latent dimensions in subplots of a single big plot.
    Each subplot shows multiple rollouts.
    """
    # Determine grid size
    cols = int(np.ceil(np.sqrt(latent_dim)))
    rows = int(np.ceil(latent_dim / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    fig.suptitle("Koopman Generated", fontsize=20)
    axes_flat = axes.flatten()
    colors = plt.cm.jet(np.linspace(0, 1, len(all_trajs)))
    
    print("\n" + "="*40)
    print(f"LATENT STATE BOUNDS ({len(all_trajs)} Rollouts)")
    print("="*40)
    
    for d in range(latent_dim):
        ax = axes_flat[d]
        
        # Collect all values for this dimension to find bounds
        dim_values = np.concatenate([traj[:, d] for traj in all_trajs])
        d_min, d_max = np.min(dim_values), np.max(dim_values)
        
        for i, traj in enumerate(all_trajs):
            ax.plot(traj[:, d], alpha=0.6, linewidth=1, color=colors[i])
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Print bounds to console as well
        print(f"Latent Dimension {d+1:2}: Min = {d_min:8.4f}, Max = {d_max:8.4f}")
    
    print("="*40 + "\n")

    # Hide unused axes
    for d in range(latent_dim, len(axes_flat)):
        axes_flat[d].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"\nLatent evolution plot saved to: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = resolve_path(DATA_PATH)
    model_path = resolve_path(MODEL_SAVE_PATH)
    
    # Load model data
    try:
        model_data = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model_config = model_data.get('model_config', model_data.get('config'))
    latent_dim = model_config['latent_dim']
    
    # Load dataset
    dataset = QuadrotorDatasetEval(data_path)
    dataset.set_normalization_stats(model_data['normalization_stats'])
    
    # Initialize model
    model_params = {k: v for k, v in model_config.items() 
                    if k in ['state_dim', 'control_dim', 'latent_dim', 'hidden_width', 'depth', 'activation_fn']}
    
    # Handle activation function
    if isinstance(model_params.get('activation_fn'), str):
        from utils import get_activation_fn
        model_params['activation_fn'] = get_activation_fn(model_params['activation_fn'])

    model = DeepKoopman(**model_params).to(device)
    model.load_state_dict(model_data['state_dict'])
    print(f"Successfully loaded model and dataset.")
    
    if ROLLOUT:
        print(f"Collecting latent trajectories for {N_EVAL_EPISODES} rollouts...")
        trajs = get_latent_trajectories(model, dataset, N_EVAL_EPISODES, device)
    else:
        print(f"Encoding latent trajectories for {N_EVAL_EPISODES} episodes (no rollout)...")
        trajs = get_encoded_latent_trajectories(model, dataset, N_EVAL_EPISODES, device)
    
    plot_latent_evolution(trajs, latent_dim, resolve_path(PLOT_SAVE_PATH))

if __name__ == "__main__":
    main()
