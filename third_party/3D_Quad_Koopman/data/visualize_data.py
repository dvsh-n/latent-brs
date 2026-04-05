import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- Root Path Setup ---
def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path

ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))
# -----------------------

from utils import resolve_path

# ==========================================
# CONFIGURATION
# ==========================================
# Choose the file to plot here
FILE_TO_PLOT = "data/train_data.pt" 
# FILE_TO_PLOT = "data/train_data.pt"

# How many trajectories to plot (to avoid overcrowding/slowness)
MAX_TRAJECTORIES = 25 
SHOW_OBSTACLE = False # Set to True to show the obstacle at [2.5, 2.5, 2.5]
# ==========================================

def visualize_3d_data(file_path):
    resolved_path = resolve_path(file_path)
    print(f"Loading data from: {resolved_path}")
    
    try:
        # Load data (handles CPU/GPU mismatch automatically)
        # Note: PyTorch 2.6+ defaults to weights_only=True, which fails for data files containing numpy arrays.
        # Since this is locally generated data, we can safely set weights_only=False.
        demos = torch.load(resolved_path, map_location=torch.device('cpu'), weights_only=False)
    except FileNotFoundError:
        print(f"Error: File {resolved_path} not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total trajectories in file: {len(demos)}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot a subset of trajectories
    num_to_plot = min(len(demos), MAX_TRAJECTORIES)
    indices = np.random.choice(len(demos), num_to_plot, replace=False)
    
    print(f"Plotting {num_to_plot} random trajectories...")

    for i, idx in enumerate(indices):
        demo = demos[idx]
        states = demo['states'] # Expected shape (12, N+1) or (N+1, 12)
        
        # Some generators might save states as (N, 12) or (12, N)
        # Check shape and transpose if necessary
        if states.shape[0] == 12:
            x = states[0, :]
            y = states[1, :]
            z = states[2, :]
        else:
            x = states[:, 0]
            y = states[:, 1]
            z = states[:, 2]
        
        # Plot trajectory
        ax.plot(x, y, z, alpha=0.4, linewidth=1)
        
        # Mark start and goal
        if i == 0:
            ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=50, label='Start')
            ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=50, label='Goal')
        else:
            ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=10, alpha=0.3)
            ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=10, alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Quadrotor Trajectories\nFile: {file_path} (Showing {num_to_plot} paths)')
    
    # Try to add obstacle if it was standard (from syn_data_gen.py)
    if SHOW_OBSTACLE:
        # Obstacle at [2.5, 2.5, 2.5] with radius 1.0
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        obs_x = 2.5 + 1.0 * np.cos(u) * np.sin(v)
        obs_y = 2.5 + 1.0 * np.sin(u) * np.sin(v)
        obs_z = 2.5 + 1.0 * np.cos(v)
        ax.plot_wireframe(obs_x, obs_y, obs_z, color="gray", alpha=0.2, label='Obstacle')

    ax.legend()
    
    # Equal aspect ratio
    # Calculate bounds
    all_x = np.concatenate([demos[idx]['states'][0,:] if demos[idx]['states'].shape[0] == 12 else demos[idx]['states'][:,0] for idx in indices])
    all_y = np.concatenate([demos[idx]['states'][1,:] if demos[idx]['states'].shape[0] == 12 else demos[idx]['states'][:,1] for idx in indices])
    all_z = np.concatenate([demos[idx]['states'][2,:] if demos[idx]['states'].shape[0] == 12 else demos[idx]['states'][:,2] for idx in indices])
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    
    
    # --- Print Control Bounds ---
    print("\n" + "="*30)
    print("CONTROL BOUNDS (All Trajectories)")
    print("="*30)
    
    all_controls = []
    for demo in demos:
        u = demo['controls'] # (4, N) or (N, 4)
        if u.shape[0] == 4:
            all_controls.append(u.T)
        else:
            all_controls.append(u)
    
    all_controls = np.concatenate(all_controls, axis=0)
    u_min = np.min(all_controls, axis=0)
    u_max = np.max(all_controls, axis=0)
    
    labels = ['u1 (Thrust)', 'u2 (Torque X)', 'u3 (Torque Y)', 'u4 (Torque Z)']
    for i, label in enumerate(labels):
        print(f"{label:15}: Min = {u_min[i]:8.4f}, Max = {u_max[i]:8.4f}")
    print("="*30 + "\n")

    plt.show()

if __name__ == "__main__":
    visualize_3d_data(FILE_TO_PLOT)

