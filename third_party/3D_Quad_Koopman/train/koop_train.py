# koop_train.py
import sys
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import DeepKoopman
from utils import parse_train_args, get_activation_fn, get_activation_name, resolve_path, print_config, get_unique_path

# ====== CONFIGURATION & HYPERPARAMETERS ======
# --- Data and Model Paths (relative to embodiment root) ---
DATA_PATH = "data/random_data.pt"
MODEL_SAVE_PATH = "models/koop_quad3d.pt"
LOG_DIR = "runs/koop_quad3d"

# --- Model Architecture ---
STATE_DIM = 12       # [px, py, pz, psi, th, phi, vx, vy, vz, p, q, r]
CONTROL_DIM = 4      # [u1, u2, u3, u4]
LATENT_DIM = 24      # Dimension of the learned latent space (N)
HIDDEN_WIDTH = 64    # Width of hidden layers in encoder/decoder
HIDDEN_DEPTH = 3     # Number of hidden layers in encoder/decoder
ACTIVATION_FN = nn.GELU  # Activation function class

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7   # The lowest learning rate the scheduler will decay to
WEIGHT_DECAY = 1e-6        # L2 Regularization
EPOCHS = 15
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 20    # Number of steps for multi-step prediction loss
NUM_WORKERS = 20

# --- Loss Function Weights (λ values) ---
LAMBDA_RECON = 1.0     # Reconstruction of the 12D state vector
LAMBDA_LINEAR = 1.0    # Koopman linear dynamics in latent space

# ====== DATA HANDLING ======
class QuadrotorDataset(Dataset):
    """Custom dataset to handle pre-processing of 3D quadrotor trajectories."""
    def __init__(self, data_path, multi_step_horizon=1):
        # Load the list of demo dictionaries
        demos = torch.load(data_path, weights_only=False)
        
        self.states = []
        self.controls = []

        # Iterate through each demo, transpose, and append
        for demo in demos:
            # demo['states'] is [12, N+1], demo['controls'] is [4, N]
            # We transpose to get [N+1, 12] and [N, 4]
            # Convert numpy arrays from syn_gen.py to torch tensors
            self.states.append(torch.tensor(demo['states'].T, dtype=torch.float32))
            self.controls.append(torch.tensor(demo['controls'].T, dtype=torch.float32))

        self.multi_step_horizon = multi_step_horizon
        self.x_k, self.u_k, self.x_kp1 = self._process_data()
        self._calculate_normalization_stats()

    def _process_data(self):
        """Transforms raw state trajectories into learning-ready (k, k+1) tuples."""
        x_k_list, u_k_list, x_kp1_list = [], [], []
        
        # self.states and self.controls are now lists of tensors
        for episode_states, episode_controls in zip(self.states, self.controls):
            
            # The state is already in the desired 12D format. No transformation needed.
            transformed_states = episode_states
            
            # Align data into (x_k, u_k, x_{k+1}) tuples
            # episode_controls is [N, 4]
            # transformed_states is [N+1, 12]
            x_k_list.append(transformed_states[:-1]) # [N, 12]
            u_k_list.append(episode_controls)        # [N, 4]
            x_kp1_list.append(transformed_states[1:])  # [N, 12]

        # Calculate episode lengths
        episode_lengths = [len(ep) for ep in u_k_list]
        x_k, u_k, x_kp1 = (torch.cat(x_k_list), torch.cat(u_k_list), torch.cat(x_kp1_list))
        
        # Build valid indices for multi-step horizon sampling
        self.valid_indices = []
        current_start_index = 0
        M = self.multi_step_horizon
        for length in episode_lengths:
            # Valid start indices in this episode must be <= length - M
            valid_end_in_episode = length - M 
            for i in range(valid_end_in_episode + 1): 
                self.valid_indices.append(current_start_index + i)
            current_start_index += length # Move to the start of the next episode
        
        return (x_k, u_k, x_kp1)

    def _calculate_normalization_stats(self):
        """Calculate min-max stats for all 12 state variables."""
        # We will normalize all 12 state variables
        self.min = self.x_k.min(dim=0).values
        self.max = self.x_k.max(dim=0).values
        # Add a small epsilon to avoid division by zero if min == max
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-6 


    def normalize(self, x):
        """Applies min-max normalization to the full state vector."""
        x_norm = x.clone()
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        
        # Apply normalization to all dimensions
        x_norm = 2 * (x_norm - min_dev) / range_dev - 1
        return x_norm

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_start_idx = self.valid_indices[idx]
        
        M = self.multi_step_horizon
        
        # Get the sequence of data
        x_k = self.x_k[actual_start_idx]
        u_seq = self.u_k[actual_start_idx : actual_start_idx + M]
        x_kp1_seq = self.x_kp1[actual_start_idx : actual_start_idx + M] 
        
        # Normalize the states
        x_k_norm = self.normalize(x_k)
        x_kp1_seq_norm = self.normalize(x_kp1_seq)

        return (x_k_norm, u_seq, x_kp1_seq_norm)

# ====== TRAINING LOOP ======
def train(args):
    """Main training loop."""
    # Resolve paths relative to embodiment root
    data_path = resolve_path(args.data_path)
    model_save_path = resolve_path(args.model_save_path)
    log_dir = resolve_path(args.log_dir)
    
    # Ensure unique paths for saving and logging
    model_save_path = get_unique_path(model_save_path)
    log_dir = get_unique_path(log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    activation_fn = get_activation_fn(args.activation)

    writer = SummaryWriter(log_dir)
    dataset = QuadrotorDataset(data_path, args.multi_step_horizon)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    model_config = {
        'state_dim': args.state_dim,
        'control_dim': args.control_dim,
        'latent_dim': args.latent_dim,
        'hidden_width': args.hidden_width,
        'depth': args.hidden_depth,
        'activation_fn': activation_fn,
    }
    
    model = DeepKoopman(**model_config).to(device)
    
    # Initialize A to identity, B to small random values
    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader), 
        eta_min=args.min_lr
    )
    
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting training for Deep Koopman model (3D Quadrotor)")
    print_config(model_config, title="Model Configuration")
    
    training_config = {
        'lr': args.lr,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'multi_step_horizon': args.multi_step_horizon,
        'lambda_recon': args.lambda_recon,
        'lambda_linear': args.lambda_linear,
    }
    print_config(training_config, title="Training Configuration")
    
    global_step = 0
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for epoch in pbar:
        for x_k, u_seq, x_kp1_seq in dataloader:
            x_k, u_seq, x_kp1_seq = (x_k.to(device), u_seq.to(device), x_kp1_seq.to(device))

            # --- Main Data Forward Pass ---
            (
                x_k_hat,
                x_target_seq_hat,
                z_pred_seq,
                x_pred_seq_hat,
                z_target_seq,
            ) = model(x_k, u_seq, x_kp1_seq)

            # --- Loss Calculation ---
            # Autoencoder components: state recon
            loss_recon_k = nn.functional.mse_loss(x_k_hat, x_k)
            loss_recon_seq = nn.functional.mse_loss(x_target_seq_hat, x_kp1_seq)
            loss_autoencoder = loss_recon_k + loss_recon_seq

            # Dynamics components: latent prediction + state prediction
            loss_latent_predict = nn.functional.mse_loss(z_pred_seq, z_target_seq)
            loss_state_predict = nn.functional.mse_loss(x_pred_seq_hat, x_kp1_seq)
            loss_dynamics = loss_latent_predict + loss_state_predict

            # Total weighted loss
            loss = (args.lambda_recon * loss_autoencoder +
                    args.lambda_linear * loss_dynamics)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # --- Logging ---
            if global_step % 10 == 0:
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Autoencoder', loss_autoencoder.item(), global_step)
                writer.add_scalar('Loss/Dynamics', loss_dynamics.item(), global_step)
                writer.add_scalar('Hyperparams/Learning_Rate', scheduler.get_last_lr()[0], global_step)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

    writer.close()
    
    # Store activation as string for serialization
    model_config_save = model_config.copy()
    model_config_save['activation_fn'] = args.activation
    
    # --- Save Model and Full Training Configuration ---
    save_data = {
        'model_config': model_config_save,
        'training_config': training_config,
        'state_dict': model.state_dict(),
        'normalization_stats': {
            'min': dataset.min, 
            'max': dataset.max,
            'range': dataset.range
        }
    }
    torch.save(save_data, model_save_path)
    
    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    # Build defaults from configuration constants
    defaults = {
        "data_path": DATA_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "log_dir": LOG_DIR,
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "hidden_depth": HIDDEN_DEPTH,
        "activation": get_activation_name(ACTIVATION_FN),
        "lr": LEARNING_RATE,
        "min_lr": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "multi_step_horizon": MULTI_STEP_HORIZON,
        "num_workers": NUM_WORKERS,
        "lambda_recon": LAMBDA_RECON,
        "lambda_linear": LAMBDA_LINEAR,
    }
    args = parse_train_args(defaults, description="Train Deep Koopman model for 3D Quadrotor")
    train(args)
