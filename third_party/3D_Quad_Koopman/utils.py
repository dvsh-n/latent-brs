# utils.py
"""Shared utilities for Koopman training scripts."""
import argparse
from pathlib import Path
import torch.nn as nn


def str2bool(value):
    """Parse common string forms into booleans for argparse."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if value in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_embodiment_root(marker: str = ".root") -> Path:
    """
    Finds the root directory of the current embodiment by searching upwards
    for a marker file.
    """
    import inspect
    import os
    
    # Try to get the file of the caller
    frame = inspect.currentframe()
    while frame:
        filename = frame.f_code.co_filename
        if filename and os.path.exists(filename) and ".root" not in filename:
            caller_file = Path(filename).resolve()
            for parent in [caller_file.parent] + list(caller_file.parents):
                if (parent / marker).exists():
                    return parent
        frame = frame.f_back

    # Fallback to CWD
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / marker).exists():
            return parent
            
    # Final fallback: parent of this file
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> str:
    """
    Resolve a path relative to the embodiment root.
    If the path is already absolute, return it as-is.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    
    root = get_embodiment_root()
    return str(root / p)


def get_unique_path(path: str) -> str:
    """
    If path exists, append _1, _2, etc. until a non-existent path is found.
    """
    p = Path(path)
    if not p.exists():
        return str(p)
    
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return str(new_path)
        counter += 1


def get_activation_fn(name: str):
    """Convert activation name string to nn.Module class."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    return activations[name.lower()]


def get_activation_name(fn) -> str:
    """Convert activation nn.Module class to string name."""
    if fn == nn.ReLU or (isinstance(fn, type) and issubclass(fn, nn.ReLU)):
        return "relu"
    elif fn == nn.GELU or (isinstance(fn, type) and issubclass(fn, nn.GELU)):
        return "gelu"
    elif fn == nn.Tanh or (isinstance(fn, type) and issubclass(fn, nn.Tanh)):
        return "tanh"
    else:
        raise ValueError(f"Unknown activation function: {fn}")


def parse_train_args(defaults: dict, description: str = "Train Deep Koopman model"):
    """
    Parse command-line arguments for training, using provided defaults.
    
    Args:
        defaults: Dictionary containing default values for all arguments.
                  Keys should match argument names (without '--' prefix).
        description: Description for the argument parser.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=description)
    
    # --- Data and Model Paths ---
    parser.add_argument("--data_path", type=str, default=defaults.get("data_path"),
                        help="Path to training data")
    parser.add_argument("--model_save_path", type=str, default=defaults.get("model_save_path"),
                        help="Path to save trained model")
    parser.add_argument("--log_dir", type=str, default=defaults.get("log_dir"),
                        help="TensorBoard log directory")
    
    # --- Model Architecture ---
    parser.add_argument("--state_dim", type=int, default=defaults.get("state_dim"),
                        help="State dimension")
    parser.add_argument("--control_dim", type=int, default=defaults.get("control_dim"),
                        help="Control dimension")
    
    # Standard latent dim (for koop_train.py)
    if "latent_dim" in defaults:
        parser.add_argument("--latent_dim", type=int, default=defaults.get("latent_dim"),
                            help="Dimension of the learned latent space")
    
    # Split latent dims (for koop_train_split.py)
    if "latent_dim_pos" in defaults:
        parser.add_argument("--latent_dim_pos", type=int, default=defaults.get("latent_dim_pos"),
                            help="Latent dimension for position branch")
    if "latent_dim_rest" in defaults:
        parser.add_argument("--latent_dim_rest", type=int, default=defaults.get("latent_dim_rest"),
                            help="Latent dimension for rest of state branch")
    
    # Base latent dim for xyz supervision (for koop_train_xyz.py)
    if "base_latent_dim" in defaults:
        parser.add_argument("--base_latent_dim", type=int, default=defaults.get("base_latent_dim"),
                            help="Base dimension of the learned latent space (3 will be added for xyz)")
    if "embedding_dim" in defaults:
        parser.add_argument("--embedding_dim", type=int, default=defaults.get("embedding_dim"),
                            help="Dimension of the learned encoder embedding")
    
    # Latent multiplier + hidden dim (for koop_train_2.py)
    if "n_mult" in defaults:
        parser.add_argument("--n_mult", type=int, default=defaults.get("n_mult"),
                            help="Latent multiplier (latent_dim = state_dim + state_dim * n_mult)")
    if "hidden_dim" in defaults:
        parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim"),
                            help="Hidden dimension for residual encoder architecture")
    
    parser.add_argument("--hidden_width", type=int, default=defaults.get("hidden_width"),
                        help="Width of hidden layers in encoder/decoder")
    parser.add_argument("--hidden_depth", type=int, default=defaults.get("hidden_depth"),
                        help="Number of hidden layers in encoder/decoder")
    parser.add_argument("--activation", type=str, default=defaults.get("activation"),
                        choices=["relu", "gelu", "tanh"],
                        help="Activation function")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=defaults.get("lr"),
                        help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=defaults.get("min_lr"),
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay"),
                        help="L2 regularization weight decay")
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs"),
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size"),
                        help="Batch size")
    parser.add_argument("--multi_step_horizon", type=int, default=defaults.get("multi_step_horizon"),
                        help="Number of steps for multi-step prediction loss")
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 16),
                        help="Number of dataloader workers")
    if "enable_normalization" in defaults:
        parser.add_argument(
            "--enable_normalization",
            type=str2bool,
            nargs="?",
            const=True,
            default=defaults.get("enable_normalization"),
            help="Whether to min-max normalize states before training",
        )
    
    # --- Loss Function Weights ---
    parser.add_argument("--lambda_recon", type=float, default=defaults.get("lambda_recon"),
                        help="Weight for reconstruction loss")
    parser.add_argument("--lambda_linear", type=float, default=defaults.get("lambda_linear"),
                        help="Weight for linear dynamics loss")
    if "lambda_state" in defaults:
        parser.add_argument("--lambda_state", type=float, default=defaults.get("lambda_state"),
                            help="Weight for state prediction loss")
    if "lambda_latent" in defaults:
        parser.add_argument("--lambda_latent", type=float, default=defaults.get("lambda_latent"),
                            help="Weight for latent consistency loss")
    
    # XYZ latent supervision (for koop_train_xyz.py)
    if "lambda_latent_xyz" in defaults:
        parser.add_argument("--lambda_latent_xyz", type=float, default=defaults.get("lambda_latent_xyz"),
                            help="Weight for XYZ latent supervision loss")
    
    # Paper-style loss weights (for koop_train_2.py)
    if "beta" in defaults:
        parser.add_argument("--beta", type=float, default=defaults.get("beta"),
                            help="Temporal discount factor for multi-step losses")
    if "w_cov" in defaults:
        parser.add_argument("--w_cov", type=float, default=defaults.get("w_cov"),
                            help="Weight for covariance regularization loss")
    if "w_ctrl" in defaults:
        parser.add_argument("--w_ctrl", type=float, default=defaults.get("w_ctrl"),
                            help="Weight for inverse control loss")
    
    return parser.parse_args()


def print_config(config: dict, title: str = "Configuration"):
    """Nicely print a configuration dictionary."""
    print("\n" + "="*45)
    print(f"      {title}")
    print("="*45)
    for k, v in config.items():
        if isinstance(v, float):
            print(f"{k:<20s} | {v:.2e}" if v < 1e-3 else f"{k:<20s} | {v:.6f}")
        else:
            print(f"{k:<20s} | {v}")
    print("="*45 + "\n")


def _format_table_value(value):
    """Compact value formatting for terminal tables."""
    # Torch scalar/tensor-like handling without importing torch directly
    if hasattr(value, "numel") and callable(value.numel):
        try:
            if value.numel() == 1 and hasattr(value, "item"):
                return f"{value.item():.6g}"
            if hasattr(value, "shape"):
                return f"Tensor(shape={tuple(value.shape)})"
        except Exception:
            pass
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, type):
        return value.__name__
    if callable(value) and hasattr(value, "__name__"):
        return value.__name__
    return str(value)


def print_table(title: str, headers: list[str], rows: list[tuple]):
    """Print an ASCII table with a title."""
    if not rows:
        return
    n_cols = len(headers)
    widths = []
    for i in range(n_cols):
        col_values = [str(headers[i])] + [str(row[i]) for row in rows]
        widths.append(max(len(v) for v in col_values))

    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(f"\n{title}")
    print(border)
    header_row = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(n_cols)) + " |"
    print(header_row)
    print(border)
    for row in rows:
        row_str = "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(n_cols)) + " |"
        print(row_str)
    print(border)


def print_config_table(config: dict, title: str = "Configuration"):
    """Print configuration dictionary as a two-column ASCII table."""
    rows = [(str(k), _format_table_value(v)) for k, v in config.items()]
    print_table(title, ["Key", "Value"], rows)


def print_rmse_table(state_labels: list[str], rmse_values, title: str = "Koopman 3D Performance Metrics (RMSE)"):
    """Print state-wise RMSE as a table."""
    rows = [(label, f"{value:.6f}") for label, value in zip(state_labels, rmse_values)]
    print_table(title, ["State", "RMSE"], rows)
