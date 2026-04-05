import sys
import argparse
from pathlib import Path


def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import DeepKoopmanLinDec
from utils import resolve_path, print_config_table, print_rmse_table, print_table, get_activation_fn


DATA_PATH = "data/random_data.pt"
MODEL_SAVE_PATH = "models/koop_quad3d_lindec_dagger_1.pt"
PLOT_SAVE_DIR = "eval/eval_plots_lindec"
N_EVAL_EPISODES = 25
MAX_TIMESTEP = 100
PURE_LATENT_ROLLOUT = True
OVERALL_RMSE_STEPS = [1, 50, 100, 150]


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate linear-decoder Deep Koopman model")
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default=DATA_PATH,
                        help="Path to evaluation dataset")
    parser.add_argument("--plot_save_dir", type=str, default=PLOT_SAVE_DIR,
                        help="Directory to save rollout plots")
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max_timestep", type=int, default=MAX_TIMESTEP,
                        help="Maximum number of control steps to evaluate per episode")
    parser.add_argument(
        "--pure_latent_rollout",
        action=argparse.BooleanOptionalAction,
        default=PURE_LATENT_ROLLOUT,
        help="Use pure latent rollout instead of recurrent re-lifting",
    )
    return parser.parse_args()


class QuadrotorDatasetEval(torch.nn.Module):
    """Evaluation dataset with optional normalization preprocessing."""

    def __init__(self, data_path):
        super().__init__()
        demos = torch.load(data_path, weights_only=False)

        self.states_data = []
        self.controls_data = []
        for demo in demos:
            self.states_data.append(torch.tensor(demo["states"].T, dtype=torch.float32))
            self.controls_data.append(torch.tensor(demo["controls"].T, dtype=torch.float32))

        self.min = None
        self.max = None
        self.range = None

    def set_normalization_stats(self, stats: dict):
        self.min = stats["min"]
        self.max = stats["max"]
        self.range = stats.get("range", self.max - self.min)
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        return 2 * (x - min_dev) / range_dev - 1

    def denormalize(self, x):
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        return (x + 1) / 2 * range_dev + min_dev

    def preprocess(self, x, enable_normalization: bool):
        return self.normalize(x) if enable_normalization else x

    def postprocess(self, x, enable_normalization: bool):
        return self.denormalize(x) if enable_normalization else x

    def get_raw_episode(self, idx: int):
        return self.states_data[idx], self.controls_data[idx]


def rollout_latent(model, x0_raw, u_raw, dataset, device, enable_normalization):
    model.eval()
    with torch.no_grad():
        H = len(u_raw)
        x0_raw = x0_raw.to(device)
        u_raw = u_raw.to(device)

        x0_proc = dataset.preprocess(x0_raw.unsqueeze(0), enable_normalization)
        z_k = model.encoder(x0_proc)

        x_hat_proc_list = [model.C(z_k).squeeze(0)]
        for k in range(H):
            u_k = u_raw[k].unsqueeze(0)
            z_k = model.A(z_k) + model.B(u_k)
            x_hat_proc_list.append(model.C(z_k).squeeze(0))

        x_hat_proc = torch.stack(x_hat_proc_list, dim=0)
        x_hat = dataset.postprocess(x_hat_proc, enable_normalization)
        return x_hat.cpu().numpy()


def rollout_recurrent(model, x0_raw, u_raw, dataset, device, enable_normalization):
    model.eval()
    with torch.no_grad():
        H = len(u_raw)
        x0_raw = x0_raw.to(device)
        u_raw = u_raw.to(device)

        x_hat_raw_list = [x0_raw]
        for k in range(H):
            x_k_raw = x_hat_raw_list[-1].unsqueeze(0)
            x_k_proc = dataset.preprocess(x_k_raw, enable_normalization)
            z_k = model.encoder(x_k_proc)
            u_k = u_raw[k].unsqueeze(0)
            z_kp1 = model.A(z_k) + model.B(u_k)
            x_kp1_proc = model.C(z_kp1)
            x_kp1_raw = dataset.postprocess(x_kp1_proc, enable_normalization).squeeze(0)
            x_hat_raw_list.append(x_kp1_raw)

        x_hat = torch.stack(x_hat_raw_list, dim=0)
        return x_hat.cpu().numpy()


def calculate_rmse(x_traj: np.ndarray, x_hat_traj: np.ndarray) -> np.ndarray:
    T = min(len(x_traj), len(x_hat_traj))
    err = x_traj[:T] - x_hat_traj[:T]
    return np.sqrt(np.mean(err ** 2, axis=0))


def plot_rollout(x_traj: np.ndarray, x_hat_traj: np.ndarray, episode_idx: int, plot_save_dir: str):
    labels = [
        "Pos X (m)", "Pos Y (m)", "Pos Z (m)",
        "Angle Psi (rad)", "Angle Theta (rad)", "Angle Phi (rad)",
        "Vel X (m/s)", "Vel Y (m/s)", "Vel Z (m/s)",
        "Ang Vel p (rad/s)", "Ang Vel q (rad/s)", "Ang Vel r (rad/s)",
    ]

    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    fig.suptitle(f"Linear-Decoder Koopman Rollout vs. Ground Truth (Episode {episode_idx})", fontsize=20)
    outer_grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.4)

    axs_pos = outer_grid[0, 0].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_pos[0].set_title("Position", fontsize=16)
    axs_ang = outer_grid[0, 1].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_ang[0].set_title("Euler Angles", fontsize=16)
    axs_vel = outer_grid[1, 0].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_vel[0].set_title("Linear Velocity", fontsize=16)
    axs_ang_vel = outer_grid[1, 1].subgridspec(3, 1, hspace=0.1).subplots(sharex=True)
    axs_ang_vel[0].set_title("Angular Velocity", fontsize=16)

    all_axs = [*axs_pos, *axs_ang, *axs_vel, *axs_ang_vel]
    T = min(len(x_traj), len(x_hat_traj))
    time_ax = np.arange(T)

    for i, ax in enumerate(all_axs):
        ax.plot(time_ax, x_traj[:T, i], label="Ground Truth", color="blue", linewidth=2)
        ax.plot(time_ax, x_hat_traj[:T, i], label="Koopman Rollout", color="red", linestyle="--", linewidth=2)
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle=":")

    for ax in [axs_vel[-1], axs_ang_vel[-1]]:
        ax.set_xlabel("Time Step", fontsize=12)

    handles, leg_labels = all_axs[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", fontsize=14, bbox_to_anchor=(0.95, 0.98))

    Path(plot_save_dir).mkdir(exist_ok=True)
    save_path = Path(plot_save_dir) / f"episode_{episode_idx}.png"
    plt.savefig(save_path)
    plt.close(fig)


def evaluate(model_save_path_arg=None, data_path_arg=None, plot_save_dir_arg=None,
             n_eval_episodes=N_EVAL_EPISODES, max_timestep=MAX_TIMESTEP,
             pure_latent_rollout=PURE_LATENT_ROLLOUT):
    data_path = resolve_path(data_path_arg or DATA_PATH)
    model_save_path = resolve_path(model_save_path_arg or MODEL_SAVE_PATH)
    plot_save_dir = resolve_path(plot_save_dir_arg or PLOT_SAVE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_data = torch.load(model_save_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_save_path}.")
        print("Please run 'koop_train_lindec.py' first.")
        return

    dataset = QuadrotorDatasetEval(data_path)
    if "normalization_stats" not in model_data:
        print("ERROR: 'normalization_stats' not found in model file.")
        return
    dataset.set_normalization_stats(model_data["normalization_stats"])

    model_config = model_data.get("model_config", model_data.get("config"))
    training_config = model_data.get("training_config", {})
    enable_normalization = training_config.get("enable_normalization", True)

    print_config_table(model_config, title="Model Configuration")
    if training_config:
        print_config_table(training_config, title="Training Configuration")

    if isinstance(model_config.get("activation_fn"), str):
        model_config["activation_fn"] = get_activation_fn(model_config["activation_fn"])

    model_params = {
        k: v for k, v in model_config.items()
        if k in ["state_dim", "control_dim", "latent_dim", "hidden_width", "depth", "activation_fn"]
    }
    model = DeepKoopmanLinDec(**model_params).to(device)
    model.load_state_dict(model_data["state_dict"])

    n_eval_episodes = min(n_eval_episodes, len(dataset.states_data))
    state_errs = []
    step_mse = {step: [] for step in OVERALL_RMSE_STEPS}

    pbar = tqdm(range(n_eval_episodes), desc="Evaluating episodes")
    for ep_id in pbar:
        x_traj, u = dataset.get_raw_episode(ep_id)
        if max_timestep is not None and max_timestep > 0:
            u = u[:max_timestep]
            x_traj = x_traj[:max_timestep + 1]
        x0 = x_traj[0]

        if pure_latent_rollout:
            x_hat_traj = rollout_latent(model, x0, u, dataset, device, enable_normalization)
        else:
            x_hat_traj = rollout_recurrent(model, x0, u, dataset, device, enable_normalization)

        x_traj_np = x_traj.numpy()
        plot_rollout(x_traj_np, x_hat_traj, ep_id, plot_save_dir)

        rmse_state = calculate_rmse(x_traj_np, x_hat_traj)
        state_errs.append(rmse_state)

        T = min(len(x_traj_np), len(x_hat_traj))
        err = x_traj_np[:T] - x_hat_traj[:T]
        for step in OVERALL_RMSE_STEPS:
            if step < T:
                step_mse[step].append(float(np.mean(err[step] ** 2)))

        pbar.set_postfix(mean_pos_err=f"{np.mean(np.array(state_errs)[:, :3]):.4f}")

    state_errs_np = np.array(state_errs)
    rmse_state_mean = state_errs_np.mean(axis=0)

    grouped_labels = ["Position", "Angles", "Velocity", "Angular Velocity"]
    grouped_rmse = np.array([
        rmse_state_mean[0:3].mean(),
        rmse_state_mean[3:6].mean(),
        rmse_state_mean[6:9].mean(),
        rmse_state_mean[9:12].mean(),
    ])
    print_rmse_table(grouped_labels, grouped_rmse, title="Linear-Decoder Koopman 3D Performance Metrics (RMSE)")

    horizon_rows = []
    for step in OVERALL_RMSE_STEPS:
        count = len(step_mse[step])
        rmse_val = np.sqrt(np.mean(step_mse[step])) if count > 0 else np.nan
        rmse_str = f"{rmse_val:.6f}" if count > 0 else "N/A"
        horizon_rows.append((f"{step}-step", rmse_str, f"{count}/{n_eval_episodes}"))
    print_table("Overall Horizon RMSE", ["Horizon", "RMSE", "Episodes Used"], horizon_rows)


if __name__ == "__main__":
    args = parse_eval_args()
    evaluate(
        model_save_path_arg=args.model_save_path,
        data_path_arg=args.data_path,
        plot_save_dir_arg=args.plot_save_dir,
        n_eval_episodes=args.n_eval_episodes,
        max_timestep=args.max_timestep,
        pure_latent_rollout=args.pure_latent_rollout,
    )
