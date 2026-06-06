#!/usr/bin/env python3
"""Compare obstacle classifiers against geometric labels on saved rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.plan import benchmark_rope_hard as hard
from rope.safety.check_rollout_obstacle_geometry import load_obstacle_rule, score_task_states
from rope.safety.compat import register_legacy_checkpoint_aliases
from rope.safety.obstacle_classifier import (
    ObstacleMLP,
    imagenet_pixel_stats,
    load_config,
    load_world_model,
    preprocess_pixels,
)
from rope.shared.lab_env import LabEnv

DEFAULT_OLD = REPO_ROOT / "rope/safety/obs_net_dino_like/23af9d1d3eda01ce/model.pt"
DEFAULT_NEW = REPO_ROOT / "rope/safety/obs_net_latent_safety_dino_20260523_142517/8ab92febdd128ee4/model.pt"
DEFAULT_OBSTACLE_DATA = REPO_ROOT / "rope/safety/obstacle_data/obstacle_classifier_data.pt"


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(value).expanduser()
    return path.parent.name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--obstacle-data", type=Path, default=DEFAULT_OBSTACLE_DATA)
    parser.add_argument("--branch", choices=("nominal", "hj_filtered", "all"), default="nominal")
    parser.add_argument("--model", action="append", default=None, help="Classifier model as name=path. Repeatable.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_checkpoint(path_string: str, model_dir: Path) -> Path:
    path = Path(path_string).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = model_dir / path.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve world-model checkpoint from classifier artifact: {path_string}")


class ClassifierBundle:
    def __init__(self, *, name: str, path: Path, device: torch.device) -> None:
        self.name = name
        self.path = path.expanduser().resolve()
        artifact = torch.load(self.path, map_location="cpu", weights_only=False)
        self.artifact = artifact
        self.input_dim = int(artifact["input_dim"])
        self.threshold = float(artifact.get("conformal_safe_score_threshold", artifact.get("base_decision_threshold", 0.0)))
        self.model = ObstacleMLP(
            self.input_dim,
            int(artifact["hidden_dim"]),
            int(artifact["depth"]),
            float(artifact["dropout"]),
            head_style=str(artifact.get("head_style", "postnorm-gelu")),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.feature_mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device)
        self.feature_std = torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)

    @torch.no_grad()
    def score_latents(self, latents: torch.Tensor) -> torch.Tensor:
        features = (latents[:, : self.input_dim] - self.feature_mean) / self.feature_std
        return self.model(features).reshape(-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "input_dim": self.input_dim,
            "hidden_dim": int(self.artifact["hidden_dim"]),
            "depth": int(self.artifact["depth"]),
            "dropout": float(self.artifact["dropout"]),
            "head_style": str(self.artifact.get("head_style", "postnorm-gelu")),
            "threshold": self.threshold,
            "loss": self.artifact.get("loss", self.artifact.get("cache_config", {}).get("loss")),
        }


def load_metrics_dataset(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Path(str(metrics["dataset_path"])).expanduser().resolve()


def replay_case_frames(
    *,
    summary_path: Path,
    dataset_path: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_idx = int(summary["episode_idx"])
    start_step = int(summary["start_step"])
    actions = np.asarray(summary.get("executed_actions_raw", []), dtype=np.float32).reshape(-1, 3)
    episode = hard.load_dataset_episode(dataset_path, episode_idx)
    task_target_np = np.asarray(episode["task_target"], dtype=np.float32)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    control_np = np.asarray(episode["control"], dtype=np.float32)
    time_np = np.asarray(episode["time"], dtype=np.float32)
    camera_id = env.model.camera(str(episode["camera"])).id
    control_decimation = int(episode["control_decimation"])
    control_timestep = float(episode["control_timestep"])

    frame, info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[start_step],
        qvel=qvel_np[start_step],
        control=control_np[start_step],
        task_target=task_target_np[start_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[start_step, 0]),
    )
    frames = [frame]
    task_states = [np.asarray(info["task_target"], dtype=np.float32)]
    for action in actions:
        elapsed_time = float(info["time"][0]) + control_timestep
        frame, info = hard.step_env_with_action(
            env,
            renderer,
            action=action,
            control_decimation=control_decimation,
            camera_id=camera_id,
            elapsed_time=elapsed_time,
        )
        frames.append(frame)
        task_states.append(np.asarray(info["task_target"], dtype=np.float32))
    case_info = {
        "summary_path": str(summary_path),
        "episode_idx": episode_idx,
        "label": str(summary.get("label", summary_path.parent.parent.name)),
        "steps": int(actions.shape[0]),
        "classifier_reported_safety_violation": bool(summary.get("safety_violation", False)),
        "classifier_min_l": summary.get("min_l"),
    }
    return np.stack(frames, axis=0), np.stack(task_states, axis=0), case_info


@torch.no_grad()
def encode_frames(
    frames: np.ndarray,
    *,
    world_model: torch.nn.Module,
    img_size: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    latents: list[torch.Tensor] = []
    for start in range(0, frames.shape[0], batch_size):
        batch = preprocess_pixels(
            frames[start : start + batch_size],
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        encoded = world_model.encoder(batch, interpolate_pos_encoding=True)
        latent = world_model.projector(encoded.last_hidden_state[:, 0])
        latents.append(latent.detach().cpu())
    return torch.cat(latents, dim=0)


def classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    pred_unsafe = np.asarray(scores, dtype=np.float32) <= float(threshold)
    tp = int(np.sum(pred_unsafe & labels))
    fp = int(np.sum(pred_unsafe & ~labels))
    tn = int(np.sum(~pred_unsafe & ~labels))
    fn = int(np.sum(~pred_unsafe & labels))
    total = int(labels.size)
    unsafe_count = int(np.sum(labels))
    safe_count = total - unsafe_count
    return {
        "threshold": float(threshold),
        "total_frames": total,
        "geometry_unsafe_frames": unsafe_count,
        "pred_unsafe_frames": int(np.sum(pred_unsafe)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": float((tp + tn) / total) if total else None,
        "recall_on_geometry_unsafe": float(tp / unsafe_count) if unsafe_count else None,
        "specificity_on_geometry_safe": float(tn / safe_count) if safe_count else None,
        "precision_pred_unsafe": float(tp / max(tp + fp, 1)),
        "false_negative_rate": float(fn / unsafe_count) if unsafe_count else None,
        "score_min": float(np.min(scores)) if scores.size else None,
        "score_max": float(np.max(scores)) if scores.size else None,
        "unsafe_score_mean": float(np.mean(scores[labels])) if unsafe_count else None,
        "safe_score_mean": float(np.mean(scores[~labels])) if safe_count else None,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    dataset_path = load_metrics_dataset(run_dir, args.dataset_path)
    device = require_device(args.device)
    register_legacy_checkpoint_aliases()

    model_specs = [parse_model_spec(item) for item in args.model] if args.model else [
        ("old_dino_like", DEFAULT_OLD),
        ("new_latent_safety_dino", DEFAULT_NEW),
    ]
    bundles = [ClassifierBundle(name=name, path=path, device=device) for name, path in model_specs]
    cache_config = bundles[0].artifact.get("cache_config", {})
    model_dir = Path(cache_config.get("model_dir", REPO_ROOT / "rope/models/mlpdyn_noshadow_ft")).expanduser()
    if not model_dir.is_dir():
        model_dir = REPO_ROOT / "rope/models/mlpdyn_noshadow_ft"
    model_dir = model_dir.resolve()
    checkpoint = resolve_checkpoint(str(cache_config["checkpoint_path"]), model_dir)
    config = load_config(model_dir)
    img_size = int(cache_config.get("img_size", config.get("img_size", 224)))
    world_model = load_world_model(checkpoint, device)

    rule = load_obstacle_rule(args.obstacle_data)
    branches = ["nominal", "hj_filtered"] if args.branch == "all" else [str(args.branch)]
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "obstacle_data": str(args.obstacle_data.expanduser().resolve()),
        "world_model_checkpoint": str(checkpoint),
        "classifiers": [bundle.metadata() for bundle in bundles],
        "branches": {},
    }

    env = LabEnv()
    with mujoco.Renderer(env.model, height=224, width=224) as renderer:
        for branch in branches:
            summary_paths = sorted((run_dir / branch).glob("case_*/summary.json"))
            all_labels: list[np.ndarray] = []
            all_scores: dict[str, list[np.ndarray]] = {bundle.name: [] for bundle in bundles}
            cases: list[dict[str, Any]] = []
            for summary_path in tqdm(summary_paths, desc=f"Scoring {branch}", unit="case"):
                frames, task_states, case_info = replay_case_frames(
                    summary_path=summary_path,
                    dataset_path=dataset_path,
                    env=env,
                    renderer=renderer,
                )
                geometry = score_task_states(task_states, rule)
                labels = np.asarray(geometry["labels"], dtype=np.int64)
                latents = encode_frames(
                    frames,
                    world_model=world_model,
                    img_size=img_size,
                    device=device,
                    batch_size=int(args.batch_size),
                ).to(device)
                case_scores: dict[str, Any] = {}
                for bundle in bundles:
                    scores = bundle.score_latents(latents).detach().cpu().numpy().astype(np.float32)
                    all_scores[bundle.name].append(scores)
                    case_scores[bundle.name] = {
                        **classification_metrics(scores, labels, bundle.threshold),
                        "scores": scores.tolist(),
                    }
                all_labels.append(labels)
                cases.append(
                    {
                        **case_info,
                        "geometry_violates": bool(geometry["violates"]),
                        "geometry_first_unsafe_step": geometry["first_unsafe_step"],
                        "geometry_unsafe_step_count": int(geometry["unsafe_step_count"]),
                        "geometry_min_clearance": geometry["min_clearance"],
                        "geometry_labels": labels.tolist(),
                        "classifier_metrics": case_scores,
                    }
                )
            labels_cat = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
            aggregate_models = {}
            for bundle in bundles:
                scores_cat = np.concatenate(all_scores[bundle.name], axis=0) if all_scores[bundle.name] else np.zeros((0,))
                aggregate_models[bundle.name] = classification_metrics(scores_cat, labels_cat, bundle.threshold)
            result["branches"][branch] = {
                "aggregate": {
                    "num_cases": int(len(cases)),
                    "geometry_violation_rate": float(np.mean([case["geometry_violates"] for case in cases]) * 100.0)
                    if cases
                    else 0.0,
                    "classifiers": aggregate_models,
                },
                "cases": cases,
            }

    output = args.output.expanduser().resolve() if args.output else run_dir / "classifier_geometry_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"output": str(output), "branches": result["branches"]}), indent=2)[:12000])


if __name__ == "__main__":
    main()
