#!/usr/bin/env python3
"""Create one-step noisy-action rope rollouts from an existing HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import mujoco
import numpy as np
from tqdm.auto import tqdm

from rope.data.rope_data_gen import append_rows, create_resizable_dataset, extract_step_info, render_rgb_frame
from rope.shared.lab_env import LabEnv, TaskState


DEFAULT_INPUT = Path("rope/data/test_data_noshadow.h5")
DEFAULT_OUTPUT = Path("rope/data/test_data_noshadow_noisy_onestep.h5")
DEFAULT_CAMERA = "video_cam"
DEFAULT_NOISE_STD = np.array([0.00671364, 0.00318584, 0.00928658], dtype=np.float32)
ACTION_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source HDF5 dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output augmented HDF5 dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="First source row to consider.")
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=None,
        help="Maximum number of valid source transitions to augment. Defaults to all valid transitions.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth valid source transition.")
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs=ACTION_DIM,
        default=DEFAULT_NOISE_STD.tolist(),
        metavar=("REACH", "HEIGHT", "WIDTH"),
        help="Per-axis Gaussian action noise std in task-space meters.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to --noise-std.",
    )
    parser.add_argument(
        "--noise-clip-sigma",
        type=float,
        default=2.5,
        help="Clip sampled noise to +/- this many per-axis standard deviations. Use <=0 to disable.",
    )
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=None,
        help="MuJoCo substeps per augmented transition. Defaults to source attr control_decimation.",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=None, help="Rendered output width. Defaults to source width.")
    parser.add_argument("--height", type=int, default=None, help="Rendered output height. Defaults to source height.")
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=None,
        help="Disable shadows for rendered next frames. Defaults to source attr disable_shadows when present.",
    )
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="Do not write a pixels dataset. Useful for state-only dynamics augmentation.",
    )
    return parser.parse_args()


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    if name not in h5:
        raise KeyError(f"Input dataset is missing required key: {name}")
    return h5[name]


def parse_source_resolution(h5: h5py.File) -> tuple[int, int]:
    if "pixels" in h5:
        pixels = h5["pixels"]
        if pixels.ndim != 4:
            raise ValueError(f"Expected pixels with shape [N,H,W,C], got {pixels.shape}")
        return int(pixels.shape[1]), int(pixels.shape[2])
    value = h5.attrs.get("video_resolution")
    if value is not None:
        height, width = json.loads(value)
        return int(height), int(width)
    return 224, 224


def restore_env_to_source_row(
    env: LabEnv,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
) -> None:
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def sample_noisy_action(
    rng: np.random.Generator,
    action: np.ndarray,
    noise_std: np.ndarray,
    *,
    clip_sigma: float,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=noise_std).astype(np.float32)
    if clip_sigma > 0.0:
        limit = clip_sigma * noise_std
        noise = np.clip(noise, -limit, limit)
    return (np.asarray(action, dtype=np.float32) + noise).astype(np.float32)


def copy_current_info(src: h5py.File, row: int) -> dict[str, np.ndarray]:
    return {
        "observation": np.asarray(src["observation"][row], dtype=np.float32),
        "task_target": np.asarray(src["task_target"][row], dtype=np.float32),
        "qpos": np.asarray(src["qpos"][row], dtype=np.float32),
        "qvel": np.asarray(src["qvel"][row], dtype=np.float32),
        "control": np.asarray(src["control"][row], dtype=np.float32),
        "left_attachment_pos": np.asarray(src["left_attachment_pos"][row], dtype=np.float32),
        "right_attachment_pos": np.asarray(src["right_attachment_pos"][row], dtype=np.float32),
        "rope_length": np.asarray(src["rope_length"][row], dtype=np.float32),
        "time": np.asarray(src["time"][row], dtype=np.float32),
    }


def append_one_step_episode(
    *,
    out: h5py.File,
    datasets: dict[str, h5py.Dataset],
    current: dict[str, np.ndarray],
    next_info: dict[str, np.ndarray],
    action: np.ndarray,
    source_row: int,
    episode_idx: int,
    current_frame: np.ndarray | None,
    next_frame: np.ndarray | None,
) -> None:
    episode_len = 2
    if "pixels" in datasets:
        if current_frame is None or next_frame is None:
            raise ValueError("Pixel output is enabled but frames were not provided.")
        offset, _ = append_rows(datasets["pixels"], np.stack([current_frame, next_frame], axis=0))
    else:
        offset = episode_idx * episode_len

    padded_actions = np.empty((episode_len, ACTION_DIM), dtype=np.float32)
    padded_actions[0] = np.asarray(action, dtype=np.float32)
    padded_actions[1] = np.nan

    append_rows(datasets["action"], padded_actions)
    for name in (
        "observation",
        "task_target",
        "qpos",
        "qvel",
        "control",
        "left_attachment_pos",
        "right_attachment_pos",
        "rope_length",
        "time",
    ):
        append_rows(datasets[name], np.stack([current[name], next_info[name]], axis=0))

    append_rows(datasets["episode_idx"], np.full((episode_len,), episode_idx, dtype=np.int64))
    append_rows(datasets["step_idx"], np.arange(episode_len, dtype=np.int64))
    append_rows(datasets["source_row"], np.full((episode_len,), source_row, dtype=np.int64))
    append_rows(datasets["ep_len"], np.asarray([episode_len], dtype=np.int64))
    append_rows(datasets["ep_offset"], np.asarray([offset], dtype=np.int64))
    append_rows(datasets["reward"], np.asarray([0.0], dtype=np.float32))
    append_rows(datasets["episode_seed"], np.asarray([episode_idx], dtype=np.int64))
    append_rows(datasets["terminated"], np.asarray([True], dtype=np.bool_))
    append_rows(datasets["truncated"], np.asarray([False], dtype=np.bool_))


def create_output_datasets(
    out: h5py.File,
    *,
    sample_info: dict[str, np.ndarray],
    pixel_shape: tuple[int, int, int] | None,
    compression: str | None,
) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {
        "ep_len": create_resizable_dataset(out, "ep_len", (), np.int64, chunks=True),
        "ep_offset": create_resizable_dataset(out, "ep_offset", (), np.int64, chunks=True),
        "reward": create_resizable_dataset(out, "reward", (), np.float32, chunks=True),
        "episode_seed": create_resizable_dataset(out, "episode_seed", (), np.int64, chunks=True),
        "terminated": create_resizable_dataset(out, "terminated", (), np.bool_, chunks=True),
        "truncated": create_resizable_dataset(out, "truncated", (), np.bool_, chunks=True),
        "action": create_resizable_dataset(out, "action", (ACTION_DIM,), np.float32, chunks=True),
        "episode_idx": create_resizable_dataset(out, "episode_idx", (), np.int64, chunks=True),
        "step_idx": create_resizable_dataset(out, "step_idx", (), np.int64, chunks=True),
        "source_row": create_resizable_dataset(out, "source_row", (), np.int64, chunks=True),
    }
    for name in (
        "observation",
        "task_target",
        "qpos",
        "qvel",
        "control",
        "left_attachment_pos",
        "right_attachment_pos",
        "rope_length",
        "time",
    ):
        datasets[name] = create_resizable_dataset(out, name, sample_info[name].shape, np.float32, chunks=True)

    if pixel_shape is not None:
        datasets["pixels"] = create_resizable_dataset(
            out,
            "pixels",
            pixel_shape,
            np.uint8,
            compression=compression,
            chunks=(1, *pixel_shape),
        )
    return datasets


def valid_source_rows(actions: h5py.Dataset, *, start_index: int, stride: int, max_transitions: int | None) -> list[int]:
    rows: list[int] = []
    valid_count = 0
    for row in range(start_index, int(actions.shape[0])):
        action = np.asarray(actions[row], dtype=np.float32)
        if not np.isfinite(action).all():
            continue
        if (valid_count % stride) != 0:
            valid_count += 1
            continue
        valid_count += 1
        rows.append(row)
        if max_transitions is not None and len(rows) >= max_transitions:
            break
    return rows


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative.")
    if args.stride < 1:
        raise ValueError("--stride must be positive.")
    if args.max_transitions is not None and args.max_transitions < 1:
        raise ValueError("--max-transitions must be positive when provided.")
    if args.noise_scale < 0.0:
        raise ValueError("--noise-scale cannot be negative.")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    noise_std = np.asarray(args.noise_std, dtype=np.float32) * float(args.noise_scale)
    if noise_std.shape != (ACTION_DIM,):
        raise ValueError(f"--noise-std must have shape ({ACTION_DIM},), got {noise_std.shape}")
    if np.any(noise_std < 0.0):
        raise ValueError("--noise-std values must be non-negative.")

    compression = None if args.compression == "none" else args.compression
    rng = np.random.default_rng(args.seed)

    with h5py.File(input_path, "r") as src:
        for name in (
            "action",
            "observation",
            "task_target",
            "qpos",
            "qvel",
            "control",
            "left_attachment_pos",
            "right_attachment_pos",
            "rope_length",
            "time",
        ):
            require_dataset(src, name)

        source_height, source_width = parse_source_resolution(src)
        height = source_height if args.height is None else int(args.height)
        width = source_width if args.width is None else int(args.width)
        control_decimation = args.control_decimation
        if control_decimation is None:
            control_decimation = int(src.attrs.get("control_decimation", 25))
        if control_decimation < 1:
            raise ValueError("--control-decimation must be positive.")
        control_timestep = float(src.attrs.get("control_timestep", 0.05))
        disable_shadows = bool(src.attrs.get("disable_shadows", False)) if args.disable_shadows is None else bool(args.disable_shadows)

        rows = valid_source_rows(
            src["action"],
            start_index=args.start_index,
            stride=args.stride,
            max_transitions=args.max_transitions,
        )
        if not rows:
            raise ValueError("No valid finite-action source rows were selected.")

        env = LabEnv()
        camera_id = env.model.camera(args.camera).id
        sample_info = copy_current_info(src, rows[0])
        pixel_shape = None if args.no_pixels else (height, width, 3)

        with h5py.File(output_path, "w") as out:
            for key, value in src.attrs.items():
                out.attrs[key] = value
            out.attrs["source"] = "rope/data/rope_data_augment.py"
            out.attrs["augmentation_input"] = str(input_path)
            out.attrs["augmentation_seed"] = int(args.seed)
            out.attrs["augmentation_noise_std"] = json.dumps(noise_std.astype(float).tolist())
            out.attrs["augmentation_noise_clip_sigma"] = float(args.noise_clip_sigma)
            out.attrs["augmentation_source_rows"] = int(len(rows))
            out.attrs["format"] = "stable_worldmodel_hdf5"
            out.attrs["action_dim"] = ACTION_DIM
            out.attrs["control_decimation"] = int(control_decimation)
            out.attrs["control_timestep"] = float(control_timestep)
            out.attrs["video_resolution"] = json.dumps([height, width])
            out.attrs["camera"] = args.camera
            out.attrs["disable_shadows"] = disable_shadows
            out.attrs["one_step_episodes"] = True
            out.attrs["pixels_enabled"] = not args.no_pixels

            datasets = create_output_datasets(
                out,
                sample_info=sample_info,
                pixel_shape=pixel_shape,
                compression=compression,
            )

            renderer_context = (
                mujoco.Renderer(env.model, height=height, width=width)
                if not args.no_pixels
                else None
            )
            if renderer_context is None:
                renderer = None
                close_renderer = None
            else:
                renderer = renderer_context.__enter__()
                close_renderer = renderer_context.__exit__

            try:
                for episode_idx, row in enumerate(tqdm(rows, desc="Augmenting transitions", unit="step")):
                    current = copy_current_info(src, row)
                    action = np.asarray(src["action"][row], dtype=np.float32)
                    noisy_action = sample_noisy_action(
                        rng,
                        action,
                        noise_std,
                        clip_sigma=float(args.noise_clip_sigma),
                    )

                    restore_env_to_source_row(
                        env,
                        qpos=current["qpos"],
                        qvel=current["qvel"],
                        control=current["control"],
                        task_target=current["task_target"],
                    )
                    target_before = env.task_controller.desired_state.as_array().astype(np.float32)
                    env.apply_task_delta(noisy_action)
                    target_after = env.task_controller.desired_state.as_array().astype(np.float32)
                    applied_noisy_action = (target_after - target_before).astype(np.float32)
                    env.step(control_decimation)
                    next_time = float(np.asarray(current["time"]).reshape(-1)[0]) + control_timestep
                    next_info = extract_step_info(env, elapsed_time=next_time)

                    if renderer is None:
                        current_frame = None
                        next_frame = None
                    else:
                        if "pixels" in src and src["pixels"].shape[1:] == pixel_shape:
                            current_frame = np.asarray(src["pixels"][row], dtype=np.uint8).copy()
                        else:
                            restore_env_to_source_row(
                                env,
                                qpos=current["qpos"],
                                qvel=current["qvel"],
                                control=current["control"],
                                task_target=current["task_target"],
                            )
                            current_frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)
                            restore_env_to_source_row(
                                env,
                                qpos=next_info["qpos"],
                                qvel=next_info["qvel"],
                                control=next_info["control"],
                                task_target=next_info["task_target"],
                            )
                        next_frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)

                    append_one_step_episode(
                        out=out,
                        datasets=datasets,
                        current=current,
                        next_info=next_info,
                        action=applied_noisy_action,
                        source_row=row,
                        episode_idx=episode_idx,
                        current_frame=current_frame,
                        next_frame=next_frame,
                    )
            finally:
                if close_renderer is not None:
                    close_renderer(None, None, None)

            out.attrs["num_episodes"] = len(rows)
            out.attrs["total_frames"] = int(datasets["action"].shape[0])
            out.attrs["total_transitions"] = len(rows)
            out.attrs["mean_reward"] = 0.0
            out.attrs["mean_episode_steps"] = 1.0
            out.attrs["terminated_fraction"] = 1.0
            out.attrs["truncated_fraction"] = 0.0
            out.attrs["usable_train_windows_default"] = 0

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_episodes": len(rows),
        "total_transitions": len(rows),
        "noise_std": noise_std.astype(float).tolist(),
        "noise_clip_sigma": float(args.noise_clip_sigma),
        "control_decimation": int(control_decimation),
        "pixels_enabled": not args.no_pixels,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
