#!/usr/bin/env python3
"""Replay an OGBench episode with a provided action sequence and save precise state traces."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import gymnasium
import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie
from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _pick_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of keys {names}, got {sorted(mapping.keys())}.")


def _pick_optional_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _pick_optional_or(mapping: dict, names: tuple[str, ...], default):
    value = _pick_optional_key(mapping, names)
    return default if value is None else value


def _infer_pair_count(payload) -> Optional[int]:
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and "pair_count" in metadata:
            return int(metadata["pair_count"])
        if "pair_count" in payload:
            return int(payload["pair_count"])
        for key in ("pairs", "episodes", "endpoint_pairs"):
            if key in payload and isinstance(payload[key], (list, tuple)):
                return len(payload[key])
        if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
            start = payload["start"]
            if isinstance(start, dict):
                for item in start.values():
                    if isinstance(item, (torch.Tensor, np.ndarray)) and item.ndim > 0:
                        return int(item.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return None


def _select_pair_value(value, episode_idx: int, pair_count: Optional[int]):
    if isinstance(value, dict):
        return {key: _select_pair_value(item, episode_idx, pair_count) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and pair_count is not None and len(value) == pair_count:
        return value[episode_idx]
    if isinstance(value, torch.Tensor) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    if isinstance(value, np.ndarray) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    return value


def _select_endpoint_pair(payload, episode_idx: int):
    pair_count = _infer_pair_count(payload)
    if pair_count is not None and not 0 <= episode_idx < pair_count:
        raise ValueError(f"episode_idx must be in [0, {pair_count - 1}], got {episode_idx}.")
    if isinstance(payload, (list, tuple)):
        return payload[episode_idx], pair_count
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported endpoint payload type: {type(payload)!r}.")
    for key in ("pairs", "episodes", "endpoint_pairs"):
        if key in payload:
            pairs = payload[key]
            if not isinstance(pairs, (list, tuple)):
                raise TypeError(f"Endpoint payload key '{key}' must be a list/tuple, got {type(pairs)!r}.")
            return pairs[episode_idx], len(pairs)
    if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
        return {
            "start": _select_pair_value(payload["start"], episode_idx, pair_count),
            "goal": _select_pair_value(_pick_key(payload, ("goal", "end", "target")), episode_idx, pair_count),
        }, pair_count
    raise KeyError(
        "Endpoint .pt payload must be a list of pairs, contain a 'pairs'/'episodes' list, "
        "or contain top-level 'start' and 'goal'/'end'/'target' entries."
    )


def _load_endpoint_pair(path: Path, episode_idx: int, seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    pair, _ = _select_endpoint_pair(payload, int(episode_idx))
    if not isinstance(pair, dict):
        raise TypeError(f"Selected endpoint pair must be a dict, got {type(pair)!r}.")

    start = _pick_key(pair, ("start", "initial", "source"))
    goal = _pick_key(pair, ("goal", "end", "target"))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("Endpoint pair 'start' and 'goal'/'end' entries must both be dicts.")

    start_qpos_raw = _pick_optional_key(start, ("qpos", "q_pos"))
    goal_qpos_raw = _pick_optional_key(goal, ("qpos", "q_pos"))
    start_block_pos = _as_numpy(_pick_key(start, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    goal_block_pos = _as_numpy(_pick_key(goal, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    start_block_yaw = float(_as_numpy(_pick_key(start, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    goal_block_yaw = float(_as_numpy(_pick_key(goal, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    start_qpos = None if start_qpos_raw is None else _as_numpy(start_qpos_raw, dtype=np.float32)
    goal_qpos = None if goal_qpos_raw is None else _as_numpy(goal_qpos_raw, dtype=np.float32)
    start_qvel_raw = _pick_optional_key(start, ("qvel", "q_vel"))
    goal_qvel_raw = _pick_optional_key(goal, ("qvel", "q_vel"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    episode = {
        "needs_qpos_synthesis": start_qpos is None or goal_qpos is None,
        "block_pos_init": start_block_pos,
        "block_yaw_init": start_block_yaw,
        "block_pos_goal": goal_block_pos,
        "block_yaw_goal": goal_block_yaw,
        "qpos_init": start_qpos,
        "qvel_init": None if start_qpos is None else (
            np.zeros_like(start_qpos, dtype=np.float32) if start_qvel_raw is None else _as_numpy(start_qvel_raw, dtype=np.float32)
        ),
        "qpos_goal": goal_qpos,
        "qvel_goal": None if goal_qpos is None else (
            np.zeros_like(goal_qpos, dtype=np.float32) if goal_qvel_raw is None else _as_numpy(goal_qvel_raw, dtype=np.float32)
        ),
        "target_block_pos_init": _as_numpy(
            _pick_optional_or(start, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos),
            dtype=np.float32,
        ),
        "target_block_yaw_init": float(
            _as_numpy(
                _pick_optional_or(start, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)
            ).reshape(-1)[0]
        ),
        "target_block_pos_goal": _as_numpy(
            _pick_optional_or(goal, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos),
            dtype=np.float32,
        ),
        "target_block_yaw_goal": float(
            _as_numpy(
                _pick_optional_or(goal, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)
            ).reshape(-1)[0]
        ),
        "episode_seed": int(metadata.get("episode_seed", seed)) if isinstance(metadata, dict) else int(seed),
        "env_name": str(metadata.get("env_name", "cube-single-v0")) if isinstance(metadata, dict) else "cube-single-v0",
        "camera": str(metadata.get("camera", "front_pixels")) if isinstance(metadata, dict) else "front_pixels",
    }
    return episode, int(episode_idx)


def synthesize_qpos_qvel_from_block_pose(env: gymnasium.Env, pos: np.ndarray, yaw: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    joint_qpos = unwrapped._data.joint("object_joint_0").qpos
    joint_qpos[:3] = np.asarray(pos, dtype=np.float64)
    joint_qpos[3:] = np.asarray(lie.SO3.from_z_radians(float(yaw)).wxyz, dtype=np.float64)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return (
        np.asarray(unwrapped._data.qpos, dtype=np.float32).copy(),
        np.zeros_like(np.asarray(unwrapped._data.qvel, dtype=np.float32)),
    )


def load_planning_episode(path: Path, episode_idx: int, seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    path = path.expanduser()
    if path.suffix.lower() == ".pt":
        return _load_endpoint_pair(path, episode_idx, seed)
    with h5py.File(path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        if not 0 <= episode_idx < len(ep_len):
            raise ValueError(f"episode_idx must be in [0, {len(ep_len) - 1}], got {episode_idx}.")
        rows = np.arange(int(h5["ep_offset"][episode_idx]), int(h5["ep_offset"][episode_idx]) + int(h5["ep_len"][episode_idx]))
        episode = {
            "qpos_init": np.asarray(h5["qpos"][rows[0]], dtype=np.float32),
            "qvel_init": np.asarray(h5["qvel"][rows[0]], dtype=np.float32),
            "qpos_goal": np.asarray(h5["qpos"][rows[-1]], dtype=np.float32),
            "qvel_goal": np.asarray(h5["qvel"][rows[-1]], dtype=np.float32),
            "target_block_pos_init": np.asarray(h5["target_block_pos"][rows[0]], dtype=np.float32),
            "target_block_yaw_init": float(h5["target_block_yaw"][rows[0], 0]),
            "target_block_pos_goal": np.asarray(h5["target_block_pos"][rows[-1]], dtype=np.float32),
            "target_block_yaw_goal": float(h5["target_block_yaw"][rows[-1], 0]),
            "episode_seed": int(h5["episode_seed"][episode_idx]) if "episode_seed" in h5 else int(seed),
            "env_name": str(h5.attrs.get("env_name", "cube-single-v0")),
            "camera": str(h5.attrs.get("camera", "front_pixels")),
            "needs_qpos_synthesis": False,
        }
    return episode, int(episode_idx)


def restore_target_pose(env: gymnasium.Env, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )


def reset_env_to_state(
    env: gymnasium.Env,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_block_pos: np.ndarray,
    target_block_yaw: float,
) -> dict[str, np.ndarray]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(env, target_block_pos, target_block_yaw)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return unwrapped.get_step_info()


def render_frame(env: gymnasium.Env, camera: str) -> np.ndarray:
    return np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)


def cube_is_grasped(info: dict[str, np.ndarray], contact_thresh: float, align_thresh: float) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_thresh and block_alignment <= align_thresh)


def load_actions(path: Path, key: str) -> np.ndarray:
    path = path.expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as payload:
            if key not in payload:
                raise KeyError(f"{path} does not contain key '{key}'. Available keys: {sorted(payload.files)}")
            return np.asarray(payload[key], dtype=np.float32)
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if key not in payload:
                raise KeyError(f"{path} does not contain key '{key}'. Available keys: {sorted(payload.keys())}")
            return _as_numpy(payload[key], dtype=np.float32)
        return _as_numpy(payload, dtype=np.float32)
    raise ValueError(f"Unsupported actions file suffix '{suffix}'. Expected .npz, .npy, or .pt.")


def extract_step_state(info: dict[str, np.ndarray]) -> dict[str, np.ndarray | float | bool]:
    target_block = int(info["privileged/target_block"])
    return {
        "qpos": np.asarray(info["qpos"], dtype=np.float32),
        "qvel": np.asarray(info["qvel"], dtype=np.float32),
        "control": np.asarray(info["control"], dtype=np.float32),
        "time": np.asarray(info["time"], dtype=np.float32),
        "effector_pos": np.asarray(info["proprio/effector_pos"], dtype=np.float32),
        "effector_yaw": np.asarray(info["proprio/effector_yaw"], dtype=np.float32),
        "gripper_opening": np.asarray(info["proprio/gripper_opening"], dtype=np.float32),
        "gripper_contact": np.asarray(info["proprio/gripper_contact"], dtype=np.float32),
        "block_pos": np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32),
        "block_quat": np.asarray(info[f"privileged/block_{target_block}_quat"], dtype=np.float32),
        "block_yaw": np.asarray(info[f"privileged/block_{target_block}_yaw"], dtype=np.float32),
        "target_block_pos": np.asarray(info["privileged/target_block_pos"], dtype=np.float32),
        "target_block_yaw": np.asarray(info["privileged/target_block_yaw"], dtype=np.float32),
        "target_block": target_block,
        "success": bool(np.asarray(info.get("success", False)).item()),
    }


def stack_state_dicts(states: list[dict[str, np.ndarray | float | bool]]) -> dict[str, object]:
    keys = states[0].keys()
    result: dict[str, object] = {}
    for key in keys:
        values = [state[key] for state in states]
        first = values[0]
        if isinstance(first, np.ndarray):
            result[key] = np.stack(values, axis=0)
        elif isinstance(first, (bool, np.bool_)):
            result[key] = np.asarray(values, dtype=np.bool_)
        elif isinstance(first, (int, np.integer)):
            result[key] = np.asarray(values, dtype=np.int64)
        else:
            result[key] = np.asarray(values, dtype=np.float32)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-idx", type=int, required=True)
    parser.add_argument("--actions-file", type=Path, required=True)
    parser.add_argument("--actions-key", default="executed_actions_raw")
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--video-out", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--skip-oracle-grasp", action="store_true", default=False)
    parser.add_argument("--max-oracle-steps", type=int, default=80)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = load_actions(args.actions_file, args.actions_key)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T, action_dim], got {actions.shape}.")
    video_out = args.video_out if args.video_out is not None else args.out_file.with_suffix(".mp4")

    episode, episode_idx = load_planning_episode(args.dataset_path, args.episode_idx, args.seed)
    env = gymnasium.make(
        str(episode["env_name"]),
        terminate_at_goal=False,
        mode="data_collection",
        width=256,
        height=256,
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

    try:
        qpos_init = episode["qpos_init"]
        qvel_init = episode["qvel_init"]
        if bool(episode.get("needs_qpos_synthesis", False)):
            qpos_init, qvel_init = synthesize_qpos_qvel_from_block_pose(
                env,
                np.asarray(episode["block_pos_init"], dtype=np.float32),
                float(episode["block_yaw_init"]),
                int(episode["episode_seed"]),
            )

        info = reset_env_to_state(
            env,
            seed=int(episode["episode_seed"]),
            qpos=np.asarray(qpos_init, dtype=np.float32),
            qvel=np.asarray(qvel_init, dtype=np.float32),
            target_block_pos=np.asarray(episode["target_block_pos_init"], dtype=np.float32),
            target_block_yaw=float(episode["target_block_yaw_init"]),
        )

        states = [extract_step_state(info)]
        frames = [render_frame(env, str(episode["camera"]))]
        print(
            f"step=0 "
            f"arm_pos={states[-1]['effector_pos'].tolist()} "
            f"cube_pos={states[-1]['block_pos'].tolist()}"
        )
        run_oracle_grasp = not args.skip_oracle_grasp
        if run_oracle_grasp:
            oracle.reset(None, info)
            grasped = False
            for _ in range(args.max_oracle_steps):
                if cube_is_grasped(info, args.grasp_contact_threshold, args.grasp_alignment_threshold):
                    grasped = True
                    break
                oracle_action = np.asarray(oracle.select_action(None, info), dtype=np.float32)
                _, _, _, _, info = env.step(oracle_action)
                states.append(extract_step_state(info))
                frames.append(render_frame(env, str(episode["camera"])))
                step_idx = len(states) - 1
                print(
                    f"step={step_idx} phase=oracle "
                    f"arm_pos={states[-1]['effector_pos'].tolist()} "
                    f"cube_pos={states[-1]['block_pos'].tolist()}"
                )
            if not grasped:
                raise RuntimeError(
                    "Oracle grasp phase did not reach a grasped state within "
                    f"{args.max_oracle_steps} steps."
                )
        for action in actions:
            _, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
            states.append(extract_step_state(info))
            frames.append(render_frame(env, str(episode["camera"])))
            step_idx = len(states) - 1
            print(
                f"step={step_idx} phase=replay "
                f"arm_pos={states[-1]['effector_pos'].tolist()} "
                f"cube_pos={states[-1]['block_pos'].tolist()}"
            )
    finally:
        env.close()

    payload = {
        "metadata": {
            "dataset_path": str(args.dataset_path.expanduser()),
            "episode_idx": int(episode_idx),
            "actions_file": str(args.actions_file.expanduser()),
            "actions_key": str(args.actions_key),
            "num_actions": int(actions.shape[0]),
            "episode_seed": int(episode["episode_seed"]),
            "env_name": str(episode["env_name"]),
            "camera": str(episode["camera"]),
            "run_oracle_grasp": bool(run_oracle_grasp),
            "max_oracle_steps": int(args.max_oracle_steps),
            "grasp_contact_threshold": float(args.grasp_contact_threshold),
            "grasp_alignment_threshold": float(args.grasp_alignment_threshold),
        },
        "actions": actions.astype(np.float32),
        "states": stack_state_dicts(states),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_file)
    imageio.mimwrite(video_out, frames, fps=args.video_fps)
    print(f"Saved replayed state trace with {len(states)} states to {args.out_file}")
    print(f"Saved replay video to {video_out}")


if __name__ == "__main__":
    main()
