#!/usr/bin/env python3
"""Render one Reacher test video with arm segmentation mask and overlay."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.reacher_data_gen import (
    REACHER_XML,
    TOP_DOWN_CAMERA,
    configure_mujoco_gl,
    import_mujoco,
)


OUTPUT_VIDEO_PATH = Path("test/reacher_texture_test.mp4")
MASK_VIDEO_PATH = Path("test/reacher_texture_mask.mp4")
OVERLAY_VIDEO_PATH = Path("test/reacher_texture_overlay.mp4")
WIDTH = 512
HEIGHT = 512
FPS = 50
NUM_FRAMES = 150
GL_BACKEND = "osmesa"
OVERLAY_ALPHA = 0.45
OVERLAY_COLOR = np.array([255.0, 64.0, 32.0], dtype=np.float32)


def ensure_offscreen_buffer(model: object, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def camera_from_config(mujoco: object) -> object:
    camera = mujoco.MjvCamera()
    camera.distance = float(TOP_DOWN_CAMERA["distance"])
    camera.azimuth = float(TOP_DOWN_CAMERA["azimuth"])
    camera.elevation = float(TOP_DOWN_CAMERA["elevation"])
    camera.lookat[:] = TOP_DOWN_CAMERA["lookat"]
    return camera


def make_qpos_trajectory(num_frames: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    joint0 = 1.05 * np.sin(2.0 * np.pi * t)
    joint1 = -1.2 + 1.0 * np.sin(4.0 * np.pi * t + 0.4)
    target_x = np.full_like(t, 0.16)
    target_y = np.full_like(t, -0.08)
    return np.stack([joint0, joint1, target_x, target_y], axis=1).astype(np.float32)


def get_arm_geom_pairs(mujoco: object, model: object) -> np.ndarray:
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_names = ("root", "link0", "link1")
    geom_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) for name in geom_names]
    return np.asarray([(geom_id, geom_type) for geom_id in geom_ids], dtype=np.int32)


def hide_non_arm_geoms_for_segmentation(mujoco: object, model: object) -> object:
    """Move the target geom to group 3 and return an MjvOption that hides group 3.

    This ensures the target sphere cannot occlude the arm in segmentation,
    preventing circular holes in the arm mask.
    """
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    model.geom_group[target_geom_id] = 3

    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0  # hide group 3
    return vopt


def build_arm_mask(segmentation: np.ndarray, arm_geom_pairs: np.ndarray) -> np.ndarray:
    mask = np.zeros(segmentation.shape[:2], dtype=bool)
    for geom_id, geom_type in arm_geom_pairs:
        mask |= (segmentation[..., 0] == geom_id) & (segmentation[..., 1] == geom_type)
    return mask


def render_overlay_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    composed = frame.astype(np.float32).copy()
    composed[mask] = (1.0 - OVERLAY_ALPHA) * composed[mask] + OVERLAY_ALPHA * OVERLAY_COLOR
    return np.clip(composed, 0, 255).astype(np.uint8)


def render_videos(
    mujoco: object,
    model: object,
    qpos_trajectory: np.ndarray,
    *,
    output_path: Path,
    mask_output_path: Path,
    overlay_output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    camera = camera_from_config(mujoco)
    ensure_offscreen_buffer(model, width=width, height=height)
    arm_geom_pairs = get_arm_geom_pairs(mujoco, model)
    seg_vopt = hide_non_arm_geoms_for_segmentation(mujoco, model)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        frames: list[np.ndarray] = []
        mask_frames: list[np.ndarray] = []
        overlay_frames: list[np.ndarray] = []
        for qpos in qpos_trajectory:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()

            renderer.update_scene(data, camera=camera, scene_option=seg_vopt)
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            renderer.disable_segmentation_rendering()

            arm_mask = build_arm_mask(segmentation, arm_geom_pairs)
            mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
            mask_frame[arm_mask] = 255

            frames.append(frame)
            mask_frames.append(mask_frame)
            overlay_frames.append(render_overlay_frame(frame, arm_mask))

    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    imageio.mimwrite(mask_output_path, mask_frames, fps=fps, quality=8)
    imageio.mimwrite(overlay_output_path, overlay_frames, fps=fps, quality=8)


def main() -> None:
    configure_mujoco_gl(GL_BACKEND)
    mujoco = import_mujoco()

    model = mujoco.MjModel.from_xml_path(str(REACHER_XML))
    qpos_trajectory = make_qpos_trajectory(NUM_FRAMES)
    render_videos(
        mujoco,
        model,
        qpos_trajectory,
        output_path=OUTPUT_VIDEO_PATH.resolve(),
        mask_output_path=MASK_VIDEO_PATH.resolve(),
        overlay_output_path=OVERLAY_VIDEO_PATH.resolve(),
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
    )

    print(
        {
            "output_video": str(OUTPUT_VIDEO_PATH.resolve()),
            "mask_video": str(MASK_VIDEO_PATH.resolve()),
            "overlay_video": str(OVERLAY_VIDEO_PATH.resolve()),
            "frames": NUM_FRAMES,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "gl_backend": GL_BACKEND,
        }
    )


if __name__ == "__main__":
    main()
