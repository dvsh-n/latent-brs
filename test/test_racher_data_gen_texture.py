#!/usr/bin/env python3
"""Render one textured Reacher test video with a wood arm and tiled floor."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from xml.sax.saxutils import escape

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


def make_wood_texture(size: int = 256) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    rng = np.random.default_rng(7)

    base = np.array([156.0, 110.0, 72.0], dtype=np.float32)
    streaks = 22.0 * np.sin((x / size) * 10.0 * np.pi + 0.6 * np.sin((y / size) * 4.0 * np.pi))
    fine_grain = 10.0 * np.sin((x / size) * 34.0 * np.pi + (y / size) * 2.0 * np.pi)
    knots = 18.0 * np.exp(-((x - 0.28 * size) ** 2 + (y - 0.35 * size) ** 2) / (0.018 * size**2))
    knots += 15.0 * np.exp(-((x - 0.72 * size) ** 2 + (y - 0.68 * size) ** 2) / (0.022 * size**2))
    noise = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)

    intensity = streaks + fine_grain - knots + noise
    texture = np.clip(base + intensity[..., None], 0.0, 255.0)
    texture[..., 1] *= 0.94
    texture[..., 2] *= 0.82
    return texture.astype(np.uint8)


def make_tile_texture(size: int = 256, tile_size: int = 64, grout: int = 4) -> np.ndarray:
    rng = np.random.default_rng(11)
    texture = np.empty((size, size, 3), dtype=np.uint8)

    light_tile = np.array([168, 194, 214], dtype=np.int16)
    dark_tile = np.array([110, 142, 170], dtype=np.int16)
    grout_color = np.array([188, 205, 220], dtype=np.uint8)

    for y0 in range(0, size, tile_size):
        for x0 in range(0, size, tile_size):
            is_light = ((y0 // tile_size) + (x0 // tile_size)) % 2 == 0
            base_color = light_tile if is_light else dark_tile
            tint = rng.integers(-6, 7, size=3, dtype=np.int16)
            tile_color = np.clip(base_color + tint, 0, 255).astype(np.uint8)
            y1 = min(y0 + tile_size, size)
            x1 = min(x0 + tile_size, size)
            texture[y0:y1, x0:x1] = tile_color

    texture[::tile_size, :, :] = grout_color
    texture[:, ::tile_size, :] = grout_color
    for offset in range(1, grout):
        texture[offset::tile_size, :, :] = grout_color
        texture[:, offset::tile_size, :] = grout_color

    speckle = rng.integers(-4, 5, size=(size, size, 1), dtype=np.int16)
    texture = np.clip(texture.astype(np.int16) + speckle, 0, 255).astype(np.uint8)
    return texture


def build_textured_xml(wood_texture_path: Path, tile_texture_path: Path) -> str:
    xml_text = Path(REACHER_XML).read_text()
    xml_text = xml_text.replace(
        '\t\t<geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>\n',
        "",
        1,
    )
    xml_text = xml_text.replace(
        '\t\t<geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>\n',
        "",
        1,
    )
    xml_text = xml_text.replace(
        '\t\t<geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>\n',
        "",
        1,
    )
    xml_text = xml_text.replace(
        '\t\t<geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>\n',
        "",
        1,
    )
    asset_block = f"""
    <asset>
        <texture name="wood_tex" type="2d" file="{escape(str(wood_texture_path))}"/>
        <texture name="tile_tex" type="2d" file="{escape(str(tile_texture_path))}"/>
        <material name="wood_arm_mat" texture="wood_tex" texuniform="true" texrepeat="3 1" specular="0.15" shininess="0.08"/>
        <material name="tile_floor_mat" texture="tile_tex" texuniform="true" texrepeat="6 6" reflectance="0.05"/>
    </asset>
"""
    xml_text = xml_text.replace("<worldbody>", f"{asset_block}\n\t<worldbody>", 1)
    xml_text = xml_text.replace(
        '<geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>',
        '<geom conaffinity="0" contype="0" material="tile_floor_mat" name="ground" pos="0 0 0" size="1 1 10" type="plane"/>',
        1,
    )
    xml_text = xml_text.replace(
        '<geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>',
        '<geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" material="wood_arm_mat" name="root" size=".011" type="cylinder"/>',
        1,
    )
    xml_text = xml_text.replace(
        '<geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>',
        '<geom fromto="0 0 0 0.1 0 0" material="wood_arm_mat" name="link0" size=".01" type="capsule"/>',
        1,
    )
    xml_text = xml_text.replace(
        '<geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>',
        '<geom fromto="0 0 0 0.1 0 0" material="wood_arm_mat" name="link1" size=".01" type="capsule"/>',
        1,
    )
    return xml_text


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

    with tempfile.TemporaryDirectory(prefix="reacher_texture_test_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        wood_texture_path = temp_dir / "wood_texture.png"
        tile_texture_path = temp_dir / "tile_texture.png"

        imageio.imwrite(wood_texture_path, make_wood_texture())
        imageio.imwrite(tile_texture_path, make_tile_texture())

        textured_xml = build_textured_xml(wood_texture_path, tile_texture_path)
        model = mujoco.MjModel.from_xml_string(textured_xml)
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
