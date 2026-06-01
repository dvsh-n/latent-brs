import os
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from pathlib import Path
from collections import deque
from pupil_apriltags import Detector

from pydrake.all import (
    LeafSystem,
    RigidTransform,
    RotationMatrix,
)

# tag_ids_required = [0, 1, 10]  # IDs of the tags we care about
# tag_sizes_default = {
#     0: 0.06667,  # meters
#     1: 0.06667,
#     10: 0.098,
# }

tag_ids_required = [0, 1, 18]  # IDs of the tags we care about
tag_sizes_default = {
    0: 0.06667,  # meters
    1: 0.06667,
    18: 0.0278
}



class PlanarPoseDetectorAPI:
    """Camera + AprilTag planar pose detector (non-Drake API).

    Returns planar pose (x, y, theta) in the *tag-defined frame* used by the
    original implementation. In your real pipeline, you can interpret this as
    already being in the workspace frame (as you requested for now).
    """

    def __init__(
        self,
        *,
        w: int = 1280,
        h: int = 720,
        fps: int = 30,
        max_stored_images: int = 999,
        tag_sizes: dict[int, float] | None = None,
        detector_families: str = "tagStandard41h12",
    ):
        # ---- RealSense stream ----
        self._w, self._h, self._fps = int(w), int(h), int(fps)

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._w, self._h, rs.format.bgr8, self._fps)
        self._pipeline.start(config)

        profile = self._pipeline.get_active_profile()
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        fx, fy = intr.fx, intr.fy
        cx, cy = intr.ppx, intr.ppy

        self._camera_params = (fx, fy, cx, cy)
        self._camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self._distortion_coeffs = np.array(intr.coeffs, dtype=float)

        # ---- AprilTags ----
        self._tag_sizes = tag_sizes or tag_sizes_default


        self._detector = Detector(   
                families=detector_families,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.8,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
        )
        self._last_detections = []
        self._last_pose = None
        self._detect_lock = threading.RLock()

        # ---- Background latest-pose cache ----
        self._latest_lock = threading.Lock()
        self._latest_planar_pose = None
        self._latest_pose_valid = False
        self._latest_pose_seq = 0
        self._latest_pose_time = None
        self._bg_thread = None
        self._bg_stop_event = None
        self._bg_store_image = False
        self._bg_period_sec = 1.0 / float(self._fps)
        self._bg_first_valid_logged = False

        # ---- Image storage (optional) ----
        self._images = deque(maxlen=int(max_stored_images))

        # Warm-up: make sure we can see required tags at least once.
        img0 = self.get_color_image(blocking=True, store_image=False)
        pose0 = None if img0 is None else self.detect_planar_pose(img0)
        if pose0 is None:
            raise RuntimeError("Missing required tags; cannot initialize planar pose detector.")
        self._update_latest_pose_cache(pose0)

    @property
    def fps(self) -> int:
        return self._fps

    def close(self) -> None:
        # Best-effort shutdown.
        self.stop_background_detection()
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def save_images(self, dir_path) -> None:
        for i, image in enumerate(self._images):
            img_path = os.path.join(dir_path, f"{i:03d}.png")
            cv2.imwrite(img_path, image)
        print(f"Saved {len(self._images)} images to {dir_path}.")
        # merge images into video for easier viewing
        video_path = os.path.join(dir_path, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, self._fps, (self._w, self._h))
        for image in self._images:
            out.write(image)
        out.release()
        print(f"Saved video to {video_path}.")

    # ----------------------------
    # Public "getters"
    # ----------------------------
    def get_planar_pose_in_world_mm(self, *, blocking: bool = False, store_image: bool = True) -> np.ndarray | None:
        pose = self.get_planar_pose(blocking=blocking, store_image=store_image)
        if pose is None:
            return None
        pose[:2] = pose[:2] * 1000  # m to mm
        return pose

    def get_planar_pose(self, *, blocking: bool = False, store_image: bool = True) -> np.ndarray | None:
        """Returns np.array([x, y, theta]) or None if not detected."""
        with self._detect_lock:
            color_image = self.get_color_image(blocking=blocking, store_image=store_image)
            if color_image is None:
                return None
            pose = self.detect_planar_pose(color_image)
        if pose is not None:
            self._update_latest_pose_cache(pose)
        return pose

    def get_latest_planar_pose(self, *, require_valid: bool = False) -> np.ndarray | None:
        with self._latest_lock:
            if self._latest_pose_valid and self._latest_planar_pose is not None:
                return self._latest_planar_pose.copy()
        if require_valid:
            raise RuntimeError("No valid cached planar pose is available yet.")
        return None

    def get_latest_planar_pose_in_world_mm(self, *, require_valid: bool = False) -> np.ndarray | None:
        pose = self.get_latest_planar_pose(require_valid=require_valid)
        if pose is None:
            return None
        pose_mm = pose.copy()
        pose_mm[:2] = pose_mm[:2] * 1000.0
        return pose_mm

    def is_background_detection_running(self) -> bool:
        return self._bg_thread is not None and self._bg_thread.is_alive()

    def start_background_detection(self, *, store_image: bool = False, period_sec: float | None = None) -> None:
        if self.is_background_detection_running():
            return
        self._bg_store_image = bool(store_image)
        self._bg_period_sec = float(period_sec) if period_sec is not None else (1.0 / float(self._fps))
        self._bg_stop_event = threading.Event()
        self._bg_first_valid_logged = False
        self._bg_thread = threading.Thread(
            target=self._background_detection_loop,
            name="realsense_background_detection",
            daemon=True,
        )
        self._bg_thread.start()
        print(f"[realsense] Background detection started (dt={self._bg_period_sec:.3f}s, store_image={self._bg_store_image})")

    def stop_background_detection(self, join_timeout_sec: float = 2.0) -> None:
        if not self.is_background_detection_running():
            return
        print("[realsense] Stopping background detection...")
        self._bg_stop_event.set()
        self._bg_thread.join(timeout=float(join_timeout_sec))
        if self._bg_thread.is_alive():
            raise RuntimeError("Background detection thread did not stop cleanly.")
        self._bg_thread = None
        self._bg_stop_event = None
        print("[realsense] Background detection stopped")

    def _update_latest_pose_cache(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float).reshape(3)
        with self._latest_lock:
            self._latest_planar_pose = pose.copy()
            self._latest_pose_valid = True
            self._latest_pose_seq += 1
            self._latest_pose_time = time.time()

    def _background_detection_loop(self) -> None:
        while not self._bg_stop_event.is_set():
            pose = self.get_planar_pose(blocking=True, store_image=self._bg_store_image)
            if pose is not None and not self._bg_first_valid_logged:
                print(f"[realsense] First valid cached pose: {pose}")
                self._bg_first_valid_logged = True

    def get_color_image(self, *, blocking: bool = False, store_image: bool = True) -> np.ndarray | None:
        """Returns undistorted BGR image (H,W,3) or None."""
        newest_frame = None
        if blocking:
            newest_frame = self._pipeline.wait_for_frames()
        else:
            # Drain the queue and keep only the newest frame.
            while True:
                frame = self._pipeline.poll_for_frames()
                if not frame:
                    break
                newest_frame = frame

        if newest_frame is None:
            return None

        color_frame = newest_frame.get_color_frame()
        if color_frame is None:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.undistort(color_image, self._camera_matrix, self._distortion_coeffs)

        if store_image:
            self._images.append(color_image)
        return color_image

    # ----------------------------
    # Detection math (same as before)
    # ----------------------------
    def detect_planar_pose(self, color_image: np.ndarray) -> np.ndarray | None:
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # gray_image = cv2.GaussianBlur(cv2.equalizeHist(gray_image), (5,5), 0)
        
        detections = self._detect_tags(gray_image)
        # return self._calc_planar_position(detections)
        return self._calc_planar_position_v1(detections)

    # def _detect_tags(self, gray_image: np.ndarray):
    #     all_detections = []
    #     # Run detector once per distinct tag size so pose estimation uses the right size.
    #     for tag_size in set(self._tag_sizes.values()):
    #         detections = self._detector.detect(
    #             gray_image,
    #             estimate_tag_pose=True,
    #             camera_params=self._camera_params,
    #             tag_size=tag_size,
    #         )
    #         current_ids = [tag_id for tag_id, size in self._tag_sizes.items() if size == tag_size]
    #         detections = [d for d in detections if d.tag_id in current_ids]
    #         all_detections.extend(detections)
    #     return all_detections

    def _detect_tags(self, gray_image: np.ndarray):
        all_detections = []
        try:
            # Run detector once per distinct tag size so pose estimation uses the right size.
            for tag_size in set(self._tag_sizes.values()):
                detections = self._detector.detect(
                    gray_image,
                    estimate_tag_pose=True,
                    camera_params=self._camera_params,
                    tag_size=tag_size,
                )
                current_ids = [tag_id for tag_id, size in self._tag_sizes.items()
                            if size == tag_size]
                detections = [d for d in detections if d.tag_id in current_ids]
                all_detections.extend(detections)
        except Exception as e:
            # AprilTag can throw e.g. "more than one new minima found" during pose estimation.
            # In that case, keep previous detections to avoid pose jumps.
            # (Optionally log e once in a while.)
            print(f"AprilTag detection error: {e}")
            return self._last_detections

        # If we got all tags, update cache; otherwise keep last.
        if len(all_detections) == 3:
            self._last_detections = all_detections
        else:
            print(f"Warning: detected {len(all_detections)} tags; need 3. Keeping last detections.")
        return self._last_detections


    def _calc_planar_position_v1(self, detections) -> np.ndarray | None:
        X_CA, X_CB, X_CT = None, None, None
        for det in detections:
            pose = RigidTransform(RotationMatrix(det.pose_R), det.pose_t)
            if det.tag_id == tag_ids_required[0]:
                X_CA = pose
            elif det.tag_id == tag_ids_required[1]:
                X_CB = pose
            elif det.tag_id == tag_ids_required[2]:
                X_CT = pose

        if any([X_CA is None, X_CB is None, X_CT is None]):
            return None

        # offset the tag T to center of T object
        offset = np.array([0.0, -0.107, 0.0])  # meters
        X_CT = X_CT.multiply(RigidTransform(RotationMatrix.Identity(), offset))

        # Express A and B in tag-T frame, then define object pose from them.
        p_TA = X_CT.inverse() @ X_CA.translation()
        p_TB = X_CT.inverse() @ X_CB.translation()
        # print(f'X_CA: {X_CA.GetAsMatrix4()[0,0]}')
        # print(f'X_CB: {X_CB.GetAsMatrix4()[0,0]}')
        # print(f"X_CT: {X_CT.GetAsMatrix4()[0,0]}")
        # print(f"X_CT: {X_CT.GetAsMatrix4()}")
        # print(f"X_CT.inverse(): {X_CT.inverse().GetAsMatrix4()}")
        # print(f"p_TA: {p_TA}")
        # print(f"p_TB: {p_TB}")
        p_TO = (p_TA + p_TB) / 2

        x_TO = p_TB - p_TA
        x_TO = x_TO / np.linalg.norm(x_TO)
        z_TO = np.array([0, 0, -1])
        y_TO = np.cross(z_TO, x_TO)
        R_TO = RotationMatrix(np.vstack((x_TO, y_TO, z_TO)).T)

        X_TO = RigidTransform(R_TO, p_TO)
        X_OT = X_TO.inverse()
        
        R_OT = X_OT.rotation().matrix()
        p_OT = X_OT.translation()
        # print(f"p_OT: {p_OT}, R_OT: {R_OT}")

        # NOTE: keeping original offset behavior for now.
        pos = p_OT[:2] - np.array([-0.75, 0.04])
        angle = np.arctan2(R_OT[1, 0], R_OT[0, 0])
        # normalize angle to [0, 2pi)
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        return np.concatenate((pos, [angle]))


    def _calc_planar_position(self, detections) -> np.ndarray | None:
        X_CA, X_CB, X_CT = None, None, None
        for det in detections:
            pose = RigidTransform(RotationMatrix(det.pose_R), det.pose_t)
            if det.tag_id == tag_ids_required[0]:
                X_CA = pose
            elif det.tag_id == tag_ids_required[1]:
                X_CB = pose
            elif det.tag_id == tag_ids_required[2]:
                X_CT = pose

        if any([X_CA is None, X_CB is None, X_CT is None]):
            return None


        # --- Build workspace frame O in camera frame C using fixed tags A,B ---
        p_CA = X_CA.translation()
        p_CB = X_CB.translation()
        p_CO = (p_CA + p_CB) / 2.0

        # print(f'X_CA: {X_CA.GetAsMatrix4()[0,0]}')
        # print(f'X_CB: {X_CB.GetAsMatrix4()[0,0]}')
        # print(f"X_CT: {X_CT.GetAsMatrix4()[0,0]}")
        # Table normal (gravity-up) from tag plane normals (A and B are parallel to table)
        ez = np.array([0.0, 0.0, -1.0])
        z_CA = X_CA.rotation().matrix() @ ez
        z_CB = X_CB.rotation().matrix() @ ez
        z_CO = z_CA + z_CB
        z_CO = z_CO / np.linalg.norm(z_CO)

        # print(f"z_CO: {z_CO}, z_CB: {z_CB}, z_CA: {z_CA}")

        # z_CO = np.array([0.0, 0.0, -1.0])

        # x-axis: A->B projected onto the table plane to remove any small tilt/noise
        x_raw = p_CB - p_CA
        x_CO = x_raw - np.dot(x_raw, z_CO) * z_CO
        x_CO = x_CO / np.linalg.norm(x_CO)

        # y-axis: right-handed
        y_CO = np.cross(z_CO, x_CO)
        y_CO = y_CO / np.linalg.norm(y_CO)

        R_CO = RotationMatrix(np.column_stack([x_CO, y_CO, z_CO]))
        X_CO = RigidTransform(R_CO, p_CO)
        X_OC = X_CO.inverse()


        # offset the tag T to center of T object
        offset = np.array([0.0, -0.107, 0.0])  # meters
        X_CT = X_CT.multiply(RigidTransform(RotationMatrix.Identity(), offset))

        # --- Object pose in workspace ---
        X_OT = X_OC.multiply(X_CT)
        R_OT = X_OT.rotation().matrix()
        p_OT = X_OT.translation()

        # NOTE: keeping original offset behavior for now.
        pos = p_OT[:2] - np.array([-0.75, 0.04])
        angle = np.arctan2(R_OT[1, 0], R_OT[0, 0])
        # normalize angle to [0, 2pi)
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        pose = np.concatenate((pos, [angle]))

        # if self._last_pose is not None:
        #     pose_diff = pose - self._last_pose
        #     diff_norm = np.linalg.norm(pose_diff)
        #     print(f"Pose diff: {pose_diff}, norm: {diff_norm}")
        #     if diff_norm > 0.05:
        #         print(f"Warning: large jump in detected pose from {self._last_pose} to {pose}. Keeping last pose.")
        #         return self._last_pose
        self._last_pose = pose
        return pose


class PlanarPositionDetector(LeafSystem):
    """Drake wrapper around PlanarPoseDetectorAPI (keeps existing behavior)."""

    def __init__(self, max_stored_images: int = 999):
        super().__init__()

        self._api = PlanarPoseDetectorAPI(max_stored_images=max_stored_images)

        # Initialize discrete state from the first successfully detected pose.
        pose0 = self._api.get_planar_pose(blocking=True, store_image=False)
        if pose0 is None:
            raise RuntimeError("Missing tags and thus cannot detect position")

        state_index = self.DeclareDiscreteState(pose0)
        self.DeclareStateOutputPort("planar_position", state_index)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / float(self._api.fps),
            offset_sec=0.0,
            update=self.DiscreteUpdate,
        )

    def Close(self):
        self._api.close()

    def DiscreteUpdate(self, context, discrete_values):
        # Non-blocking: update only if we got a new frame and tags are visible.
        planar_pos = self._api.get_planar_pose(blocking=False, store_image=True)
        if planar_pos is not None:
            discrete_values.set_value(planar_pos)

    def SaveImages(self, filename):
        self._api.save_images(filename)


if __name__ == "__main__":
    detector = PlanarPoseDetectorAPI()
    n_steps = 50
    try:
        start_time = time.time()
        for step in range(n_steps):
            pose = detector.get_planar_pose(blocking=True, store_image=True)
            if pose is not None:
                print(f"[step {step}] Detected planar pose (m, m, rad): {pose}")
            else:
                print(f"[step {step}] No tags detected.")
        elapsed = time.time() - start_time
        print(f"Finished {n_steps} steps in {elapsed:.2f} seconds ({n_steps/elapsed:.2f} steps/sec)")
    finally:
        out_dir = "output/test"
        os.makedirs(out_dir, exist_ok=True)
        detector.save_images(out_dir)
        detector.close()
