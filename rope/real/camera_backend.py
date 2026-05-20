from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class OpenCVCamera:
    index: int = 0
    output_width: int = 224
    output_height: int = 224
    capture_width: int | None = None
    capture_height: int | None = None
    warmup_frames: int = 10

    def __post_init__(self) -> None:
        self._cv2 = None
        self._cap = None

    def connect(self) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.index)
        if self.capture_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        if self.capture_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}.")
        for _ in range(max(0, self.warmup_frames)):
            self._cap.read()
            time.sleep(0.01)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def read_rgb_224(self) -> np.ndarray:
        if self._cap is None or self._cv2 is None:
            raise RuntimeError("Camera is not connected.")
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Camera read failed.")
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        return self._center_crop_resize(frame_rgb)

    def _center_crop_resize(self, frame_rgb: np.ndarray) -> np.ndarray:
        height, width = frame_rgb.shape[:2]
        side = min(height, width)
        y0 = (height - side) // 2
        x0 = (width - side) // 2
        cropped = frame_rgb[y0 : y0 + side, x0 : x0 + side]
        resized = self._cv2.resize(
            cropped,
            (self.output_width, self.output_height),
            interpolation=self._cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.uint8)

