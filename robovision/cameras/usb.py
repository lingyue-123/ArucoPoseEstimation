"""
USB 相机封装 | USB Camera (V4L2 via OpenCV)

封装 cv2.VideoCapture，支持 4K USB 相机。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraInterface, CameraIntrinsics

logger = logging.getLogger(__name__)


class USBCamera(CameraInterface):
    """
    4K USB 相机（通过 V4L2 / OpenCV 访问）。

    Args:
        intrinsics: 相机内参
        device_id: 设备 ID（如 8 对应 /dev/video8）
        width: 请求分辨率宽度
        height: 请求分辨率高度
        fps: 请求帧率
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        device_id: int = 0,
        width: int = 3840,
        height: int = 2160,
        fps: int = 30,
    ):
        self._intrinsics = intrinsics
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 USB 相机 device_id={self._device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("USB 相机已打开: device=%d, 实际分辨率=%dx%d", self._device_id, actual_w, actual_h)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        return ok, frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("USB 相机已释放")

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics
