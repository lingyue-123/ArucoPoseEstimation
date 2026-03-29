"""
RTSP 网络相机封装 | RTSP Network Camera

带后台抓帧线程，避免主线程因解码卡顿导致延迟。
适用于 IP 摄像头（网络相机 / RTSP 流）。
"""

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraInterface, CameraIntrinsics

logger = logging.getLogger(__name__)


class RTSPCamera(CameraInterface):
    """
    RTSP 流相机，后台线程持续抓取最新帧。

    Args:
        intrinsics: 相机内参
        url: RTSP URL，如 "rtsp://admin:@192.168.1.10:554/stream1"
        buffer_size: OpenCV 内部缓冲帧数（设为 1 减少延迟）
    """

    def __init__(self, intrinsics: CameraIntrinsics, url: str, buffer_size: int = 1):
        self._intrinsics = intrinsics
        self._url = url
        self._buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._url)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 RTSP 流: {self._url}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        logger.info("RTSP 相机已连接: %s，后台抓帧线程已启动", self._url)

    def _grab_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if ok:
                with self._frame_lock:
                    self._latest_frame = frame.copy()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("RTSP 相机已释放")

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics
