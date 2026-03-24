"""
Mech-Eye 3D 相机封装 | Mech-Eye 3D Camera

封装 mecheye Python SDK (2.5.x)，提供标准 CameraInterface 接口。
仅获取 2D 彩色图像（用于 ArUco 检测）；3D 点云按需扩展。
"""

import logging
import sys
from typing import Optional, List, Tuple

import cv2
import numpy as np

from .base import CameraInterface, CameraIntrinsics

logger = logging.getLogger(__name__)

# mecheye SDK 2.5.x 安装在系统 Python 3.8 路径，需手动加入 sys.path
# 使用 append 保证 conda 的 numpy 优先于系统 numpy
_MECHEYE_SYSTEM_PATH = '/usr/local/lib/python3.8/dist-packages'
if _MECHEYE_SYSTEM_PATH not in sys.path:
    sys.path.append(_MECHEYE_SYSTEM_PATH)


class MechEyeCamera(CameraInterface):
    """
    Mech-Eye 3D 相机（mecheye Python SDK 2.5.x）。

    Args:
        intrinsics: 相机内参（或从相机自动读取）
        ip: 相机 IP 地址
    """

    def __init__(self, intrinsics: CameraIntrinsics, ip: str = ""):
        self._intrinsics = intrinsics
        self._ip = ip
        self._camera = None

    def open(self) -> None:
        try:
            from mecheye.area_scan_3d_camera import Camera
        except ImportError as e:
            raise RuntimeError(f"Mech-Eye SDK 未安装: {e}")

        try:
            self._camera = Camera()
            if self._ip:
                ret = self._camera.connect(self._ip)
            else:
                # 自动发现第一台相机
                infos = self._camera.discover_cameras()
                if len(infos) == 0:
                    raise RuntimeError("未发现 Mech-Eye 相机")
                ret = self._camera.connect(infos[0])
            if not ret.is_ok():
                raise RuntimeError(f"Mech-Eye 连接失败: {ret.error_description()}")
            logger.info("Mech-Eye 相机已连接: %s", self._ip or "auto")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Mech-Eye 初始化失败: {e}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        采集一帧 2D 彩色图像。

        Returns:
            (True, BGR numpy array) 或 (False, None)
        """
        if self._camera is None:
            return False, None
        try:
            from mecheye.area_scan_3d_camera import Frame2D
            frame_2d = Frame2D()
            ret = self._camera.capture_2d(frame_2d)
            if not ret.is_ok():
                logger.warning("Mech-Eye 采图失败: %s", ret.error_description())
                return False, None
            color_img = frame_2d.get_color_image()
            # data() 返回 numpy array (H, W, 3) BGR
            bgr = np.array(color_img.data())
            if bgr.ndim == 3 and bgr.shape[2] == 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            return True, bgr
        except Exception as e:
            logger.error("Mech-Eye 读帧异常: %s", e)
            return False, None

    def read_frame_from_file(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        """从图像文件读取（用于离线处理 Mech-Eye 图片）。"""
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        return True, img

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.disconnect()
            except Exception:
                pass
            self._camera = None
        logger.info("Mech-Eye 相机已释放")

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics


class MechEyeImageFolderCamera(CameraInterface):
    """
    Mech-Eye 图片文件夹相机（离线模式）。

    从指定文件夹按顺序读取图像文件，模拟相机行为。
    替代原 aruco_pose_from_mech_image_folder_multi_aruco.py 中的文件夹迭代逻辑。

    Args:
        intrinsics: 相机内参
        folder: 图像文件夹路径
        extensions: 支持的图像扩展名
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        folder: str,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.bmp'),
    ):
        self._intrinsics = intrinsics
        self._folder = folder
        self._extensions = extensions
        self._files: List[str] = []
        self._index = 0

    def open(self) -> None:
        import os
        all_files = sorted(os.listdir(self._folder))
        self._files = [
            os.path.join(self._folder, f)
            for f in all_files
            if any(f.lower().endswith(ext) for ext in self._extensions)
        ]
        self._index = 0
        logger.info("Mech-Eye 图片文件夹模式: %s，共 %d 张", self._folder, len(self._files))

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._index >= len(self._files):
            return False, None
        path = self._files[self._index]
        self._index += 1
        img = cv2.imread(path)
        if img is None:
            logger.warning("无法读取图像: %s", path)
            return False, None
        return True, img

    def close(self) -> None:
        pass

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def total_files(self) -> int:
        return len(self._files)
