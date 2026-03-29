"""
相机抽象层基类 | Camera Abstraction Layer Base

定义统一的相机接口，使上层算法对相机类型无感知。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    相机内参数据类。

    Attributes:
        camera_matrix: 3x3 内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist_coeffs: 畸变系数向量（5 或更多参数）
        width: 图像宽度（像素）
        height: 图像高度（像素）
        name: 相机名称（用于日志和配置识别）
    """
    camera_matrix: np.ndarray   # shape (3, 3)
    dist_coeffs: np.ndarray     # shape (N,)
    width: int = 0
    height: int = 0
    name: str = ""

    @classmethod
    def from_config(cls, cfg, name: str = "") -> 'CameraIntrinsics':
        """从 CameraIntrinsicsConfig 构造（来自 config/loader.py）。"""
        return cls(
            camera_matrix=cfg.camera_matrix.copy(),
            dist_coeffs=cfg.dist_coeffs.copy(),
            width=cfg.width,
            height=cfg.height,
            name=name,
        )


class CameraInterface(ABC):
    """
    相机统一接口（抽象基类）。

    所有相机实现（Hikvision、USB、RTSP、Mech-Eye）均继承此类，
    上层算法仅依赖此接口，不关心具体相机类型。

    支持 with 语句::

        with HikvisionCamera(intrinsics) as cam:
            ok, frame = cam.read_frame()
    """

    @abstractmethod
    def open(self) -> None:
        """打开相机连接并开始取流。"""
        ...

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像。

        Returns:
            (success, frame): success=True 时 frame 为 BGR numpy array，
                              success=False 时 frame 为 None
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """释放相机资源。"""
        ...

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """返回相机内参。"""
        ...

    def __enter__(self) -> 'CameraInterface':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        intr = self.get_intrinsics()
        return f"{self.__class__.__name__}(name={intr.name!r}, {intr.width}x{intr.height})"
