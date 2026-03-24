"""
位姿文件读写 | Pose File IO

管理 robot_tcp.txt / aruco_pose_idN.txt 等位姿文件。
格式：每行 x,y,z,rx,ry,rz（逗号或空白分隔均支持）
"""

import logging
import os
import re
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class PoseFileWriter:
    """
    CSV 格式位姿文件写入器。

    Args:
        path: 文件路径
        append: True 追加模式，False 覆盖模式
        precision: 小数位数
    """

    def __init__(self, path: str, append: bool = True, precision: int = 3):
        self._path = path
        self._append = append
        self._precision = precision
        self._count = 0
        self._file = None

    def open(self) -> None:
        mode = 'a' if self._append else 'w'
        self._file = open(self._path, mode, encoding='utf-8')

    def write(self, pose: Union[list, np.ndarray]) -> None:
        """写入一行位姿数据。"""
        if self._file is None:
            self.open()
        p = precision = self._precision
        x, y, z, rx, ry, rz = pose
        line = f"{x:.{p}f},{y:.{p}f},{z:.{p}f},{rx:.4f},{ry:.4f},{rz:.4f}\n"
        self._file.write(line)
        self._count += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def clear(self) -> None:
        """清空文件内容并重置计数。"""
        with open(self._path, 'w', encoding='utf-8') as f:
            f.write("")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self) -> 'PoseFileWriter':
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_pose_file(path: str, skip_empty: bool = True) -> np.ndarray:
    """
    加载 CSV 格式位姿文件。

    Args:
        path: 文件路径
        skip_empty: 跳过空行

    Returns:
        shape (N, 6) numpy array，每行 [x, y, z, rx, ry, rz]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"位姿文件不存在: {path}")
    poses = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line and skip_empty:
                continue
            values = [float(v) for v in re.split(r'[,\s]+', line) if v]
            if len(values) == 6:
                poses.append(values)
            else:
                logger.warning("跳过格式异常行: %s", line)
    if not poses:
        raise ValueError(f"位姿文件为空或格式错误: {path}")
    return np.array(poses, dtype=np.float64)


def save_pose_file(path: str, poses: np.ndarray, precision: int = 3) -> None:
    """
    保存位姿数据到 CSV 文件。

    Args:
        path: 文件路径
        poses: shape (N, 6) array
        precision: 小数位数
    """
    with open(path, 'w', encoding='utf-8') as f:
        for pose in poses:
            x, y, z, rx, ry, rz = pose
            p = precision
            f.write(f"{x:.{p}f},{y:.{p}f},{z:.{p}f},{rx:.4f},{ry:.4f},{rz:.4f}\n")
    logger.info("位姿文件已保存: %s (%d 行)", path, len(poses))
