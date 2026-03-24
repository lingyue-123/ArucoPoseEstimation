"""
图像保存工具 | Image Saver

带时间戳和序号的图像保存，支持目录管理。
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageSaver:
    """
    带时间戳的图像保存器。

    Args:
        output_dir: 保存目录
        prefix: 文件名前缀（可选）
    """

    def __init__(self, output_dir: str, prefix: str = ""):
        self._dir = output_dir
        self._prefix = prefix
        self._count = 0
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            logger.debug("创建图像保存目录: %s", self._dir)

    def save(self, frame: np.ndarray, extra_tag: str = "") -> Optional[str]:
        """
        保存一帧图像。

        Args:
            frame: BGR numpy array
            extra_tag: 附加文件名标签

        Returns:
            保存路径（成功）或 None（失败）
        """
        self._ensure_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        parts = [f"{self._count:04d}", timestamp]
        if self._prefix:
            parts.insert(0, self._prefix)
        if extra_tag:
            parts.append(extra_tag)
        filename = '_'.join(parts) + '.png'
        filepath = os.path.join(self._dir, filename)
        try:
            cv2.imwrite(filepath, frame)
            self._count += 1
            logger.debug("图像已保存: %s", filepath)
            return filepath
        except Exception as e:
            logger.warning("图像保存失败: %s", e)
            return None

    def clear(self) -> None:
        """清空目录并重置计数。"""
        if os.path.exists(self._dir):
            shutil.rmtree(self._dir)
        os.makedirs(self._dir)
        self._count = 0
        logger.info("图像目录已清空: %s", self._dir)

    @property
    def count(self) -> int:
        return self._count

    @property
    def output_dir(self) -> str:
        return self._dir
