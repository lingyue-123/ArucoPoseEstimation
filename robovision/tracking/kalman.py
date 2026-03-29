"""
角点卡尔曼滤波器 | Corner Kalman Filter

整合自 7 个 ArUco 脚本中的 init_kalman_filter() / apply_kalman_filter()。
消除全局字典 marker_kf_cache，改为 KalmanCornerFilter 实例属性。
"""

import logging
from typing import Optional, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KalmanCornerFilter:
    """
    单个 ArUco Marker 的角点卡尔曼滤波器。

    状态向量：4个角点的 (x, y) 坐标，共 8 维。
    不含速度项（匀速假设对角点微小运动已足够）。

    Args:
        state_dim: 状态维度（默认 8 = 4角点 × 2坐标）
        measure_dim: 测量维度（默认 8）
        process_noise: 过程噪声协方差对角线值
        measurement_noise: 测量噪声协方差对角线值
        error_cov_init: 初始误差协方差对角线值
    """

    def __init__(
        self,
        state_dim: int = 8,
        measure_dim: int = 8,
        process_noise: float = 0.1,
        measurement_noise: float = 0.05,
        error_cov_init: float = 1.0,
    ):
        self._kf = cv2.KalmanFilter(state_dim, measure_dim)
        self._kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        self._kf.measurementMatrix = np.eye(measure_dim, dtype=np.float32)
        self._kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * measurement_noise
        self._kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * error_cov_init
        self._initialized = False
        self.lost_count: int = 0

    def initialize(self, corners: np.ndarray) -> None:
        """用第一帧角点初始化状态。"""
        self._kf.statePost = corners.flatten().astype(np.float32).reshape(-1, 1)
        self._initialized = True
        self.lost_count = 0

    def update(self, corners: np.ndarray) -> np.ndarray:
        """
        用新测量值更新滤波器。

        Args:
            corners: 当前帧角点，shape (4, 2)

        Returns:
            滤波后角点，shape (4, 2)，dtype float64
        """
        self._kf.predict()
        measurement = corners.flatten().astype(np.float32).reshape(-1, 1)
        self._kf.correct(measurement)
        return self._kf.statePost.flatten().astype(np.float64).reshape(4, 2)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class MarkerKalmanCache:
    """
    多 Marker 卡尔曼滤波器缓存管理器。

    替代原脚本中的全局字典 marker_kf_cache，
    封装丢失计数、自动清理等逻辑。

    Args:
        lost_threshold: 丢失帧数超过此值时清除 Marker 缓存
        kf_kwargs: 传递给 KalmanCornerFilter 的参数
    """

    def __init__(self, lost_threshold: int = 30, **kf_kwargs):
        self._lost_threshold = lost_threshold
        self._kf_kwargs = kf_kwargs
        self._filters: Dict[int, KalmanCornerFilter] = {}

    def process(self, marker_id: int, raw_corners: np.ndarray) -> np.ndarray:
        """
        对指定 Marker 的角点应用卡尔曼滤波。

        Args:
            marker_id: Marker ID
            raw_corners: 原始检测角点，shape (4, 2)

        Returns:
            滤波后角点，shape (4, 2)
        """
        if marker_id not in self._filters:
            kf = KalmanCornerFilter(**self._kf_kwargs)
            kf.initialize(raw_corners)
            self._filters[marker_id] = kf
            return raw_corners.copy()
        else:
            kf = self._filters[marker_id]
            kf.lost_count = 0
            return kf.update(raw_corners)

    def mark_detected(self, detected_ids: List[int]) -> None:
        """标记本帧检测到的 Marker，递增未检测到的丢失计数，清理超阈值的缓存。"""
        to_remove = []
        for mid, kf in self._filters.items():
            if mid not in detected_ids:
                kf.lost_count += 1
                if kf.lost_count > self._lost_threshold:
                    to_remove.append(mid)
        for mid in to_remove:
            del self._filters[mid]
            logger.debug("Marker %d 丢失超过 %d 帧，清除缓存", mid, self._lost_threshold)

    def clear(self, marker_id: Optional[int] = None) -> None:
        """清除指定或全部 Marker 缓存。"""
        if marker_id is not None:
            self._filters.pop(marker_id, None)
        else:
            self._filters.clear()

    def __contains__(self, marker_id: int) -> bool:
        return marker_id in self._filters
