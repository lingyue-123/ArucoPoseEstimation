"""
位姿平滑器 | Pose Smoother

整合自 aruco_pose_from_hkws_cam_multi_aruco.py 中的：
- SLERP 旋转平滑（marker_rot_cache）
- EMA 平移平滑（t_filt 逻辑）
- 速度估计（marker_velocity_cache）
- 异常帧门控（marker_anomaly_cache）

消除 5 个全局字典，改为 MarkerPoseSmoother 实例属性。
"""

import logging
from collections import deque
from typing import Optional, Dict, List

import cv2
import numpy as np

from robovision.geometry.transforms import (
    rotmat_to_quat,
    quat_to_rotmat,
    quat_slerp,
    rotation_angle_deg,
)

logger = logging.getLogger(__name__)


class MarkerPoseSmoother:
    """
    单个 ArUco Marker 的位姿平滑与异常检测。

    集成：
    - SLERP 旋转平滑
    - EMA 平移平滑
    - 速度估计（用于自适应阈值）
    - 异常帧门控（WARN/REJECT/HOLD）

    Args:
        rot_alpha_normal: 正常状态旋转 SLERP 步长
        rot_alpha_cautious: 警告状态旋转 SLERP 步长
        trans_alpha_normal: 正常状态平移 EMA 步长
        trans_alpha_cautious: 警告状态平移 EMA 步长
        velocity_window: 速度估计滑动窗口大小
        motion_threshold: 判定为运动状态的速度阈值（毫米/帧）
        reproj_warn_px: 重投影误差警告阈值（像素）
        reproj_max_px: 重投影误差拒绝阈值（像素）
        angle_warn_deg: 角度跳变警告阈值（度）
        angle_max_deg: 角度跳变拒绝阈值（度）
        trans_warn_mm: 平移跳变警告阈值（毫米）
        trans_max_mm: 平移跳变拒绝阈值（毫米）
        motion_thresh_mult: 运动状态下阈值放大倍数
        anomaly_confirm: 连续异常帧数触发 HOLD
    """

    def __init__(
        self,
        rot_alpha_normal: float = 0.35,
        rot_alpha_cautious: float = 0.15,
        trans_alpha_normal: float = 0.40,
        trans_alpha_cautious: float = 0.15,
        velocity_window: int = 5,
        motion_threshold: float = 5.0,
        reproj_warn_px: float = 8.0,
        reproj_max_px: float = 20.0,
        angle_warn_deg: float = 15.0,
        angle_max_deg: float = 40.0,
        trans_warn_mm: float = 30.0,
        trans_max_mm: float = 120.0,
        motion_thresh_mult: float = 2.0,
        anomaly_confirm: int = 3,
    ):
        # 参数
        self._rot_a_n = rot_alpha_normal
        self._rot_a_c = rot_alpha_cautious
        self._trans_a_n = trans_alpha_normal
        self._trans_a_c = trans_alpha_cautious
        self._vel_window = velocity_window
        self._motion_thr = motion_threshold
        self._reproj_warn = reproj_warn_px
        self._reproj_max = reproj_max_px
        self._ang_warn = angle_warn_deg
        self._ang_max = angle_max_deg
        self._trans_warn = trans_warn_mm
        self._trans_max = trans_max_mm
        self._motion_mult = motion_thresh_mult
        self._anomaly_confirm = anomaly_confirm

        # 状态（替代原全局字典）
        self._q_filt: Optional[np.ndarray] = None          # 滤波后旋转四元数
        self._t_filt: Optional[np.ndarray] = None          # 滤波后平移向量
        self._rvec_cache: Optional[np.ndarray] = None      # 上帧 rvec（用于初始猜测）
        self._tvec_cache: Optional[np.ndarray] = None      # 上帧 tvec
        self._reproj_cache: float = 0.0                    # 上帧重投影误差
        self._velocity_hist: deque = deque(maxlen=velocity_window)
        self._anomaly_count: int = 0

    @property
    def has_prior(self) -> bool:
        """是否已有历史位姿（用于 PnP 初始猜测）。"""
        return self._rvec_cache is not None

    @property
    def prior_rvec(self) -> Optional[np.ndarray]:
        return self._rvec_cache.copy() if self._rvec_cache is not None else None

    @property
    def prior_tvec(self) -> Optional[np.ndarray]:
        return self._tvec_cache.copy() if self._tvec_cache is not None else None

    def smooth(
        self,
        rvec_m2c: np.ndarray,
        tvec_m2c: np.ndarray,
        reproj_err: float,
        method: str,
    ) -> dict:
        """
        对新的 PnP 解进行异常检测 + 位姿平滑。

        Args:
            rvec_m2c: 新的旋转向量，shape (3, 1)
            tvec_m2c: 新的平移向量，shape (3, 1)
            reproj_err: 重投影误差（像素）
            method: PnP 方法名（用于日志）

        Returns:
            dict 含:
              - rvec_m2c: 平滑后旋转向量
              - tvec_m2c: 平滑后平移向量
              - R_m2c: 平滑后旋转矩阵
              - reproj_err: 重投影误差
              - status: 'OK' | 'WARN' | 'HOLD'
              - method: 方法标签
        """
        R_m2c, _ = cv2.Rodrigues(rvec_m2c)
        t_m2c = tvec_m2c.flatten()

        # ——— 异常检测 ———
        in_motion = self._is_in_motion()
        mult = self._motion_mult if in_motion else 1.0
        anomaly_level = 'NORMAL'
        warn_reasons = []
        reject_reasons = []

        if reproj_err > self._reproj_max * mult:
            anomaly_level = 'REJECT'
            reject_reasons.append(f"reproj={reproj_err:.1f}px")
        elif reproj_err > self._reproj_warn * mult:
            anomaly_level = 'WARN'
            warn_reasons.append(f"reproj={reproj_err:.1f}px")

        if self._rvec_cache is not None:
            R_prev, _ = cv2.Rodrigues(self._rvec_cache)
            t_prev = self._tvec_cache.flatten()
            ang_jump = rotation_angle_deg(R_prev, R_m2c)
            trans_jump = float(np.linalg.norm(t_m2c - t_prev))

            if ang_jump > self._ang_max * mult:
                anomaly_level = 'REJECT'
                reject_reasons.append(f"ang={ang_jump:.1f}°")
            elif ang_jump > self._ang_warn * mult and anomaly_level != 'REJECT':
                anomaly_level = 'WARN'
                warn_reasons.append(f"ang={ang_jump:.1f}°")

            if trans_jump > self._trans_max * mult:
                anomaly_level = 'REJECT'
                reject_reasons.append(f"trans={trans_jump:.0f}mm")
            elif trans_jump > self._trans_warn * mult and anomaly_level != 'REJECT':
                anomaly_level = 'WARN'
                warn_reasons.append(f"trans={trans_jump:.0f}mm")

        # 更新异常计数
        is_anomaly = (anomaly_level == 'REJECT')
        if is_anomaly:
            self._anomaly_count += 1
        else:
            self._anomaly_count = 0
        should_hold = self._anomaly_count >= self._anomaly_confirm and self._rvec_cache is not None

        # ——— HOLD：使用上帧值 ———
        if should_hold:
            logger.debug("位姿 HOLD: %s", ', '.join(reject_reasons))
            rvec_m2c = self._rvec_cache.copy()
            tvec_m2c = self._tvec_cache.copy()
            reproj_err = self._reproj_cache
            R_m2c, _ = cv2.Rodrigues(rvec_m2c)
            t_m2c = tvec_m2c.flatten()
            status = 'HOLD'
            method_tag = 'HOLD'
            rot_alpha = 0.0
            trans_alpha = 0.0
        else:
            self._update_velocity(t_m2c)
            if anomaly_level == 'WARN':
                rot_alpha = self._rot_a_c
                trans_alpha = self._trans_a_c
                logger.debug("位姿 WARN: %s -> 保守滤波", ', '.join(warn_reasons))
                status = 'WARN'
                method_tag = f"{method}!"
            else:
                rot_alpha = self._rot_a_n
                trans_alpha = self._trans_a_n
                status = 'OK'
                method_tag = method

        # ——— SLERP 旋转平滑 ———
        q_new = rotmat_to_quat(R_m2c)
        if self._q_filt is None:
            self._q_filt = q_new.copy()
        else:
            self._q_filt = quat_slerp(self._q_filt, q_new, rot_alpha)
        R_filt = quat_to_rotmat(self._q_filt)
        rvec_filt, _ = cv2.Rodrigues(R_filt)

        # ——— EMA 平移平滑 ———
        t_prev_f = self._t_filt if self._t_filt is not None else t_m2c
        t_filt = (1.0 - trans_alpha) * t_prev_f + trans_alpha * t_m2c
        self._t_filt = t_filt
        tvec_filt = t_filt.reshape(3, 1)

        # 更新缓存
        if not should_hold:
            self._rvec_cache = rvec_filt.copy()
            self._tvec_cache = tvec_filt.copy()
            self._reproj_cache = reproj_err

        return {
            'rvec_m2c': rvec_filt,
            'tvec_m2c': tvec_filt,
            'R_m2c': R_filt,
            'reproj_err': reproj_err,
            'status': status,
            'method': method_tag,
        }

    def _update_velocity(self, t_new: np.ndarray) -> None:
        self._velocity_hist.append(t_new.copy())

    def _is_in_motion(self) -> bool:
        if len(self._velocity_hist) < 2:
            return False
        velocities = [
            np.linalg.norm(self._velocity_hist[i] - self._velocity_hist[i-1])
            for i in range(1, len(self._velocity_hist))
        ]
        return float(np.mean(velocities)) > self._motion_thr

    def reset(self) -> None:
        """重置所有状态（Marker 丢失后调用）。"""
        self._q_filt = None
        self._t_filt = None
        self._rvec_cache = None
        self._tvec_cache = None
        self._reproj_cache = 0.0
        self._velocity_hist.clear()
        self._anomaly_count = 0


class MarkerSmootherCache:
    """
    多 Marker 位姿平滑器缓存管理器。

    替代原脚本中的多个全局字典：
    - marker_rot_cache
    - marker_pose_cache
    - marker_velocity_cache
    - marker_anomaly_cache

    Args:
        lost_threshold: 丢失帧数超过此值时重置平滑器
        smoother_kwargs: 传递给 MarkerPoseSmoother 的参数
    """

    def __init__(self, lost_threshold: int = 30, **smoother_kwargs):
        self._lost_threshold = lost_threshold
        self._smoother_kwargs = smoother_kwargs
        self._smoothers: Dict[int, MarkerPoseSmoother] = {}
        self._lost_counts: Dict[int, int] = {}

    def get_smoother(self, marker_id: int) -> MarkerPoseSmoother:
        """获取或创建指定 Marker 的平滑器。"""
        if marker_id not in self._smoothers:
            self._smoothers[marker_id] = MarkerPoseSmoother(**self._smoother_kwargs)
            self._lost_counts[marker_id] = 0
        return self._smoothers[marker_id]

    def mark_detected(self, detected_ids: List[int]) -> None:
        """更新丢失计数，清除超阈值的 Marker。"""
        to_remove = []
        for mid in list(self._smoothers.keys()):
            if mid not in detected_ids:
                self._lost_counts[mid] = self._lost_counts.get(mid, 0) + 1
                if self._lost_counts[mid] > self._lost_threshold:
                    to_remove.append(mid)
            else:
                self._lost_counts[mid] = 0
        for mid in to_remove:
            self._smoothers[mid].reset()
            del self._smoothers[mid]
            del self._lost_counts[mid]
            logger.debug("Marker %d 丢失超过阈值，清除平滑器缓存", mid)

    def clear(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()
        self._smoothers.clear()
        self._lost_counts.clear()
