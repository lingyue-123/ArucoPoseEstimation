"""
自动曝光控制器 | Auto Exposure Controller

户外光照鲁棒策略：
  L1 硬件AE: 利用相机内置 Auto Exposure (Continuous) 每帧实时调整曝光
  L2 软件监控: 每 N 帧测量图像亮度，按需调整 AE 参数上限；检测到 marker 时锁定 AE ROI

用法:
    from robovision.vision.auto_exposure import AutoExposureController

    ae = AutoExposureController(camera, target_brightness=128)
    ae.setup()

    while True:
        ok, frame = camera.read_frame()
        ae.measure_frame(frame)                    # 放检测前：实时亮度显示
        markers = detector.detect(frame)
        ae.adjust_after_detect(bool(markers), markers)  # 放检测后：ROI锁定 + nudge上限
"""

import logging
import time
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_BRIGHTNESS = 128
_DEFAULT_DEADBAND = 12
_DEFAULT_ADJUST_INTERVAL = 15
_DEFAULT_EXPOSURE_LIMIT_MS = 50
_DEFAULT_GAIN_LIMIT_DB = 12


class AutoExposureController:
    def __init__(
        self,
        camera,
        target_brightness: int = _DEFAULT_TARGET_BRIGHTNESS,
        deadband: int = _DEFAULT_DEADBAND,
        adjust_interval: int = _DEFAULT_ADJUST_INTERVAL,
        exposure_limit_ms: float = _DEFAULT_EXPOSURE_LIMIT_MS,
        gain_limit_db: float = _DEFAULT_GAIN_LIMIT_DB,
    ):
        self._camera = camera
        self._target = target_brightness
        self._deadband = deadband
        self._adjust_interval = adjust_interval
        self._exposure_limit_ms = exposure_limit_ms
        self._gain_limit_db = gain_limit_db

        self._frame_count = 0
        self._last_measured: Optional[float] = None
        self._next_roi: Optional[np.ndarray] = None
        self._last_adjust_time = 0.0

        self._hwe_supported = None
        self._ae_roi_supported = None
        self._ae_roi_active = False

    def setup(self) -> bool:
        """初始化: 启用硬件自动曝光 + 自动增益 + 设置上下限。"""
        ok = True

        if hasattr(self._camera, 'setup_auto_exposure'):
            ok = self._camera.setup_auto_exposure(
                target_brightness=self._target,
                exposure_limit_ms=self._exposure_limit_ms,
                gain_limit_db=self._gain_limit_db,
            )
            self._hwe_supported = True
            logger.info("Hardware AE enabled: limits exp<=%.1fms gain<=%.1fdB target=%d",
                        self._exposure_limit_ms, self._gain_limit_db, self._target)
        else:
            self._hwe_supported = False
            logger.info("Camera does not support setup_auto_exposure, software-only mode")

        return ok

    def measure_brightness(self, frame: np.ndarray, roi: Optional[np.ndarray] = None) -> float:
        """
        测量图像亮度（使用中值，对高光点/阴影鲁棒）。

        Args:
            frame: BGR 或灰度图像
            roi: 可选裁剪区域 [[x1,y1],[x2,y2]] 像素坐标

        Returns:
            float: 中值灰度值 (0-255)
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if roi is not None and len(roi) >= 2:
            x1, y1 = max(0, int(roi[0][0])), max(0, int(roi[0][1]))
            x2, y2 = min(gray.shape[1], int(roi[1][0])), min(gray.shape[0], int(roi[1][1]))
            if x2 > x1 and y2 > y1:
                gray = gray[y1:y2, x1:x2]

        return float(np.median(gray))

    def _marker_roi_from_detections(self, detected_markers: Optional[Dict[int, dict]]) -> Optional[np.ndarray]:
        """从检测到的 marker 角点计算合并 ROI。"""
        if not detected_markers:
            return None

        all_pts = []
        for data in detected_markers.values():
            corners = data.get('filtered_corners') or data.get('raw_corners')
            if corners is not None:
                all_pts.append(corners)
        if not all_pts:
            return None

        pts = np.vstack(all_pts)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        margin = 20
        return np.array([
            [x1 - margin, y1 - margin],
            [x2 + margin, y2 + margin],
        ])

    def measure_frame(self, frame: np.ndarray, roi: Optional[np.ndarray] = None) -> float:
        """
        测量并缓存当前帧亮度（放检测前调用，用于实时显示）。

        Returns:
            float: 中值灰度值 (0-255)
        """
        brightness = self.measure_brightness(frame, roi=roi)
        self._last_measured = brightness
        self._frame_count += 1
        return brightness

    def adjust_after_detect(self, detection_ok: bool = True,
                            detected_markers: Optional[Dict[int, dict]] = None) -> None:
        """
        根据检测结果做 AE 参数调整（放检测后调用，影响后续帧）。

        - detection_ok=True: 锁定 AE ROI 到 marker 区域，让硬件AE为marker曝光
        - detection_ok=False: 清除 AE ROI，回到全画面AE
        """
        if detection_ok and detected_markers:
            self._lock_ae_roi(detected_markers)
        else:
            self._release_ae_roi()

        self._frame_count += 1
        if self._frame_count % self._adjust_interval != 0:
            return

        self._last_adjust_time = time.time()

        if self._last_measured is None:
            return

        error = self._last_measured - self._target
        if abs(error) <= self._deadband:
            return

        self._nudge_ae_limits(error) if self._hwe_supported else self._servo_manual_exposure(error)

    def _lock_ae_roi(self, detected_markers: Dict[int, dict]) -> None:
        """将相机 AE ROI 锁定到当前检测到的 marker 区域。"""
        roi = self._marker_roi_from_detections(detected_markers)
        if roi is None:
            return

        x1, y1 = max(0, int(roi[0][0])), max(0, int(roi[0][1]))
        x2, y2 = int(roi[1][0]), int(roi[1][1])
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return

        if self._ae_roi_supported is None:
            self._ae_roi_supported = self._camera.set_ae_roi(x1, y1, w, h)

        if self._ae_roi_supported and not self._ae_roi_active:
            self._camera.set_ae_roi(x1, y1, w, h)
            self._ae_roi_active = True
        elif self._ae_roi_supported:
            self._camera.set_ae_roi(x1, y1, w, h)

        self._next_roi = roi

    def _release_ae_roi(self) -> None:
        """释放 AE ROI，回到全画面自动曝光。"""
        if self._ae_roi_active:
            self._camera.set_ae_roi(0, 0, 0, 0)
            self._ae_roi_active = False
            self._next_roi = None

    def _servo_manual_exposure(self, error: float) -> None:
        """AE ROI 不可用时的手动曝光伺服（基于 marker 亮度调节曝光时间）。"""
        try:
            cur_exp = self._camera.get_exposure_time()
            if cur_exp is None:
                return
        except Exception:
            return

        factor = 1.0 - error / 640.0
        factor = max(0.5, min(2.0, factor))
        new_exp = cur_exp * factor
        new_exp = max(100, min(self._exposure_limit_ms * 1000, new_exp))

        if abs(new_exp - cur_exp) > cur_exp * 0.05:
            try:
                self._camera.set_exposure_auto(False)
                self._camera.set_exposure_time(new_exp)
                logger.debug("Manual exp servo: %.1fus -> %.1fus (bri=%.0f target=%d)",
                             cur_exp, new_exp, self._last_measured, self._target)
            except Exception:
                pass

    def adjust(self, frame: np.ndarray, detection_ok: bool = True,
               detected_markers: Optional[Dict[int, dict]] = None) -> None:
        """兼容旧 API: 内部转为 measure_frame + adjust_after_detect。"""
        self.measure_frame(frame)
        self.adjust_after_detect(detection_ok, detected_markers)

    def _nudge_ae_limits(self, error: float) -> None:
        """
        微调硬件 AE 上下限。error>0 表示过亮，需降低上限；error<0 表示过暗，需放宽上限。
        仅在硬件 AE 支持时生效。
        """
        if not self._hwe_supported:
            return

        limit_key = "AutoExposureTimeUpperLimit"
        try:
            from ctypes import memset, byref, sizeof
            st = self._camera._mvs['MVCC_FLOATVALUE']()
            memset(byref(st), 0, sizeof(st))
            if self._camera._cam.MV_CC_GetFloatValue(limit_key, st) != 0:
                return
            cur_limit_us = st.fCurValue
        except Exception:
            return

        cur_limit_ms = cur_limit_us / 1000.0
        factor = 0.85 if error > 0 else 1.15
        new_limit_ms = cur_limit_ms * factor
        new_limit_ms = max(1.0, min(self._exposure_limit_ms * 2.0, new_limit_ms))

        if abs(new_limit_ms - cur_limit_ms) < 1.0:
            return

        try:
            self._camera._cam.MV_CC_SetFloatValue(limit_key, new_limit_ms * 1000.0)
            logger.debug("AE limits adjusted: exp_upper=%.1fms (was %.1fms) brightness=%.0f target=%d",
                         new_limit_ms, cur_limit_ms, self._last_measured, self._target)
        except Exception:
            pass

    def brightness(self) -> Optional[float]:
        """返回最近一次测量的亮度值，未测量时返回 None。"""
        return self._last_measured

    @property
    def roi_active(self) -> bool:
        return self._ae_roi_active
