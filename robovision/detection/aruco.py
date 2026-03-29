"""
ArUco 检测器 | ArUco Detector

整合来自 7 个脚本的 detect_aruco() 函数，消除全局字典状态。
算法逻辑与原脚本完全一致：
- CLAHE 预处理 + 形态学增强
- 角点卡尔曼滤波（KalmanCornerFilter）
- 多方法 PnP 最优选择（solve_pnp_best）
- SLERP 旋转平滑 + EMA 平移平滑（MarkerPoseSmoother）
- 异常帧门控（WARN / REJECT / HOLD）

原 7 个脚本被本类替代：
- aruco_pose_from_hkws_cam_multi_aruco.py
- aruco_pose_from_hkws_cam_multi_aruco_without_jkrc.py
- aruco_pose_from_hkws_cam_multi_aruco_black_square_without_jkrc.py
- aruco_pose_from_web_cam_multi_aruco.py
- aruco_pose_from_mech_image_folder_multi_aruco.py
- aruco_pose_from_mech_image_folder.py
- aruco_pose_from_mech_image.py
"""

import logging
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from robovision.cameras.base import CameraIntrinsics
from robovision.tracking.kalman import MarkerKalmanCache
from robovision.tracking.pose_smoother import MarkerSmootherCache
from robovision.detection.pnp import solve_pnp_best
from robovision.geometry.transforms import rotmat_to_euler

logger = logging.getLogger(__name__)


class ArucoDetector:
    """
    ArUco 多标记检测器（含卡尔曼滤波 + 位姿平滑 + 异常检测）。

    所有状态保存为实例属性，支持多实例并发，无全局变量。

    Args:
        intrinsics: 相机内参
        valid_ids: 有效 Marker ID 列表
        marker_sizes: Marker ID -> 物理尺寸（米）的映射
        dictionary: ArUco 字典类型名称（如 'DICT_4X4_50'）
        use_kalman: 是否启用角点卡尔曼滤波
        lost_threshold: Marker 丢失阈值（帧数）
        kf_process_noise: 卡尔曼过程噪声
        kf_measurement_noise: 卡尔曼测量噪声
        smoother_kwargs: 传递给 MarkerPoseSmoother 的参数
    """

    # 粗检测阶段的最大宽度（超过此宽度的图像会被缩小）
    COARSE_MAX_WIDTH = 1280

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        valid_ids: Optional[List[int]] = None,
        marker_sizes: Optional[Dict[int, float]] = None,
        dictionary: str = 'DICT_4X4_50',
        use_kalman: bool = True,
        lost_threshold: int = 30,
        kf_process_noise: float = 0.1,
        kf_measurement_noise: float = 0.05,
        **smoother_kwargs,
    ):
        self._intrinsics = intrinsics
        self._valid_ids = set(valid_ids) if valid_ids is not None else {0, 1, 2}
        self._marker_sizes = marker_sizes or {0: 100.0, 1: 15.0, 2: 40.0}
        self._use_kalman = use_kalman
        self._use_smoother = True
        self._lost_threshold = lost_threshold

        # ArUco 字典 + 缓存检测器（避免每帧重建）
        dict_id = getattr(cv2.aruco, dictionary, cv2.aruco.DICT_4X4_50)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._aruco_detector = self._build_aruco_detector()

        # CLAHE 对象复用
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 卡尔曼滤波缓存
        self._kf_cache = MarkerKalmanCache(
            lost_threshold=lost_threshold,
            process_noise=kf_process_noise,
            measurement_noise=kf_measurement_noise,
        )

        # 位姿平滑缓存
        self._smoother_cache = MarkerSmootherCache(
            lost_threshold=lost_threshold,
            **smoother_kwargs,
        )

    @classmethod
    def from_config(cls, intrinsics: CameraIntrinsics, marker_cfg, detection_cfg) -> 'ArucoDetector':
        """
        从配置对象构造检测器。

        Args:
            intrinsics: 相机内参
            marker_cfg: MarkerConfig（来自 config/loader.py）
            detection_cfg: DetectionConfig（来自 config/loader.py）
        """
        return cls(
            intrinsics=intrinsics,
            valid_ids=marker_cfg.valid_ids,
            marker_sizes=marker_cfg.marker_sizes,
            dictionary=marker_cfg.dictionary,
            use_kalman=True,
            lost_threshold=detection_cfg.lost_threshold,
            kf_process_noise=detection_cfg.kf_process_noise,
            kf_measurement_noise=detection_cfg.kf_measurement_noise,
            rot_alpha_normal=detection_cfg.rot_slerp_alpha_normal,
            rot_alpha_cautious=detection_cfg.rot_slerp_alpha_cautious,
            trans_alpha_normal=detection_cfg.trans_ema_alpha_normal,
            trans_alpha_cautious=detection_cfg.trans_ema_alpha_cautious,
            velocity_window=detection_cfg.velocity_window_size,
            motion_threshold=detection_cfg.motion_threshold,
            reproj_warn_px=detection_cfg.reproj_warn_px,
            reproj_max_px=detection_cfg.reproj_max_px,
            angle_warn_deg=detection_cfg.angle_jump_warn_deg,
            angle_max_deg=detection_cfg.angle_jump_max_deg,
            trans_warn_mm=detection_cfg.trans_jump_warn_mm,
            trans_max_mm=detection_cfg.trans_jump_max_mm,
            motion_thresh_mult=detection_cfg.motion_threshold_multiplier,
            anomaly_confirm=detection_cfg.anomaly_confirm_frames,
        )

    def _build_aruco_detector(self):
        """构建 ArUco 检测器对象（缓存复用，避免每帧重建）。"""
        try:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 50
            params.cornerRefinementMinAccuracy = 0.001
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 53
            params.adaptiveThreshWinSizeStep = 4
            return cv2.aruco.ArucoDetector(self._aruco_dict, params)
        except AttributeError:
            return None  # 旧 API，fallback 在 _detect_markers 中处理

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """
        图像预处理：CLAHE + 高斯模糊 + 形态学操作。

        精简版：去掉 medianBlur（与 GaussianBlur 重复）和 Laplacian 锐化
        （ArUco 二值化检测不需要）。

        Args:
            gray: 灰度图像

        Returns:
            gray_morph: 预处理后的灰度图，用于 ArUco 检测
        """
        gray_clahe = self._clahe.apply(gray)
        gray_blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_morph = cv2.morphologyEx(gray_blur, cv2.MORPH_OPEN, kernel)
        gray_morph = cv2.morphologyEx(gray_morph, cv2.MORPH_CLOSE, kernel)
        return gray_morph

    def _detect_markers(self, gray_morph: np.ndarray):
        """调用 ArUco 检测（兼容新旧 API），复用缓存的检测器对象。"""
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(gray_morph)
        else:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_morph, self._aruco_dict, parameters=params)
        return corners, ids

    def _build_obj_pts(self, marker_length: float) -> np.ndarray:
        """构建 Marker 3D 角点（物体坐标系）。"""
        half = marker_length / 2
        return np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

    def detect(self, frame: np.ndarray) -> Dict[int, dict]:
        """
        检测帧中的 ArUco Marker 并估计位姿。

        优化策略：降分辨率粗检测 + 全分辨率亚像素精化。
        - 粗检测阶段将图像缩小到 COARSE_MAX_WIDTH，大幅减少计算量
        - 角点坐标映射回全分辨率后，在全分辨率灰度图上做 cornerSubPix
        - PnP 使用全分辨率角点 + 全分辨率内参，精度不损失

        Args:
            frame: BGR 图像 (H, W, 3) 或灰度图像 (H, W)

        Returns:
            dict[marker_id, data_dict]，其中 data_dict 包含：
            - raw_corners: 原始检测角点 (4, 2)（全分辨率坐标）
            - filtered_corners: 滤波 + 亚像素精化角点 (4, 2)
            - rvec_m2c: 旋转向量 (3, 1)，Marker -> Camera
            - tvec_m2c: 平移向量 (3, 1)，Marker -> Camera（毫米）
            - R_m2c: 旋转矩阵 (3, 3)
            - euler_m2c_zyx: ZYX 内旋欧拉角（度） [rx, ry, rz]
            - reproj_err: 重投影误差（像素）
            - method: 方法标签（'IPPE_SQ' | 'ITER' | 'HOLD' 等）
            - marker_length: Marker 物理尺寸（毫米）
        """
        K = self._intrinsics.camera_matrix
        dist = self._intrinsics.dist_coeffs

        # 灰度转换（兼容 mono 灰度输入）
        if frame.ndim == 2:
            gray_full = frame
        else:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 降分辨率粗检测
        h, w = gray_full.shape[:2]
        if w > self.COARSE_MAX_WIDTH:
            scale = self.COARSE_MAX_WIDTH / w
            gray_small = cv2.resize(
                gray_full, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            gray_small = gray_full

        gray_morph = self._preprocess(gray_small)
        corners, ids = self._detect_markers(gray_morph)

        result: Dict[int, dict] = {}
        detected_ids: List[int] = []

        if ids is None or len(ids) == 0:
            self._kf_cache.mark_detected([])
            self._smoother_cache.mark_detected([])
            return result

        ids = ids.flatten()
        valid_mask = np.isin(ids, list(self._valid_ids))
        if not np.any(valid_mask):
            self._kf_cache.mark_detected([])
            self._smoother_cache.mark_detected([])
            return result

        valid_ids_arr = ids[valid_mask]
        valid_corners = [corners[i] for i in np.where(valid_mask)[0]]
        unique_ids, unique_idx = np.unique(valid_ids_arr, return_index=True)
        valid_corners = [valid_corners[i] for i in unique_idx]

        # 全分辨率灰度图（用于亚像素精化）
        gray_blur_full = cv2.GaussianBlur(gray_full, (3, 3), 0)

        for i, marker_id in enumerate(unique_ids):
            marker_id = int(marker_id)
            raw_corners_small = valid_corners[i].reshape(4, 2)

            # 映射角点回全分辨率
            if scale != 1.0:
                raw_corners = raw_corners_small / scale
            else:
                raw_corners = raw_corners_small
            detected_ids.append(marker_id)

            marker_length = self._marker_sizes.get(marker_id, 100.0)
            obj_pts = self._build_obj_pts(marker_length)

            # 角点卡尔曼滤波
            if self._use_kalman:
                filtered = self._kf_cache.process(marker_id, raw_corners)
            else:
                filtered = raw_corners.copy()

            # 亚像素精化（在全分辨率灰度图上）
            corners_subpix = filtered.astype(np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(
                gray_blur_full, corners_subpix,
                winSize=(5, 5), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            refined_corners = corners_subpix.reshape(4, 2).astype(np.float64)

            # PnP 求解（全分辨率内参）
            smoother = self._smoother_cache.get_smoother(marker_id)
            ok, rvec, tvec, reproj_err, method = solve_pnp_best(
                obj_pts, refined_corners, K, dist,
                use_guess=smoother.has_prior,
                rvec_guess=smoother.prior_rvec,
                tvec_guess=smoother.prior_tvec,
            )
            if not ok:
                continue

            # 首帧过滤（reprojection 过大时丢弃）
            if not smoother.has_prior and reproj_err > 20.0:
                logger.debug("ID %d 首帧 reproj 过大 %.2fpx -> 丢弃", marker_id, reproj_err)
                continue

            # 位姿平滑 + 异常检测
            if self._use_smoother:
                smooth_result = smoother.smooth(rvec, tvec, reproj_err, method)
            else:
                R_m2c, _ = cv2.Rodrigues(rvec)
                smooth_result = {
                    'rvec_m2c': rvec, 'tvec_m2c': tvec, 'R_m2c': R_m2c,
                    'reproj_err': reproj_err, 'status': 'OK', 'method': method,
                }

            # 计算 ZYX 欧拉角
            euler_zyx = rotmat_to_euler(smooth_result['R_m2c'], order='ZYX')

            result[marker_id] = {
                'raw_corners': raw_corners,
                'filtered_corners': refined_corners,
                'rvec_m2c': smooth_result['rvec_m2c'],
                'tvec_m2c': smooth_result['tvec_m2c'],
                'R_m2c': smooth_result['R_m2c'],
                'euler_m2c_zyx': euler_zyx,
                'reproj_err': smooth_result['reproj_err'],
                'method': smooth_result['method'],
                'status': smooth_result['status'],
                'marker_length': marker_length,
            }

        self._kf_cache.mark_detected(detected_ids)
        self._smoother_cache.mark_detected(detected_ids)
        return result

    def reset(self) -> None:
        """重置所有状态（相机重启或场景切换时调用）。"""
        self._kf_cache.clear()
        self._smoother_cache.clear()

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics
