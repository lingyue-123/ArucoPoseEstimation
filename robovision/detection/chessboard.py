"""
棋盘格检测器 | Chessboard Detector

整合来自以下脚本的实现：
- Handeyecalib/chessboard_est_hkws_cam_0225.py
- Handeyecalib/chessboard_est_usb_cam_0206.py
- Handeyecalib/chessboard_est_web_cam_0212.py
- Handeyecalib/chessboard_est.py

相机无关：通过 CameraIntrinsics 参数支持任意相机。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from robovision.cameras.base import CameraIntrinsics
from robovision.detection.pnp import calc_reproj_error
from robovision.geometry.transforms import rotmat_to_euler

logger = logging.getLogger(__name__)


class ChessboardDetector:
    """
    棋盘格角点检测与 PnP 位姿估计。

    Args:
        intrinsics: 相机内参
        pattern_size: (cols, rows) 内角点数量，如 (11, 8)
        square_size: 每格物理尺寸（毫米），如 25.0
        subpix_win: 亚像素精化窗口大小
        subpix_max_iter: 亚像素精化最大迭代次数
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        pattern_size: Tuple[int, int] = (11, 8),
        square_size: float = 25.0,
        subpix_win: int = 11,
        subpix_max_iter: int = 30,
        detect_scale: float = 1.0,
    ):
        self._intrinsics = intrinsics
        self._pattern_size = pattern_size
        self._square_size = square_size
        self._subpix_win = subpix_win
        self._subpix_max_iter = subpix_max_iter
        self._detect_scale = detect_scale  # <1.0 时先缩小再粗检测，subpix 回全分辨率精化

        # 预计算物体坐标点
        cols, rows = pattern_size
        self._obj_pts = np.zeros((cols * rows, 3), np.float32)
        self._obj_pts[:, :2] = (
            np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        )

    @classmethod
    def from_config(cls, intrinsics: CameraIntrinsics, chessboard_cfg,
                    detect_scale: float = 1.0) -> 'ChessboardDetector':
        """从 ChessboardConfig 构造。"""
        return cls(
            intrinsics=intrinsics,
            pattern_size=chessboard_cfg.pattern_size,
            square_size=chessboard_cfg.square_size,
            detect_scale=detect_scale,
        )

    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """
        检测棋盘格并估计位姿（board -> camera）。

        Args:
            frame: BGR 图像

        Returns:
            成功时返回 dict：
            - corners: 精化后角点，shape (N, 1, 2)
            - rvec: 旋转向量 (3, 1)
            - tvec: 平移向量 (3, 1)，单位：毫米
            - R: 旋转矩阵 (3, 3)
            - euler_zyx: ZYX 内旋欧拉角（度），R = Rz@Ry@Rx，与 JAKA SDK 约定一致
            - reproj_err: 重投影误差（像素）
            - pose_mm: [tx, ty, tz, rx, ry, rz] 位姿向量（毫米，度）
            失败时返回 None
        """
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 粗检测：缩小分辨率加速 findChessboardCorners（速度提升约 scale^-2 倍）
        if self._detect_scale < 1.0:
            h, w = gray.shape[:2]
            small = cv2.resize(gray, (int(w * self._detect_scale), int(h * self._detect_scale)),
                               interpolation=cv2.INTER_AREA)
            ok, corners_small = cv2.findChessboardCorners(small, self._pattern_size)
            if not ok:
                return None
            corners = corners_small / self._detect_scale  # 坐标映射回全分辨率
        else:
            ok, corners = cv2.findChessboardCorners(gray, self._pattern_size)
            if not ok:
                return None

        # 亚像素精化在全分辨率 gray 上进行，保证精度
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self._subpix_max_iter,
            0.001,
        )
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (self._subpix_win, self._subpix_win), (-1, -1), criteria
        )

        K = self._intrinsics.camera_matrix
        dist = self._intrinsics.dist_coeffs
        ret, rvec, tvec = cv2.solvePnP(self._obj_pts, corners_subpix, K, dist)
        if not ret:
            return None

        R, _ = cv2.Rodrigues(rvec)
        # ZYX 内旋欧拉角（R = Rz@Ry@Rx），与 JAKA SDK 和旧脚本约定一致
        euler_zyx = rotmat_to_euler(R, order='ZYX')
        reproj_err = calc_reproj_error(self._obj_pts, corners_subpix.reshape(-1, 2), rvec, tvec, K, dist)
        t = tvec.flatten()
        pose_mm = [float(t[0]), float(t[1]), float(t[2]),
                  float(euler_zyx[0]), float(euler_zyx[1]), float(euler_zyx[2])]

        return {
            'corners': corners_subpix,
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'euler_zyx': euler_zyx,
            'reproj_err': reproj_err,
            'pose_mm': pose_mm,
        }

    def draw_result(self, frame: np.ndarray, result: Optional[dict]) -> np.ndarray:
        """在图像上绘制棋盘格角点和坐标轴。"""
        vis = frame.copy()
        if result is None:
            return vis
        cv2.drawChessboardCorners(vis, self._pattern_size, result['corners'], True)
        axis_len = self._square_size * 3
        cv2.drawFrameAxes(
            vis,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            result['rvec'],
            result['tvec'],
            axis_len,
        )
        return vis

    @property
    def obj_pts(self) -> np.ndarray:
        """棋盘格 3D 角点坐标（棋盘坐标系，单位毫米）。"""
        return self._obj_pts

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return self._pattern_size

    @property
    def square_size(self) -> float:
        return self._square_size

    @property
    def obj_pts(self) -> np.ndarray:
        return self._obj_pts.copy()
