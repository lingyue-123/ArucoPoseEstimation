"""
PnP 求解工具 | PnP Solver Utilities

整合自所有 ArUco 检测脚本中的 solve_pnp_best() / calc_reproj_error()。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def calc_reproj_error(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """
    计算重投影误差（均值，像素）。

    Args:
        obj_pts: 3D 物体点，shape (N, 3)
        img_pts: 2D 图像点，shape (N, 2)
        rvec: 旋转向量，shape (3, 1)
        tvec: 平移向量，shape (3, 1)
        camera_matrix: 3x3 内参矩阵
        dist_coeffs: 畸变系数

    Returns:
        均值重投影误差（像素）
    """
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.mean(err))


def solve_pnp_best(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    use_guess: bool = False,
    rvec_guess: Optional[np.ndarray] = None,
    tvec_guess: Optional[np.ndarray] = None,
    methods: Optional[list] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float, str]:
    """
    多方法 PnP 求解，选取重投影误差最小的结果。

    Args:
        obj_pts: 3D 物体点，shape (N, 3)
        img_pts: 2D 图像点，shape (N, 2)
        camera_matrix: 3x3 内参矩阵
        dist_coeffs: 畸变系数
        use_guess: 是否使用初始猜测
        rvec_guess: 初始旋转向量猜测
        tvec_guess: 初始平移向量猜测
        methods: PnP 求解方法列表（None 则使用默认列表）

    Returns:
        (success, rvec, tvec, reproj_err, method_name)
        失败时 success=False，其他返回 None 或 0
    """
    if methods is None:
        methods = [
            (cv2.SOLVEPNP_IPPE_SQUARE, "IPPE_SQ"),
            (cv2.SOLVEPNP_ITERATIVE, "ITER"),
        ]

    candidates = []
    for flag, name in methods:
        try:
            if use_guess and rvec_guess is not None and tvec_guess is not None:
                ok, r, t = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs,
                    rvec=rvec_guess, tvec=tvec_guess,
                    useExtrinsicGuess=True, flags=flag,
                )
            else:
                ok, r, t = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=flag)
            if ok:
                err = calc_reproj_error(obj_pts, img_pts, r, t, camera_matrix, dist_coeffs)
                candidates.append((err, r, t, name))
        except Exception:
            pass

    if not candidates:
        return False, None, None, 0.0, ""

    err, rvec, tvec, method = min(candidates, key=lambda x: x[0])
    return True, rvec, tvec, err, method
