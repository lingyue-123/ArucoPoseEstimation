"""
深度增强位姿估计 | Depth-Based Pose Estimation

基于深度图的 ArUco 位姿估计：
1. 2D 角点 + 深度图 → 3D 点（中值滤波查深度）
2. 已知 3D 模型点 vs 观测 3D 点 → SVD 刚体配准 → R, t

适用于双目/RGBD 相机（如 Orbbec Gemini 335LG），替代传统 solvePnP。
优势：平移精度直接由深度传感器决定，不受低分辨率 2D 角点精度瓶颈限制。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def corners_to_3d(
    corners_2d: np.ndarray,
    depth_map: np.ndarray,
    camera_matrix: np.ndarray,
    patch_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 2D 角点通过深度图反投影为 3D 坐标。

    对每个角点取邻域 patch 内的中值深度，减少边缘飞点影响。

    Args:
        corners_2d: 2D 角点坐标 (N, 2)，像素坐标 (u, v)
        depth_map: 深度图 (H, W)，uint16，单位毫米，0 表示无效
        camera_matrix: 3×3 相机内参矩阵
        patch_radius: 深度查询邻域半径（像素），取 (2r+1)×(2r+1) 区域中值

    Returns:
        corners_3d: 3D 坐标 (N, 3)，单位毫米，无效点为 [0,0,0]
        valid_mask: 布尔数组 (N,)，True 表示该点深度有效
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    h, w = depth_map.shape[:2]
    n = corners_2d.shape[0]
    corners_3d = np.zeros((n, 3), dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        u, v = corners_2d[i]
        u_int, v_int = int(round(u)), int(round(v))

        # 取邻域 patch
        u_min = max(0, u_int - patch_radius)
        u_max = min(w, u_int + patch_radius + 1)
        v_min = max(0, v_int - patch_radius)
        v_max = min(h, v_int + patch_radius + 1)

        patch = depth_map[v_min:v_max, u_min:u_max].astype(np.float64)
        valid_depths = patch[patch > 0]

        if len(valid_depths) < 3:
            # 有效深度点太少，标记为无效
            continue

        z_mm = float(np.median(valid_depths))

        # 反投影: (u, v, Z) → (X, Y, Z)
        x_mm = (u - cx) * z_mm / fx
        y_mm = (v - cy) * z_mm / fy
        corners_3d[i] = [x_mm, y_mm, z_mm]
        valid_mask[i] = True

    return corners_3d, valid_mask


def solve_pose_svd(
    obj_pts: np.ndarray,
    obs_pts: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    """
    SVD 刚体配准：从两组对应 3D 点求解旋转和平移。

    求解 R, t 使得 obs ≈ R @ obj + t （最小二乘意义下最优）。

    Args:
        obj_pts: 模型坐标系下的 3D 点 (N, 3)，如 ArUco 角点在码坐标系的位置
        obs_pts: 相机坐标系下观测到的 3D 点 (N, 3)
        valid_mask: 布尔数组 (N,)，True 表示该点有效（None 表示全部有效）

    Returns:
        success: 是否成功
        R: 旋转矩阵 (3, 3)，从模型坐标系到相机坐标系
        t: 平移向量 (3, 1)，单位与输入一致（毫米）
        rmse: 配准残差（毫米）
    """
    if valid_mask is not None:
        obj_pts = obj_pts[valid_mask]
        obs_pts = obs_pts[valid_mask]

    n = obj_pts.shape[0]
    if n < 3:
        return False, np.eye(3), np.zeros((3, 1)), float('inf')

    # 去中心化
    centroid_obj = obj_pts.mean(axis=0)
    centroid_obs = obs_pts.mean(axis=0)
    obj_centered = obj_pts - centroid_obj
    obs_centered = obs_pts - centroid_obs

    # SVD 分解
    H = obj_centered.T @ obs_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # 处理反射情况（确保 det(R) = +1）
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T

    # 平移
    t = (centroid_obs - R @ centroid_obj).reshape(3, 1)

    # 残差 RMSE
    transformed = (R @ obj_pts.T).T + t.T
    residuals = np.linalg.norm(transformed - obs_pts, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return True, R, t, rmse


def estimate_pose_from_depth(
    corners_2d: np.ndarray,
    depth_map: np.ndarray,
    obj_pts: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    patch_radius: int = 3,
    rmse_threshold: float = 3.0,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
           float, str]:
    """
    基于深度图的完整位姿估计流程。

    先尝试 SVD 刚体配准，如果失败（深度无效或 RMSE 过大）则回退到 PnP。

    Args:
        corners_2d: 2D 角点 (4, 2)
        depth_map: 深度图 (H, W)，uint16 毫米
        obj_pts: 模型 3D 点 (4, 3)
        camera_matrix: 3×3 内参
        dist_coeffs: 畸变系数
        patch_radius: 深度查询邻域半径
        rmse_threshold: SVD RMSE 阈值（mm），超过则回退 PnP

    Returns:
        success: 是否成功
        rvec: 旋转向量 (3, 1)
        tvec: 平移向量 (3, 1)
        R: 旋转矩阵 (3, 3)
        quality_metric: 质量指标（SVD 用 RMSE mm，PnP 用重投影误差 px）
        method: 方法标签 ('SVD' | 'PNP_FALLBACK')
    """
    # 尝试 SVD 路径
    corners_3d, valid_mask = corners_to_3d(
        corners_2d, depth_map, camera_matrix, patch_radius
    )

    valid_count = int(valid_mask.sum())
    if valid_count >= 3:
        ok, R, t, rmse = solve_pose_svd(obj_pts, corners_3d, valid_mask)
        if ok and rmse < rmse_threshold:
            rvec, _ = cv2.Rodrigues(R)
            logger.debug("SVD 配准成功: RMSE=%.2fmm, 有效点=%d/4", rmse, valid_count)
            return True, rvec, t, R, rmse, 'SVD'
        else:
            logger.debug("SVD 配准质量不佳: RMSE=%.2fmm > 阈值%.1fmm, 回退 PnP",
                         rmse, rmse_threshold)
    else:
        logger.debug("有效深度点不足: %d/4, 回退 PnP", valid_count)

    # 回退到 PnP
    from robovision.detection.pnp import solve_pnp_best
    ok, rvec, tvec, reproj_err, pnp_method = solve_pnp_best(
        obj_pts, corners_2d, camera_matrix, dist_coeffs
    )
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        return True, rvec, tvec, R, reproj_err, f'PNP_FALLBACK({pnp_method})'
    return False, None, None, None, 0.0, ''
