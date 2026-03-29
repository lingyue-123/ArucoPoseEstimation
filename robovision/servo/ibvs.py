"""
图像视觉伺服核心 | IBVS Helpers

第一版策略：
- 特征：ArUco 四角点
- 控制：x / y / rz 三自由度
- 深度：使用当前 marker 的整体深度近似
"""

import logging
import math
from typing import Dict

import numpy as np

from robovision.geometry.transforms import pose_to_matrix, matrix_to_pose

logger = logging.getLogger(__name__)


def _normalize_points(corners: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (corners[:, 0] - cx) / fx
    y = (corners[:, 1] - cy) / fy
    return np.stack([x, y], axis=1)


def _interaction_matrix_xyrz(norm_pts: np.ndarray, depth_mm: float) -> np.ndarray:
    """
    仅保留 x / y / rz 三列的 interaction matrix。
    控制量顺序：[vx, vy, wz]
    """
    z = max(float(depth_mm), 1e-6)
    rows = []
    for x, y in norm_pts:
        rows.append([-1.0 / z, 0.0, y])
        rows.append([0.0, -1.0 / z, -x])
    return np.asarray(rows, dtype=np.float64)


def compute_ibvs_step(current_corners: np.ndarray,
                      ref_corners: np.ndarray,
                      camera_matrix: np.ndarray,
                      depth_mm: float,
                      gain: float = 0.08) -> Dict[str, object]:
    """
    基于四角点做第一版 IBVS。

    Returns:
        dict:
          - ok
          - cam_vel: [vx_mm, vy_mm, wz_rad]
          - error_norm_px
          - error_vec_px
          - debug
    """
    if current_corners is None or ref_corners is None:
        return {'ok': False, 'reason': '缺少当前角点或参考角点'}

    current_corners = np.asarray(current_corners, dtype=np.float64).reshape(-1, 2)
    ref_corners = np.asarray(ref_corners, dtype=np.float64).reshape(-1, 2)
    if current_corners.shape != (4, 2) or ref_corners.shape != (4, 2):
        return {'ok': False, 'reason': f'角点维度错误: cur={current_corners.shape}, ref={ref_corners.shape}'}

    error_px = (current_corners - ref_corners).reshape(-1, 1)
    error_norm_px = float(np.linalg.norm(error_px))

    cur_norm = _normalize_points(current_corners, camera_matrix)
    L = _interaction_matrix_xyrz(cur_norm, depth_mm)
    try:
        cam_ctrl = -gain * (np.linalg.pinv(L) @ error_px)
    except np.linalg.LinAlgError:
        return {'ok': False, 'reason': 'interaction matrix 伪逆失败'}

    vx_mm = float(cam_ctrl[0, 0])
    vy_mm = float(cam_ctrl[1, 0])
    wz_rad = float(cam_ctrl[2, 0])
    return {
        'ok': True,
        'cam_vel': np.array([vx_mm, vy_mm, wz_rad], dtype=np.float64),
        'error_norm_px': error_norm_px,
        'error_vec_px': error_px.reshape(-1),
        'debug': {
            'depth_mm': float(depth_mm),
            'L_shape': L.shape,
        },
    }


def ibvs_cam_vel_to_step_pose(T_g2b_cur: np.ndarray,
                              T_c2g: np.ndarray,
                              cam_vel: np.ndarray,
                              xy_limit_mm: float = 2.0,
                              rz_limit_deg: float = 0.8) -> Dict[str, object]:
    """
    将相机系下的 [vx, vy, wz] 控制量映射为机器人一步 TCP 位姿。
    第一版只做平面内对齐：x / y / rz。
    """
    vx_mm, vy_mm, wz_rad = [float(v) for v in cam_vel]
    vx_mm = float(np.clip(vx_mm, -xy_limit_mm, xy_limit_mm))
    vy_mm = float(np.clip(vy_mm, -xy_limit_mm, xy_limit_mm))
    rz_deg = float(np.clip(math.degrees(wz_rad), -rz_limit_deg, rz_limit_deg))

    delta_cam = np.eye(4)
    delta_cam[:3, 3] = [vx_mm, vy_mm, 0.0]
    delta_cam[:3, :3] = pose_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, rz_deg])[:3, :3]

    T_g2c = np.linalg.inv(T_c2g)
    delta_tool = T_c2g @ delta_cam @ T_g2c
    T_step = T_g2b_cur @ delta_tool
    step_pose = matrix_to_pose(T_step)

    return {
        'step_pose': step_pose,
        'step_trans_mm': float(np.linalg.norm([vx_mm, vy_mm])),
        'step_rot_deg': abs(rz_deg),
        'debug': {
            'vx_mm': vx_mm,
            'vy_mm': vy_mm,
            'rz_deg': rz_deg,
        },
    }
