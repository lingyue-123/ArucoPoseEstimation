"""
统一几何变换工具 | Unified Geometry Transforms

整合来自以下三处的重复实现：
- Handeyecalib/hand_eye_calib.py: euler_xyz_to_rotation()
- aruco_pose_from_hkws_cam_multi_aruco.py: rotmat_to_quat(), quat_slerp()
- verification_hkws_cam_0225.py: pose_to_matrix(), matrix_to_pose()

坐标约定（euler_order 参数）：
- 'ZYX': scipy ZYX intrinsic → R = Rz @ Ry @ Rx（默认，JAKA SDK / 手眼标定 / ArUco 均用此约定）
- 'XYZ': scipy XYZ intrinsic → R = Rx @ Ry @ Rz
"""

import math
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# 欧拉角 <-> 旋转矩阵
# ============================================================

def euler_to_rotmat(rx: float, ry: float, rz: float, order: str = 'ZYX') -> np.ndarray:
    """
    欧拉角转旋转矩阵。

    Args:
        rx, ry, rz: 绕 X/Y/Z 轴的旋转角度（度）
        order: 'XYZ' 或 'ZYX'，指定内旋顺序

    Returns:
        R: 3x3 旋转矩阵
    """
    if order == 'XYZ':
        return Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
    elif order == 'ZYX':
        return Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
    else:
        raise ValueError(f"不支持的欧拉角顺序: {order}，支持 'XYZ' 或 'ZYX'")


def rotmat_to_euler(R: np.ndarray, order: str = 'ZYX') -> np.ndarray:
    """
    旋转矩阵转欧拉角。

    Args:
        R: 3x3 旋转矩阵
        order: 'XYZ' 或 'ZYX'

    Returns:
        np.ndarray: [rx, ry, rz]（度）
    """
    if order == 'XYZ':
        rx, ry, rz = Rotation.from_matrix(R).as_euler('XYZ', degrees=True)
        return np.array([rx, ry, rz])
    elif order == 'ZYX':
        rz, ry, rx = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
        return np.array([rx, ry, rz])
    else:
        raise ValueError(f"不支持的欧拉角顺序: {order}")


# ============================================================
# 位姿向量 <-> 齐次变换矩阵
# ============================================================

def pose_to_matrix(pose: Union[list, np.ndarray], euler_order: str = 'ZYX') -> np.ndarray:
    """
    位姿向量转 4x4 齐次变换矩阵。

    Args:
        pose: [x, y, z, rx, ry, rz]（平移单位任意，角度为度）
        euler_order: 'XYZ' 或 'ZYX'

    Returns:
        T: 4x4 齐次变换矩阵
    """
    x, y, z, rx, ry, rz = pose
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = euler_to_rotmat(rx, ry, rz, order=euler_order)
    return T


def matrix_to_pose(T: np.ndarray, euler_order: str = 'ZYX') -> list:
    """
    4x4 齐次变换矩阵转位姿向量。

    Args:
        T: 4x4 齐次变换矩阵
        euler_order: 'XYZ' 或 'ZYX'

    Returns:
        [x, y, z, rx, ry, rz]
    """
    x, y, z = T[:3, 3]
    rx, ry, rz = rotmat_to_euler(T[:3, :3], order=euler_order)
    return [x, y, z, float(rx), float(ry), float(rz)]


# ============================================================
# 四元数工具
# ============================================================

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数 [w, x, y, z]。

    使用 Shepperd 方法，数值稳定。
    """
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    四元数 [w, x, y, z] 转旋转矩阵。
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    球面线性插值（SLERP）。

    Args:
        q0, q1: 四元数 [w, x, y, z]
        t: 插值参数 [0, 1]，0 = q0，1 = q1

    Returns:
        插值后的四元数（归一化）
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = (s0 * q0) + (s1 * q1)
    return q / np.linalg.norm(q)


# ============================================================
# 旋转角度计算
# ============================================================

def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    计算两个旋转矩阵之间的角度差（度）。
    """
    R = Ra.T @ Rb
    cosv = (np.trace(R) - 1.0) / 2.0
    cosv = min(1.0, max(-1.0, float(cosv)))
    return math.degrees(math.acos(cosv))


# ============================================================
# 坐标链计算（用于验证和位姿估计）
# ============================================================

def compute_new_tool_pose(
    base_from_tool_old: np.ndarray,
    cam_from_target_old: np.ndarray,
    tool_from_cam: np.ndarray,
    cam_from_target_new: np.ndarray
) -> np.ndarray:
    """
    基于手眼标定结果计算新位置的机器人位姿。

    原理：标定板在基座坐标系下的位置固定不变。
    base_from_target = base_from_tool @ tool_from_cam @ cam_from_target

    Args:
        base_from_tool_old: 4x4，旧位置机器人位姿矩阵
        cam_from_target_old: 4x4，旧位置相机看到的标定板位姿
        tool_from_cam: 4x4，手眼标定结果（相机到末端）
        cam_from_target_new: 4x4，新位置相机看到的标定板位姿

    Returns:
        4x4 新位置机器人位姿矩阵
    """
    base_from_target = base_from_tool_old @ tool_from_cam @ cam_from_target_old
    target_from_cam_new = np.linalg.inv(cam_from_target_new)
    cam_from_tool = np.linalg.inv(tool_from_cam)
    return base_from_target @ target_from_cam_new @ cam_from_tool


def compute_tool_delta(
    base_from_tool_old: np.ndarray,
    base_from_tool_new: np.ndarray,
    euler_order: str = 'ZYX'
) -> dict:
    """
    计算机器人位姿变化量（在 base 坐标系下）。

    Returns:
        dict 含 'translation_m'（米）和 'euler_deg'（度）
    """
    delta_T = np.linalg.inv(base_from_tool_old) @ base_from_tool_new
    dx, dy, dz = delta_T[:3, 3]
    euler = rotmat_to_euler(delta_T[:3, :3], order=euler_order)
    return {
        'translation_m': np.array([dx, dy, dz]),
        'euler_deg': euler,
    }


def offset_pose_along_tool_axis(pose: Union[list, np.ndarray], distance_mm: float,
                                axis: str = 'z', euler_order: str = 'ZYX') -> list:
    """沿工具局部坐标轴平移，姿态保持不变。"""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError(f"Unsupported axis: {axis}")

    T = pose_to_matrix(pose, euler_order=euler_order)
    axis_dir_in_base = T[:3, :3][:, axis_map[axis]]
    T_new = T.copy()
    T_new[:3, 3] = T[:3, 3] + axis_dir_in_base * float(distance_mm)
    return matrix_to_pose(T_new, euler_order=euler_order)
