"""
视觉伺服核心计算 | Visual Servo Core

步进计算、误差计算、安全检查、运动执行。
"""

import logging

import numpy as np

from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose,
    rotmat_to_quat, quat_to_rotmat, quat_slerp,
    rotation_angle_deg,
)

logger = logging.getLogger(__name__)

# 安全阈值 & gain 参数
MAX_STEP_TRANS_MM = 50.0
MAX_STEP_ROT_DEG = 5.0
GAIN_MIN = 0.05
GAIN_MAX = 1.0
GAIN_STEP = 0.05
DEFAULT_GAIN = 0.1
MOVE_TIMEOUT = 30.0


def aruco_to_matrix(data: dict) -> np.ndarray:
    """ArUco 检测结果 dict → 4x4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = data['R_m2c']
    T[:3, 3] = data['tvec_m2c'].flatten()
    return T


def compute_step_pose(T_g2b_cur, T_g2b_target, gain):
    """
    计算带 gain 的步进目标位姿。

    在工具坐标系下计算增量，对平移缩放、对旋转做 SLERP，
    再合成回基座坐标系。

    Returns:
        T_step: 4x4 步进目标矩阵
        trans_mm: 步进平移量 (mm)
        rot_deg: 步进旋转量 (deg)
    """
    T_delta = np.linalg.inv(T_g2b_cur) @ T_g2b_target

    # 平移缩放
    t_delta = T_delta[:3, 3] * gain

    # 旋转 SLERP
    R_delta = T_delta[:3, :3]
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    q_delta = rotmat_to_quat(R_delta)
    q_gained = quat_slerp(q_identity, q_delta, gain)
    R_gained = quat_to_rotmat(q_gained)

    # 合成步进目标
    T_step_local = np.eye(4)
    T_step_local[:3, :3] = R_gained
    T_step_local[:3, 3] = t_delta
    T_step = T_g2b_cur @ T_step_local

    trans_mm = float(np.linalg.norm(t_delta))
    rot_deg = rotation_angle_deg(np.eye(3), R_gained)

    return T_step, trans_mm, rot_deg


def compute_pose_error(T_g2b_cur, T_g2b_target):
    """计算当前位姿与目标位姿之间的误差 (trans_mm, rot_deg)。"""
    T_delta = np.linalg.inv(T_g2b_cur) @ T_g2b_target
    trans_err = float(np.linalg.norm(T_delta[:3, 3]))
    rot_err = rotation_angle_deg(np.eye(3), T_delta[:3, :3])
    return trans_err, rot_err


def compute_pose_error_in_frame(T_g2b_cur, T_g2b_ref):
    """
    计算当前 TCP 相对于参考 TCP 的误差，分解到参考坐标系的 x/y/z 轴。

    Returns:
        trans_xyz: ndarray (3,) — 参考坐标系下 [dx, dy, dz] (mm)
        trans_norm: float — 平移误差范数 (mm)
        rot_err: float — 旋转误差 (deg)
    """
    R_ref = T_g2b_ref[:3, :3]
    t_diff_base = T_g2b_cur[:3, 3] - T_g2b_ref[:3, 3]
    trans_xyz = R_ref.T @ t_diff_base

    trans_norm = float(np.linalg.norm(trans_xyz))

    R_delta = R_ref.T @ T_g2b_cur[:3, :3]
    rot_err = rotation_angle_deg(np.eye(3), R_delta)

    return trans_xyz, trans_norm, rot_err


def check_step_safety(step_trans, step_rot):
    """
    检查步进是否超出安全阈值。

    Returns:
        (safe: bool, reason: str)
    """
    if step_trans > MAX_STEP_TRANS_MM:
        return False, (f"步进平移 {step_trans:.1f} mm 超过阈值 {MAX_STEP_TRANS_MM:.1f} mm，"
                       "降低 gain 或手动移近后重试")
    if step_rot > MAX_STEP_ROT_DEG:
        return False, (f"步进旋转 {step_rot:.2f} deg 超过阈值 {MAX_STEP_ROT_DEG:.1f} deg，"
                       "降低 gain 或手动调整姿态后重试")
    return True, ""


def execute_move(robot, step_pose_6dof, timeout=MOVE_TIMEOUT):
    """
    封装 CartesianPose 创建 + move_and_wait。

    Args:
        robot: RobotBase 实例
        step_pose_6dof: [x, y, z, rx, ry, rz]
        timeout: 运动超时秒数

    Returns:
        bool: 运动是否成功
    """
    from third_party.robot_driver.robot_driver_interface import CartesianPose
    cart = CartesianPose(
        x=step_pose_6dof[0], y=step_pose_6dof[1], z=step_pose_6dof[2],
        rx=step_pose_6dof[3], ry=step_pose_6dof[4], rz=step_pose_6dof[5],
    )
    logger.info("执行运动...")
    ok = robot.move_and_wait(cart, timeout=timeout)
    if ok:
        logger.info("运动完成")
    else:
        logger.warning("运动超时或失败")
    return ok


def adaptive_gain(trans_err_mm, base_gain=0.3):
    """自适应 gain：远距离大步、近距离小步。"""
    if trans_err_mm > 30:
        return min(base_gain * 1.5, GAIN_MAX)
    elif trans_err_mm < 5:
        return base_gain * 0.5
    return base_gain
