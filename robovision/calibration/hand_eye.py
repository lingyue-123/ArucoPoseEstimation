"""
手眼标定算法 | Hand-Eye Calibration

从 Handeyecalib/hand_eye_calib.py 提升的核心算法，
转换为无副作用的纯函数接口。
支持文件读写、误差计算、结果格式化。

v0.2: 多方法对比、异常点剔除、非线性优化、留一法分析
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robovision.geometry.transforms import euler_to_rotmat, rotmat_to_euler

logger = logging.getLogger(__name__)

# OpenCV 手眼标定方法映射
HAND_EYE_METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeResult:
    """手眼标定结果。"""
    R_cam2gripper: np.ndarray  # (3, 3) 旋转矩阵
    t_cam2gripper: np.ndarray  # (3, 1) 平移向量（毫米）
    mean_pos: np.ndarray       # 标定板在基座下的平均位置（毫米）
    std_pos: np.ndarray        # 标准差（毫米）
    avg_error: float           # 平均误差（毫米）
    method_name: str = "TSAI"
    per_pose_errors: Optional[np.ndarray] = None
    rejected_indices: Optional[List[int]] = None
    used_indices: Optional[List[int]] = None
    all_method_results: Optional[list] = None
    refined: bool = False

    @property
    def euler_zyx_deg(self) -> np.ndarray:
        """旋转矩阵转 ZYX 内旋欧拉角（度）。"""
        return rotmat_to_euler(self.R_cam2gripper, order='ZYX')

    @property
    def translation_mm(self) -> np.ndarray:
        return self.t_cam2gripper.flatten()

    def to_matrix_4x4(self) -> np.ndarray:
        """转换为 4x4 齐次变换矩阵（cam2gripper）。"""
        T = np.eye(4)
        T[:3, :3] = self.R_cam2gripper
        T[:3, 3:4] = self.t_cam2gripper
        return T

    def quality_label(self) -> str:
        if self.avg_error < 2:
            return "优秀"
        elif self.avg_error < 5:
            return "良好"
        elif self.avg_error < 10:
            return "一般"
        else:
            return "较差"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _prepare_input_lists(
    cam_data: list, robot_data: list
) -> Tuple[list, list, list, list]:
    """
    将原始位姿数据转换为 OpenCV calibrateHandEye 所需的旋转矩阵/平移向量列表。

    Args:
        cam_data: board2cam 数据，每项 [tx(mm), ty(mm), tz(mm), rx, ry, rz(度)]
        robot_data: end2base 数据，同上

    Returns:
        (R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
    """
    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    for cam, robot in zip(cam_data, robot_data):
        t_cam = np.array([[cam[0]], [cam[1]], [cam[2]]])
        R_cam = euler_to_rotmat(cam[3], cam[4], cam[5], order='ZYX')
        R_t2c_list.append(R_cam)
        t_t2c_list.append(t_cam)

        t_robot = np.array([[robot[0]], [robot[1]], [robot[2]]])
        R_robot = euler_to_rotmat(robot[3], robot[4], robot[5], order='ZYX')
        R_g2b_list.append(R_robot)
        t_g2b_list.append(t_robot)

    return R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list


def _select_indices(lists_tuple, indices):
    """从多个列表中按索引选取子集。"""
    return tuple([lst[i] for i in indices] for lst in lists_tuple)


# ---------------------------------------------------------------------------
# 核心标定函数
# ---------------------------------------------------------------------------

def hand_eye_calibration(
    cam_data: list,
    robot_data: list,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> tuple:
    """
    手眼标定主函数（Eye-in-Hand 配置）。

    Args:
        cam_data: 相机数据列表，每项 [tx(mm), ty(mm), tz(mm), rx(°), ry(°), rz(°)]
                  含义：board2cam，平移单位毫米，ZYX 内旋欧拉角（度），即 R = Rz(rz)@Ry(ry)@Rx(rx)
        robot_data: 机器人数据列表，每项 [tx(mm), ty(mm), tz(mm), rx(°), ry(°), rz(°)]
                    含义：end2base，平移单位毫米（由 JAKARobot.get_tcp_pose() 写入），ZYX 内旋欧拉角（度，JAKA SDK 格式，R=Rz@Ry@Rx）
        method: OpenCV 手眼标定方法，默认 TSAI

    Returns:
        (R_cam2gripper, t_cam2gripper,
         R_gripper2base_list, t_gripper2base_list,
         R_target2cam_list, t_target2cam_list)
    """
    R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list = _prepare_input_lists(
        cam_data, robot_data
    )
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b_list, t_g2b_list,
        R_t2c_list, t_t2c_list,
        method=method,
    )
    return R_c2g, t_c2g, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list


def compute_reprojection_error(
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    R_gripper2base_list: list,
    t_gripper2base_list: list,
    R_target2cam_list: list,
    t_target2cam_list: list,
) -> tuple:
    """
    计算手眼标定重投影误差。

    原理：标定板在基座坐标系下的位置应固定不变。
    T_base_target = T_gripper2base @ T_cam2gripper @ T_target2cam

    Returns:
        (mean_pos, std_pos, avg_error, target_positions)
        单位：毫米
    """
    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_cam2gripper
    T_c2g[:3, 3:4] = t_cam2gripper

    target_positions = []
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
    ):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b
        T_g2b[:3, 3:4] = t_g2b

        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_t2c
        T_t2c[:3, 3:4] = t_t2c

        T_base_target = T_g2b @ T_c2g @ T_t2c
        target_positions.append(T_base_target[:3, 3])

    target_positions = np.array(target_positions)
    mean_pos = np.mean(target_positions, axis=0)
    std_pos = np.std(target_positions, axis=0)
    avg_error = float(np.mean(std_pos))
    return mean_pos, std_pos, avg_error, target_positions


# ---------------------------------------------------------------------------
# 逐点误差
# ---------------------------------------------------------------------------

def compute_per_pose_errors(
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    R_gripper2base_list: list,
    t_gripper2base_list: list,
    R_target2cam_list: list,
    t_target2cam_list: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个位姿的误差（与均值的欧氏距离）。

    Returns:
        (per_pose_errors, mean_pos, target_positions)  单位：毫米
    """
    mean_pos, _, _, target_positions = compute_reprojection_error(
        R_cam2gripper, t_cam2gripper,
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
    )
    per_pose_errors = np.linalg.norm(target_positions - mean_pos, axis=1)
    return per_pose_errors, mean_pos, target_positions


# ---------------------------------------------------------------------------
# 多方法对比
# ---------------------------------------------------------------------------

def calibrate_all_methods(
    R_g2b_list: list, t_g2b_list: list,
    R_t2c_list: list, t_t2c_list: list,
) -> List[dict]:
    """
    尝试所有 OpenCV 手眼标定方法，返回按误差排序的结果列表。

    Returns:
        [{'name': str, 'R': ndarray, 't': ndarray, 'avg_error': float,
          'mean_pos': ndarray, 'std_pos': ndarray, 'quality': str}, ...]
    """
    results = []
    for name, method_id in HAND_EYE_METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_g2b_list, t_g2b_list,
                R_t2c_list, t_t2c_list,
                method=method_id,
            )
            mean_pos, std_pos, avg_error, _ = compute_reprojection_error(
                R_c2g, t_c2g, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
            )
            # 检查结果是否合理（旋转矩阵行列式应接近 1）
            det = np.linalg.det(R_c2g)
            if abs(det - 1.0) > 0.1:
                logger.warning("方法 %s 旋转矩阵行列式异常: %.4f，跳过", name, det)
                continue

            result_entry = {
                'name': name,
                'R': R_c2g,
                't': t_c2g,
                'avg_error': avg_error,
                'mean_pos': mean_pos,
                'std_pos': std_pos,
                'quality': _quality_label(avg_error),
            }
            results.append(result_entry)
        except Exception as e:
            logger.warning("方法 %s 失败: %s", name, e)

    results.sort(key=lambda x: x['avg_error'])
    return results


def _quality_label(avg_error: float) -> str:
    if avg_error < 2:
        return "优秀"
    elif avg_error < 5:
        return "良好"
    elif avg_error < 10:
        return "一般"
    else:
        return "较差"


# ---------------------------------------------------------------------------
# 迭代异常点剔除
# ---------------------------------------------------------------------------

def calibrate_with_outlier_rejection(
    R_g2b_list: list, t_g2b_list: list,
    R_t2c_list: list, t_t2c_list: list,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
    max_error_mm: float = 5.0,
    min_poses: int = 8,
    max_iterations: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    迭代剔除异常点后标定。

    算法：
    1. 用全部数据标定，计算每点误差
    2. 若最差点误差 > max_error_mm 且剩余点数 > min_poses：剔除
    3. 重复直到收敛

    Returns:
        (R_c2g, t_c2g, used_indices, rejected_indices)
    """
    n = len(R_g2b_list)
    used = list(range(n))
    rejected = []

    for iteration in range(max_iterations):
        if len(used) < min_poses:
            break

        R_g2b_sub = [R_g2b_list[i] for i in used]
        t_g2b_sub = [t_g2b_list[i] for i in used]
        R_t2c_sub = [R_t2c_list[i] for i in used]
        t_t2c_sub = [t_t2c_list[i] for i in used]

        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub, method=method,
        )

        per_errors, _, _ = compute_per_pose_errors(
            R_c2g, t_c2g, R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub,
        )

        worst_local_idx = int(np.argmax(per_errors))
        worst_error = per_errors[worst_local_idx]

        if worst_error <= max_error_mm:
            logger.info("迭代 %d: 最大误差 %.2f mm <= 阈值 %.2f mm，收敛",
                        iteration + 1, worst_error, max_error_mm)
            break

        worst_global_idx = used[worst_local_idx]
        rejected.append(worst_global_idx)
        used.remove(worst_global_idx)
        logger.info("迭代 %d: 剔除位姿 #%d（误差 %.2f mm），剩余 %d 组",
                    iteration + 1, worst_global_idx, worst_error, len(used))

    # 最终标定
    R_g2b_sub = [R_g2b_list[i] for i in used]
    t_g2b_sub = [t_g2b_list[i] for i in used]
    R_t2c_sub = [R_t2c_list[i] for i in used]
    t_t2c_sub = [t_t2c_list[i] for i in used]

    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub, method=method,
    )

    return R_c2g, t_c2g, used, rejected


# ---------------------------------------------------------------------------
# 留一法分析
# ---------------------------------------------------------------------------

def leave_one_out_analysis(
    R_g2b_list: list, t_g2b_list: list,
    R_t2c_list: list, t_t2c_list: list,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> List[dict]:
    """
    留一法分析：依次去掉每个位姿，评估对标定结果的影响。

    Returns:
        [{'index': int, 'avg_error_without': float, 'delta_error': float}, ...]
        delta_error < 0 表示去掉后误差下降（该点可能是坏点）
    """
    n = len(R_g2b_list)

    # 基线：全部数据
    R_c2g_all, t_c2g_all = cv2.calibrateHandEye(
        R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, method=method,
    )
    _, _, baseline_error, _ = compute_reprojection_error(
        R_c2g_all, t_c2g_all, R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
    )

    results = []
    for i in range(n):
        indices = [j for j in range(n) if j != i]
        R_g2b_sub = [R_g2b_list[j] for j in indices]
        t_g2b_sub = [t_g2b_list[j] for j in indices]
        R_t2c_sub = [R_t2c_list[j] for j in indices]
        t_t2c_sub = [t_t2c_list[j] for j in indices]

        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub, method=method,
            )
            _, _, err_without, _ = compute_reprojection_error(
                R_c2g, t_c2g, R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub,
            )
            delta = err_without - baseline_error
        except Exception:
            err_without = float('inf')
            delta = float('inf')

        results.append({
            'index': i,
            'avg_error_without': err_without,
            'delta_error': delta,
        })

    results.sort(key=lambda x: x['delta_error'])
    return results


# ---------------------------------------------------------------------------
# 非线性精炼
# ---------------------------------------------------------------------------

def refine_hand_eye(
    R_c2g_init: np.ndarray,
    t_c2g_init: np.ndarray,
    R_g2b_list: list, t_g2b_list: list,
    R_t2c_list: list, t_t2c_list: list,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Levenberg-Marquardt 非线性精炼手眼标定结果。

    最小化所有位姿下标定板在基座中位置的散布。
    参数化：Rodrigues 3 + 平移 3。

    Returns:
        (R_refined, t_refined, error_before, error_after)
        若优化后误差反而变大，返回原始结果。
    """
    # 初始参数：Rodrigues + 平移
    rvec_init = Rotation.from_matrix(R_c2g_init).as_rotvec()
    t_init = t_c2g_init.flatten()
    x0 = np.concatenate([rvec_init, t_init])

    n = len(R_g2b_list)

    def residuals(x):
        rvec = x[:3]
        tvec = x[3:6]
        R_c2g = Rotation.from_rotvec(rvec).as_matrix()
        t_c2g = tvec.reshape(3, 1)

        T_c2g = np.eye(4)
        T_c2g[:3, :3] = R_c2g
        T_c2g[:3, 3:4] = t_c2g

        positions = np.zeros((n, 3))
        for i in range(n):
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_g2b_list[i]
            T_g2b[:3, 3:4] = t_g2b_list[i]
            T_t2c = np.eye(4)
            T_t2c[:3, :3] = R_t2c_list[i]
            T_t2c[:3, 3:4] = t_t2c_list[i]
            T_bt = T_g2b @ T_c2g @ T_t2c
            positions[i] = T_bt[:3, 3]

        mean_pos = positions.mean(axis=0)
        # 残差：每个位姿的位置偏差（3*n 维）
        return (positions - mean_pos).flatten()

    _, _, error_before, _ = compute_reprojection_error(
        R_c2g_init, t_c2g_init,
        R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
    )

    result = least_squares(residuals, x0, method='lm')

    R_refined = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t_refined = result.x[3:6].reshape(3, 1)

    # SVD 正交化
    U, _, Vt = np.linalg.svd(R_refined)
    R_refined = U @ Vt
    if np.linalg.det(R_refined) < 0:
        U[:, -1] *= -1
        R_refined = U @ Vt

    _, _, error_after, _ = compute_reprojection_error(
        R_refined, t_refined,
        R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
    )

    if error_after >= error_before:
        logger.info("非线性优化未改善误差 (%.2f -> %.2f mm)，保留原始结果",
                    error_before, error_after)
        return R_c2g_init, t_c2g_init, error_before, error_before

    logger.info("非线性优化: %.2f -> %.2f mm", error_before, error_after)
    return R_refined, t_refined, error_before, error_after


# ---------------------------------------------------------------------------
# 编排函数
# ---------------------------------------------------------------------------

def run_calibration(
    cam_data: list,
    robot_data: list,
    compare_methods: bool = False,
    outlier_rejection: bool = False,
    max_error_mm: float = 5.0,
    min_poses: int = 8,
    refine: bool = False,
    method: Optional[str] = None,
) -> HandEyeResult:
    """
    完整手眼标定流程。

    所有新参数默认值使行为与 v0.1 完全一致。

    Args:
        cam_data: board2cam 数据
        robot_data: end2base 数据
        compare_methods: 尝试所有方法并选最优
        outlier_rejection: 迭代剔除异常点
        max_error_mm: 剔除阈值
        min_poses: 最少保留组数
        refine: 非线性优化
        method: 指定方法名（TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS）
    """
    R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list = _prepare_input_lists(
        cam_data, robot_data
    )

    # 确定使用的 OpenCV 方法
    if method:
        method_upper = method.upper()
        if method_upper not in HAND_EYE_METHODS:
            raise ValueError(f"未知方法: {method}，可选: {list(HAND_EYE_METHODS.keys())}")
        cv_method = HAND_EYE_METHODS[method_upper]
        method_name = method_upper
    else:
        cv_method = cv2.CALIB_HAND_EYE_TSAI
        method_name = "TSAI"

    # 多方法对比
    all_method_results = None
    if compare_methods:
        all_method_results = calibrate_all_methods(
            R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
        )
        if all_method_results and not method:
            best = all_method_results[0]
            method_name = best['name']
            cv_method = HAND_EYE_METHODS[method_name]
            logger.info("多方法对比: 最优方法 = %s (误差 %.2f mm)",
                        method_name, best['avg_error'])

    # 异常点剔除
    used_indices = None
    rejected_indices = None
    if outlier_rejection:
        R_c2g, t_c2g, used_indices, rejected_indices = calibrate_with_outlier_rejection(
            R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
            method=cv_method, max_error_mm=max_error_mm, min_poses=min_poses,
        )
        # 用剔除后的子集计算误差
        R_g2b_sub = [R_g2b_list[i] for i in used_indices]
        t_g2b_sub = [t_g2b_list[i] for i in used_indices]
        R_t2c_sub = [R_t2c_list[i] for i in used_indices]
        t_t2c_sub = [t_t2c_list[i] for i in used_indices]
    else:
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, method=cv_method,
        )
        R_g2b_sub = R_g2b_list
        t_g2b_sub = t_g2b_list
        R_t2c_sub = R_t2c_list
        t_t2c_sub = t_t2c_list

    # 非线性精炼
    refined = False
    if refine:
        R_c2g, t_c2g, err_before, err_after = refine_hand_eye(
            R_c2g, t_c2g, R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub,
        )
        refined = err_after < err_before

    # 计算最终误差（用实际使用的数据子集）
    mean_pos, std_pos, avg_error, target_positions = compute_reprojection_error(
        R_c2g, t_c2g, R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub,
    )

    # 逐点误差
    per_errors, _, _ = compute_per_pose_errors(
        R_c2g, t_c2g, R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub,
    )

    return HandEyeResult(
        R_cam2gripper=R_c2g,
        t_cam2gripper=t_c2g,
        mean_pos=mean_pos,
        std_pos=std_pos,
        avg_error=avg_error,
        method_name=method_name,
        per_pose_errors=per_errors,
        rejected_indices=rejected_indices if rejected_indices else None,
        used_indices=used_indices,
        all_method_results=all_method_results,
        refined=refined,
    )


# ---------------------------------------------------------------------------
# 文件 I/O
# ---------------------------------------------------------------------------

def load_hand_eye_result(path: str) -> np.ndarray:
    """
    从 hand_eye_result.txt 加载 cam2gripper 变换矩阵。

    文件格式（来自 Handeyecalib/hand_eye_result.txt）。

    Returns:
        4x4 numpy 矩阵
    """
    T = np.loadtxt(path)
    if T.shape == (3, 4):
        mat = np.eye(4)
        mat[:3, :] = T
        return mat
    elif T.shape == (4, 4):
        return T
    else:
        raise ValueError(f"不支持的矩阵格式: shape={T.shape}")


def save_hand_eye_result(path: str, result: HandEyeResult) -> None:
    """
    将手眼标定结果保存到文件（4x4 矩阵格式）。
    """
    T = result.to_matrix_4x4()
    np.savetxt(path, T, fmt='%.8f')
    logger.info("手眼标定结果已保存到 %s", path)


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------

def print_results(result: HandEyeResult, target_positions: Optional[np.ndarray] = None) -> None:
    """输出手眼标定结果到日志。"""
    t = result.translation_mm
    euler = result.euler_zyx_deg
    p = result.mean_pos
    s = result.std_pos
    lines = [
        "=" * 60,
        "           手眼标定结果 (cam2gripper)",
        "=" * 60,
        "",
        f"  使用方法: {result.method_name}"
        + (" + 非线性优化" if result.refined else ""),
        "",
        "【平移向量】",
        f"  X: {t[0]:>10.2f} mm",
        f"  Y: {t[1]:>10.2f} mm",
        f"  Z: {t[2]:>10.2f} mm",
        f"  总距离: {np.linalg.norm(t):>7.2f} mm",
        "",
        "【欧拉角 (ZYX内旋)】",
        f"  Rx: {euler[0]:>10.2f} 度",
        f"  Ry: {euler[1]:>10.2f} 度",
        f"  Rz: {euler[2]:>10.2f} 度",
        "",
        "【旋转矩阵】",
    ]
    for row in result.R_cam2gripper:
        lines.append(f"  [{row[0]:>10.6f}, {row[1]:>10.6f}, {row[2]:>10.6f}]")
    lines += [
        "",
        "=" * 60,
        "           误差分析",
        "=" * 60,
        "",
        "【标定板在基座下的平均位置】",
        f"  平均位置: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] mm",
        f"  位置标准差: [{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}] mm",
        "",
        f"  平均误差: {result.avg_error:.2f} mm",
        f"  标定质量: {result.quality_label()}",
    ]

    # 多方法对比表
    if result.all_method_results:
        lines += [
            "",
            "=" * 60,
            "           多方法对比",
            "=" * 60,
            "",
            f"  {'排名':<4} {'方法':<12} {'误差(mm)':<10} {'质量':<6}",
            f"  {'----':<4} {'----------':<12} {'--------':<10} {'----':<6}",
        ]
        for rank, mr in enumerate(result.all_method_results, 1):
            marker = " <-- 当前" if mr['name'] == result.method_name else ""
            lines.append(
                f"  {rank:<4} {mr['name']:<12} {mr['avg_error']:<10.2f} {mr['quality']:<6}{marker}"
            )

    # 异常点剔除摘要
    if result.rejected_indices:
        lines += [
            "",
            "=" * 60,
            "           异常点剔除",
            "=" * 60,
            "",
            f"  剔除 {len(result.rejected_indices)} 个异常点: {result.rejected_indices}",
            f"  保留 {len(result.used_indices)} 组数据",
        ]

    # 逐点误差表
    if result.per_pose_errors is not None:
        lines += [
            "",
            "=" * 60,
            "           逐点误差",
            "=" * 60,
            "",
        ]
        indices = result.used_indices if result.used_indices is not None else list(range(len(result.per_pose_errors)))
        for local_i, (global_i, err) in enumerate(zip(indices, result.per_pose_errors)):
            flag = " ***" if err > 5.0 else ""
            lines.append(f"  #{global_i:<3}  {err:>7.2f} mm{flag}")

    lines.append("=" * 60)
    logger.info("\n%s", "\n".join(lines))


def print_loo_results(loo_results: list, baseline_error: float) -> None:
    """输出留一法分析结果。"""
    lines = [
        "=" * 60,
        "           留一法 (Leave-One-Out) 分析",
        "=" * 60,
        "",
        f"  基线误差: {baseline_error:.2f} mm",
        "",
        f"  {'位姿':<6} {'去掉后误差(mm)':<16} {'变化(mm)':<10} {'建议':<8}",
        f"  {'----':<6} {'--------------':<16} {'--------':<10} {'------':<8}",
    ]
    for r in loo_results:
        delta = r['delta_error']
        suggestion = ""
        if delta < -0.5:
            suggestion = "建议剔除"
        elif delta < -0.1:
            suggestion = "可疑"
        lines.append(
            f"  #{r['index']:<5} {r['avg_error_without']:<16.2f} {delta:<+10.2f} {suggestion}"
        )

    # 总结
    bad_points = [r for r in loo_results if r['delta_error'] < -0.5]
    if bad_points:
        lines += [
            "",
            f"  发现 {len(bad_points)} 个疑似坏点: "
            + ", ".join(f"#{r['index']}" for r in bad_points),
        ]
    else:
        lines.append("\n  未发现明显坏点。")

    lines.append("=" * 60)
    logger.info("\n%s", "\n".join(lines))
