#!/usr/bin/env python3
"""
极简版 求 One-Shot Move 的目标 TCP 位姿：
仅读一张图片并输出 final_pose（无相机拉流/无机械臂控制/无可视化）。
"""

import argparse
import logging
import os

import cv2
import numpy as np

from robovision.cameras.base import CameraIntrinsics
from robovision.detection.aruco import ArucoDetector
from robovision.geometry.transforms import (
    pose_to_matrix,
    matrix_to_pose,
    compute_new_tool_pose,
)

# ----- 硬编码参数 -----
# 目标 Marker ID（硬编码）
TARGET_MARKER = 0

# 参考 ArUco 6D 位姿（xyz, rpy，单位 mm/deg）
REF_POSE_XYZRPY = [27.712, 75.672, 318.761, 175.3878, 5.1931, 1.2222]

# 手眼标定矩阵（工具末端到摄像机）
T_C2G = np.array([
    [0.03313067, -0.99869212, 0.03894114, 82.43],
    [0.99467736, 0.02914385, -0.09883107, 40.3258],
    [0.09756692, 0.04200821, 0.994342, 130.754],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# 相机内参矩阵
CAMERA_MATRIX = np.array([
    [1818.77, 0.0, 626.282],
    [0.0, 1819.39, 515.309],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# 畸变系数
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# 相机分辨率
CAMERA_WIDTH = 2000
CAMERA_HEIGHT = 1500

# Marker 物理边长（硬编码，单位 mm）
MARKER_LENGTH_MM = 15.0


def aruco_to_matrix(data: dict) -> np.ndarray:
    """将 ArUco 位姿数据转换为 4x4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = data['R_m2c']
    T[:3, 3] = data['tvec_m2c'].flatten()
    return T


def main():
    """主函数：解析参数，检测 Marker，计算目标 TCP 位姿。"""
    parser = argparse.ArgumentParser(
        description='clean.py: 读取单张图片，输出 final_pose。'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='待检测图像路径'
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help='使用 RAW 检测模式（原生 OpenCV ArUco）'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s'
    )

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # 参考 Marker 位姿矩阵
    T_aruco2cam_ref = pose_to_matrix(REF_POSE_XYZRPY)

    # 相机内参
    intrinsics = CameraIntrinsics(
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        name='hardcoded',
    )

    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    if args.raw:
        # 使用原生 OpenCV ArUco 检测
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        raw_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # 检测 Marker
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = raw_detector.detectMarkers(gray)
        if ids is None:
            raise RuntimeError(f"No markers found in image {args.image}")

        # 查找目标 Marker
        target_idx = next(
            (i for i, mid in enumerate(ids.flatten()) if int(mid) == TARGET_MARKER),
            None
        )
        if target_idx is None:
            raise RuntimeError(f"Target marker {TARGET_MARKER} not found")

        # 构造 Marker 物理坐标
        half = MARKER_LENGTH_MM / 2.0
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

        # 亚像素角点优化
        c = corners[target_idx].reshape(-1, 1, 2).astype(np.float32)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.cornerSubPix(
            gray_blur, c, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        img_pts = c.reshape(4, 2).astype(np.float64)

        # PnP 求解
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, intrinsics.camera_matrix, intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            raise RuntimeError("solvePnP failed")
        
        # 计算重投影误差
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(
            proj_pts.reshape(-1, 2) - img_pts, axis=1)))

        # 输出
        R, _ = cv2.Rodrigues(rvec)
        target_data = {'R_m2c': R, 'tvec_m2c': tvec}
    else:
        # 使用增强 ArucoDetector
        detector = ArucoDetector(
            intrinsics=intrinsics,
            valid_ids=[TARGET_MARKER],
            marker_sizes={TARGET_MARKER: MARKER_LENGTH_MM},
            dictionary='DICT_4X4_50',
            use_kalman=False,
        )
        result = detector.detect(img)
        target_data = result.get(TARGET_MARKER)
        if target_data is None:
            raise RuntimeError(f"Target marker {TARGET_MARKER} not found")

    # 当前 Marker 位姿矩阵
    T_aruco2cam_cur = aruco_to_matrix(target_data)

    # 当前 TCP 位姿（相对于基座，单位 mm/deg）
    # 注意：此处硬编码为零，实际需从 JAKA 机械臂接口获取
    tcp_cur = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T_g2b_cur = pose_to_matrix(tcp_cur)

    # 计算目标 TCP 位姿
    T_g2b_target = compute_new_tool_pose(
        base_from_tool_old=T_g2b_cur,
        cam_from_target_old=T_aruco2cam_cur,
        tool_from_cam=T_C2G,
        cam_from_target_new=T_aruco2cam_ref,
    )

    # 输出最终位姿
    final_pose = matrix_to_pose(T_g2b_target)

    print("#sym:final_pose")
    print(final_pose.tolist())


if __name__ == '__main__':
    main()
 