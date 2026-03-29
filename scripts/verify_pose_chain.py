#!/usr/bin/env python3
"""
离线全链路精度验证 | Offline Full-Chain Pose Verification

原理：已知配对 (aruco_i, tcp_i) 及手眼标定结果，利用坐标链预测 tcp_j，
与真值 tcp_j 对比，反映手眼标定 + ArUco 检测的**联合误差**。

公式：
    T_base_target = T_base_tool_i @ T_tool_cam @ T_cam_target_i
    T_base_tool_j_pred = T_base_target @ inv(T_cam_target_j) @ inv(T_tool_cam)

配对方式：全组合 N*(N-1)/2（i < j），充分利用所有数据点。

用法：
    python scripts/verify_pose_chain.py
    python scripts/verify_pose_chain.py \
        --aruco-file aruco_pose_id0.txt \
        --tcp-file robot_tcp_id0.txt \
        --hand-eye Handeyecalib/hand_eye_result.txt
"""

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file
from robovision.geometry.transforms import (
    pose_to_matrix,
    matrix_to_pose,
    compute_new_tool_pose,
    rotation_angle_deg,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='离线全链路精度验证')
    parser.add_argument('--aruco-file', type=str, default='data/aruco/aruco_pose_id1.txt',
                        help='ArUco 位姿文件（毫米，ZYX内旋）')
    parser.add_argument('--tcp-file', type=str, default='data/aruco/robot_tcp_id1.txt',
                        help='机械臂 TCP 文件，与 aruco-file 行对行（毫米）')
    parser.add_argument('--hand-eye', type=str,
                        default='data/handeye/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵，cam2gripper）')
    parser.add_argument('--marker-id', type=int, default=1,
                        help='Marker ID，仅用于日志提示（默认 0）')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='打印每一对的详细误差')
    return parser.parse_args()


def _fmt_pose(pose):
    """格式化 (x,y,z,rx,ry,rz) 为对齐字符串"""
    return (f"({pose[0]:8.4f},{pose[1]:8.4f},{pose[2]:8.4f}"
            f" | {pose[3]:7.2f},{pose[4]:7.2f},{pose[5]:7.2f})")


def main():
    args = parse_args()

    # ——— 加载数据 ———
    logger.info("加载 ArUco 位姿: %s", args.aruco_file)
    aruco_poses = load_pose_file(args.aruco_file)   # (N, 6)，mm，ZYX 欧拉角

    logger.info("加载 TCP 位姿:   %s", args.tcp_file)
    tcp_poses = load_pose_file(args.tcp_file)        # (N, 6)，mm，ZYX 欧拉角

    if len(aruco_poses) != len(tcp_poses):
        logger.error(
            "aruco (%d 行) 与 tcp (%d 行) 行数不一致，请检查数据对齐",
            len(aruco_poses), len(tcp_poses)
        )
        sys.exit(1)

    N = len(aruco_poses)
    if N < 2:
        logger.error("至少需要 2 组数据，当前只有 %d 组", N)
        sys.exit(1)

    logger.info("加载手眼标定: %s", args.hand_eye)
    T_tool_cam = load_hand_eye_result(args.hand_eye)  # cam2gripper (tool_from_cam)
    # hand_eye_result.txt 存储 cam2gripper，即 T_tool_cam，单位 mm

    logger.info("数据量：%d 组，将生成 %d 对全组合", N, N * (N - 1) // 2)

    # ——— 全组合验证 ———
    x_errors = []       # mm, 基座 X 轴误差（带符号）
    y_errors = []       # mm, 基座 Y 轴误差（带符号）
    z_errors = []       # mm, 基座 Z 轴误差（带符号）
    rot_errors = []     # deg
    pair_indices = []    # (i, j) for each pair

    # 逐点误差累加器
    point_x_errors = [[] for _ in range(N)]
    point_y_errors = [[] for _ in range(N)]
    point_z_errors = [[] for _ in range(N)]
    point_r_errors = [[] for _ in range(N)]

    # 逐点预测 TCP 累加器（用 i 预测 j 时，记录 j 的预测值）
    point_preds = [[] for _ in range(N)]  # point_preds[k] = list of predicted (x,y,z,rx,ry,rz)

    pair_idx = 0
    for i in range(N):
        T_tcp_i = pose_to_matrix(tcp_poses[i])
        T_arc_i = pose_to_matrix(aruco_poses[i])

        for j in range(i + 1, N):
            T_arc_j = pose_to_matrix(aruco_poses[j])
            T_tcp_j_actual = pose_to_matrix(tcp_poses[j])

            # i -> j
            T_tcp_j_pred = compute_new_tool_pose(
                base_from_tool_old=T_tcp_i,
                cam_from_target_old=T_arc_i,
                tool_from_cam=T_tool_cam,
                cam_from_target_new=T_arc_j,
            )
            # j -> i
            T_tcp_i_pred = compute_new_tool_pose(
                base_from_tool_old=T_tcp_j_actual,
                cam_from_target_old=T_arc_j,
                tool_from_cam=T_tool_cam,
                cam_from_target_new=T_arc_i,
            )

            diff = T_tcp_j_pred[:3, 3] - T_tcp_j_actual[:3, 3]
            x_err = diff[0]
            y_err = diff[1]
            z_err = diff[2]
            r_err = rotation_angle_deg(T_tcp_j_pred[:3, :3], T_tcp_j_actual[:3, :3])
            x_errors.append(x_err)
            y_errors.append(y_err)
            z_errors.append(z_err)
            rot_errors.append(r_err)
            pair_indices.append((i, j))

            # 累加到两端点
            point_x_errors[i].append(abs(x_err))
            point_x_errors[j].append(abs(x_err))
            point_y_errors[i].append(abs(y_err))
            point_y_errors[j].append(abs(y_err))
            point_z_errors[i].append(abs(z_err))
            point_z_errors[j].append(abs(z_err))
            point_r_errors[i].append(r_err)
            point_r_errors[j].append(r_err)

            # 记录预测值（位置 + 欧拉角）
            point_preds[i].append(matrix_to_pose(T_tcp_i_pred))
            point_preds[j].append(matrix_to_pose(T_tcp_j_pred))

            if args.verbose:
                pair_idx += 1
                pred_j = matrix_to_pose(T_tcp_j_pred)
                actual_j = tcp_poses[j]
                print(f"[{pair_idx:03d}] ({i}->{j})  "
                      f"Δx={x_err:+.3f}  Δy={y_err:+.3f}  Δz={z_err:+.3f}mm  "
                      f"r_err={r_err:.4f}°")
                print(f"       真值  ={_fmt_pose(actual_j)}")
                print(f"       预测  ={_fmt_pose(pred_j)}")

    x_errors = np.array(x_errors)
    y_errors = np.array(y_errors)
    z_errors = np.array(z_errors)
    rot_errors = np.array(rot_errors)
    n_pairs = len(x_errors)

    if args.verbose:
        print()

    # ——— 逐点平均误差 + 真值 vs 预测均值 ———
    point_x_means = np.array([np.mean(e) for e in point_x_errors])
    point_y_means = np.array([np.mean(e) for e in point_y_errors])
    point_z_means = np.array([np.mean(e) for e in point_z_errors])
    point_r_means = np.array([np.mean(e) for e in point_r_errors])
    # 综合平移误差 = sqrt(x^2 + y^2 + z^2) 的均值
    point_total_means = np.array([
        np.mean(np.sqrt(np.array(point_x_errors[k])**2 +
                        np.array(point_y_errors[k])**2 +
                        np.array(point_z_errors[k])**2))
        for k in range(N)
    ])
    worst_idx = int(np.argmax(point_total_means))

    print("【逐点平均误差】")
    for k in range(N):
        marker = "  ***" if k == worst_idx else ""
        actual_k = tcp_poses[k]  # 真值 (x,y,z,rx,ry,rz)
        pred_arr = np.array(point_preds[k])  # (n_pairs, 6)
        pred_mean = pred_arr.mean(axis=0)
        t_diff_mm = pred_mean[:3] - actual_k[:3]
        r_diff_deg = pred_mean[3:] - actual_k[3:]
        print(f"  #{k}:  |Δx|={point_x_means[k]:.3f}  |Δy|={point_y_means[k]:.3f}  "
              f"|Δz|={point_z_means[k]:.3f}mm  r_mean={point_r_means[k]:.4f}°  "
              f"(参与 {len(point_x_errors[k])} 对){marker}")
        print(f"       真值  ={_fmt_pose(actual_k)}")
        print(f"       预测  ={_fmt_pose(pred_mean)}"
              f"  Δ=({t_diff_mm[0]:+6.2f},{t_diff_mm[1]:+6.2f},{t_diff_mm[2]:+6.2f})mm"
              f" ({r_diff_deg[0]:+5.2f},{r_diff_deg[1]:+5.2f},{r_diff_deg[2]:+5.2f})°")
    print()

    # ——— 输出统计 ———
    abs_x = np.abs(x_errors)
    abs_y = np.abs(y_errors)
    abs_z = np.abs(z_errors)
    total_t = np.sqrt(x_errors**2 + y_errors**2 + z_errors**2)

    def _stat_block(name, arr, fmt=".3f"):
        return [
            f"【{name}】",
            f"  均值:   {arr.mean():>8{fmt}}",
            f"  中位数: {np.median(arr):>8{fmt}}",
            f"  最大值: {arr.max():>8{fmt}}",
            f"  标准差: {arr.std():>8{fmt}}",
        ]

    lines = [
        "=" * 60,
        f"  全链路精度验证 — Marker ID {args.marker_id}",
        f"  数据量: {N} 组  |  配对数: {n_pairs}",
        "=" * 60,
        "",
        *_stat_block("X 轴误差 (mm)", abs_x),
        "",
        *_stat_block("Y 轴误差 (mm)", abs_y),
        "",
        *_stat_block("Z 轴误差 (mm)", abs_z),
        "",
        *_stat_block("总平移误差 (mm)", total_t),
        "",
        *_stat_block("旋转误差 (°)", rot_errors, fmt=".4f"),
        "",
        f"  偏差方向 (mm):  X={x_errors.mean():+.3f}  Y={y_errors.mean():+.3f}  Z={z_errors.mean():+.3f}",
        "",
        "说明：误差反映手眼标定 + ArUco 检测的联合精度",
        "=" * 60,
    ]
    print("\n".join(lines))


if __name__ == '__main__':
    main()
