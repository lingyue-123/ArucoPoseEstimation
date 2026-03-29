#!/usr/bin/env python3
"""
ArUco 检测运动一致性验证 | ArUco Motion Consistency Verification

自动从 TCP 数据中检测纯平移段和纯旋转段，每段内部做 C(n,2) 全配对比较。
不同段之间不跨组配对。与手眼标定解耦，单独验证 ArUco 检测精度。

误差定义（每对 i,j）：
    平移：|d_tool - d_cam|  （mm），d_cam 来自相对变换 T_ci_t @ inv(T_cj_t)
    旋转：|theta_tool - theta_cam|  （°）

用法：
    python scripts/verify_aruco_motion.py
    python scripts/verify_aruco_motion.py --aruco-file aruco.txt --tcp-file tcp.txt -v
"""

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from robovision.io.pose_file import load_pose_file
from robovision.geometry.transforms import pose_to_matrix, rotation_angle_deg

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='ArUco 运动一致性验证（自动分段，C(N,2) 全配对）')
    parser.add_argument('--aruco-file', type=str, default='data/aruco/aruco_pose_id0.txt',
                        help='ArUco 位姿文件（毫米，度）')
    parser.add_argument('--tcp-file', type=str, default='data/aruco/robot_tcp_id0.txt',
                        help='机械臂 TCP 文件，与 aruco-file 行对行（毫米）')
    parser.add_argument('--angle-tol', type=float, default=0.05,
                        help='角度变化容差（度，默认 0.05），相邻帧角度差 < 此值视为纯平移')
    parser.add_argument('--pos-tol', type=float, default=0.5,
                        help='位置变化容差（毫米，默认 0.5），相邻帧位移 < 此值视为纯旋转')
    parser.add_argument('--marker-id', type=int, default=0,
                        help='Marker ID，仅用于日志提示（默认 0）')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='打印每对详细信息')
    return parser.parse_args()


# ------------------------------------------------------------------
# 自动分段
# ------------------------------------------------------------------

def auto_detect_segments(tcp_poses: np.ndarray, angle_tol: float, pos_tol: float):
    """
    根据机械臂 TCP 数据自动检测连续的纯平移段和纯旋转段。

    逐行与前一行比较：
    - 角度变化 < angle_tol 且位置变化 > pos_tol → 纯平移 'T'
    - 位置变化 < pos_tol 且角度变化 > angle_tol → 纯旋转 'R'
    - 其他 → 混合 'M'（不归入任何段）

    Returns:
        trans_ranges: [(start, end), ...] 纯平移段列表
        rot_ranges:   [(start, end), ...] 纯旋转段列表
        labels:       每行的分类标签列表
    """
    N = len(tcp_poses)
    labels = ['M'] * N

    for i in range(1, N):
        pos_diff = np.linalg.norm(tcp_poses[i, :3] - tcp_poses[i - 1, :3])
        R_prev = pose_to_matrix(tcp_poses[i - 1])[:3, :3]
        R_curr = pose_to_matrix(tcp_poses[i])[:3, :3]
        ang_diff = rotation_angle_deg(R_prev, R_curr)

        if ang_diff < angle_tol and pos_diff > pos_tol:
            labels[i] = 'T'
        elif pos_diff < pos_tol and ang_diff > angle_tol:
            labels[i] = 'R'

    # 第 0 行：继承第 1 行的标签
    if N > 1 and labels[1] != 'M':
        labels[0] = labels[1]

    def indices_to_ranges(indices):
        """[0,1,2,5,6,7] → [(0,3), (5,8)]"""
        if not indices:
            return []
        ranges = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                ranges.append((start, prev + 1))
                start = prev = idx
        ranges.append((start, prev + 1))
        return ranges

    trans_ranges = indices_to_ranges([i for i, l in enumerate(labels) if l == 'T'])
    rot_ranges = indices_to_ranges([i for i, l in enumerate(labels) if l == 'R'])

    return trans_ranges, rot_ranges, labels


# ------------------------------------------------------------------
# 核心验证函数
# ------------------------------------------------------------------

def rel_cam_transform(T_ci_t, T_cj_t):
    """相机相对变换 ^Ci T_Cj = T_ci_t @ inv(T_cj_t)"""
    return T_ci_t @ np.linalg.inv(T_cj_t)


def summarize(arr, name, unit=""):
    arr = np.asarray(arr, dtype=float)
    print(f"  {name}:")
    print(f"    mean = {arr.mean():.4f} {unit}".rstrip())
    if arr.size > 1:
        print(f"    std  = {arr.std(ddof=1):.4f} {unit}".rstrip())
    print(f"    max  = {arr.max():.4f} {unit}".rstrip())
    print(f"    min  = {arr.min():.4f} {unit}".rstrip())


def verify_translation(tcp_poses, aruco_poses, indices, verbose=False):
    """
    纯平移段验证（C(n,2) 全配对）。
    比较 d_tool vs d_cam，以及 cam_rel_rot（纯平移时应接近 0）。
    """
    n = len(indices)
    if n < 2:
        return None

    T_tool = [pose_to_matrix(tcp_poses[i]) for i in indices]
    T_ct = [pose_to_matrix(aruco_poses[i]) for i in indices]
    n_pairs = n * (n - 1) // 2

    abs_errs_mm = []
    rel_errs = []
    cam_rel_rot_deg = []

    print("=" * 90)
    print(f"  [Translation] n={n}, pairs=C({n},2)={n_pairs}")
    print(f"  Metric: |d_tool - d_cam| (mm), rel_err (%), cam_rel_rot (deg, should be ~0)")
    print("=" * 90)

    idx = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            idx += 1
            i_g, j_g = indices[ii], indices[jj]

            d_tool = np.linalg.norm(T_tool[jj][:3, 3] - T_tool[ii][:3, 3])

            T_ci_cj = rel_cam_transform(T_ct[ii], T_ct[jj])
            d_cam = np.linalg.norm(T_ci_cj[:3, 3])
            rot_deg = rotation_angle_deg(np.eye(3), T_ci_cj[:3, :3])

            abs_err = abs(d_cam - d_tool)
            rel_err = abs_err / d_tool if d_tool > 1e-12 else float('nan')

            abs_errs_mm.append(abs_err)
            rel_errs.append(rel_err)
            cam_rel_rot_deg.append(rot_deg)

            if verbose:
                print(f"  [{idx:03d}] ({i_g}->{j_g})"
                      f"  d_tool={d_tool:.3f}mm  d_cam={d_cam:.3f}mm"
                      f"  err={abs_err:.3f}mm  rel={rel_err * 100:.2f}%"
                      f"  cam_rot={rot_deg:.4f}")

    abs_errs_mm = np.array(abs_errs_mm)
    rel_errs = np.array(rel_errs)
    cam_rel_rot_deg = np.array(cam_rel_rot_deg)

    print("-" * 90)
    print("  [Translation] Summary")
    summarize(abs_errs_mm, "abs_err", "mm")
    valid_rel = rel_errs[np.isfinite(rel_errs)]
    if len(valid_rel) > 0:
        summarize(valid_rel * 100, "rel_err", "%")
    summarize(cam_rel_rot_deg, "cam_rel_rot (should be ~0)", "deg")

    thr_mean_mm, thr_max_mm = 2.0, 7.0
    thr_rot_deg = 1.2
    ok = (abs_errs_mm.mean() < thr_mean_mm and
          abs_errs_mm.max() < thr_max_mm and
          cam_rel_rot_deg.max() < thr_rot_deg)
    print(f"  Verdict: mean<{thr_mean_mm}mm, max<{thr_max_mm}mm, cam_rot<{thr_rot_deg}deg"
          f"  => {'PASS' if ok else 'FAIL'}")
    print("=" * 90)
    print()
    return ok


def verify_rotation(tcp_poses, aruco_poses, indices, verbose=False):
    """
    纯旋转段验证（C(n,2) 全配对）。
    比较 theta_tool vs theta_cam。
    """
    n = len(indices)
    if n < 2:
        return None

    T_tool = [pose_to_matrix(tcp_poses[i]) for i in indices]
    T_ct = [pose_to_matrix(aruco_poses[i]) for i in indices]
    n_pairs = n * (n - 1) // 2

    # 位置漂移 sanity check
    tool_pos = np.array([T[:3, 3] for T in T_tool])
    drift = np.linalg.norm(tool_pos - tool_pos[0], axis=1)
    max_drift_mm = drift.max()

    abs_err_theta = []

    print("=" * 90)
    print(f"  [Rotation] n={n}, pairs=C({n},2)={n_pairs}")
    print(f"  Sanity: max tool position drift = {max_drift_mm:.3f} mm")
    print(f"  Metric: |theta_tool - theta_cam| (deg)")
    print("=" * 90)

    idx = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            idx += 1
            i_g, j_g = indices[ii], indices[jj]

            R_tool_rel = T_tool[jj][:3, :3] @ T_tool[ii][:3, :3].T
            theta_tool = rotation_angle_deg(np.eye(3), R_tool_rel)

            T_ci_cj = rel_cam_transform(T_ct[ii], T_ct[jj])
            theta_cam = rotation_angle_deg(np.eye(3), T_ci_cj[:3, :3])

            err = abs(theta_cam - theta_tool)
            abs_err_theta.append(err)

            if verbose:
                print(f"  [{idx:03d}] ({i_g}->{j_g})"
                      f"  theta_tool={theta_tool:.4f}  theta_cam={theta_cam:.4f}"
                      f"  err={err:.4f}")

    abs_err_theta = np.array(abs_err_theta)

    print("-" * 90)
    print("  [Rotation] Summary")
    summarize(abs_err_theta, "|theta_tool - theta_cam|", "deg")

    thr_mean_deg, thr_max_deg = 0.2, 0.5
    ok = abs_err_theta.mean() < thr_mean_deg and abs_err_theta.max() < thr_max_deg
    print(f"  Verdict: mean<{thr_mean_deg}deg, max<{thr_max_deg}deg"
          f"  => {'PASS' if ok else 'FAIL'}")
    print("=" * 90)
    print()
    return ok


# ------------------------------------------------------------------

def main():
    args = parse_args()

    # --- 加载数据 ---
    logger.info("ArUco: %s", args.aruco_file)
    aruco_poses = load_pose_file(args.aruco_file)

    logger.info("TCP:   %s", args.tcp_file)
    tcp_poses = load_pose_file(args.tcp_file)

    if len(aruco_poses) != len(tcp_poses):
        logger.error("aruco (%d) 与 tcp (%d) 行数不一致", len(aruco_poses), len(tcp_poses))
        sys.exit(1)

    N = len(aruco_poses)
    if N < 2:
        logger.error("至少需要 2 组数据，当前只有 %d 组", N)
        sys.exit(1)

    # --- 自动分段 ---
    trans_ranges, rot_ranges, labels = auto_detect_segments(
        tcp_poses, args.angle_tol, args.pos_tol)

    label_str = ''.join(labels)

    print()
    print("=" * 90)
    print(f"  ArUco Motion Consistency — Marker ID {args.marker_id}")
    print(f"  Data: {N} poses | Decoupled from hand-eye calibration")
    print(f"  Auto-segment: {label_str}")
    print(f"    T=translation, R=rotation, M=mixed (skipped)")
    n_trans = sum(e - s for s, e in trans_ranges)
    n_rot = sum(e - s for s, e in rot_ranges)
    print(f"    Translation: {len(trans_ranges)} group(s), {n_trans} poses")
    print(f"    Rotation:    {len(rot_ranges)} group(s), {n_rot} poses")
    print("=" * 90)
    print()

    # --- 每段独立验证 ---
    found_any = False

    for gi, (s, e) in enumerate(trans_ranges):
        seg = list(range(s, e))
        if len(seg) >= 2:
            found_any = True
            print(f">>> Translation Group {gi + 1}: indices [{s}, {e}), {len(seg)} poses\n")
            verify_translation(tcp_poses, aruco_poses, seg, verbose=args.verbose)

    for gi, (s, e) in enumerate(rot_ranges):
        seg = list(range(s, e))
        if len(seg) >= 2:
            found_any = True
            print(f">>> Rotation Group {gi + 1}: indices [{s}, {e}), {len(seg)} poses\n")
            verify_rotation(tcp_poses, aruco_poses, seg, verbose=args.verbose)

    if not found_any:
        logger.warning("未检测到有效段 (>=2 poses)，请检查数据或调整 --angle-tol / --pos-tol")


if __name__ == '__main__':
    main()
