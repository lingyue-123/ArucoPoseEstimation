#!/usr/bin/env python3
"""
自动 ArUco 位姿采集脚本 | Automatic ArUco Pose Collection

围绕给定的起始 TCP 位姿，在可配置立方体范围内自动移动并采集 ArUco 位姿 + robot TCP。
替代手动示教器操作 + 按 s 保存的流程。

检测策略：到位后连采 N 帧（默认 5），对每个 marker 的 tvec/euler 取 median，
同时计算重投影误差，超阈值则跳过。默认使用 RAW 检测（IPPE_SQUARE，无时序滤波），
因为逐点采集场景下 Kalman/SLERP 无法建立时序状态。

用法：
    python scripts/auto_collect_aruco.py                        # 实际运行（读当前 TCP 为中心）
    python scripts/auto_collect_aruco.py --dry-run              # 只打印轨迹，不运动
    python scripts/auto_collect_aruco.py --start-pose -350 750 350 -177.5 -1.6 89.5
    python scripts/auto_collect_aruco.py --no-raw               # 使用 ArucoDetector（Kalman+多方法PnP）

⚠️ 单位约定：
    - move_linear() 输入：mm, deg
    - get_tcp_pose() 输出：mm, deg
    - 起始位姿/轨迹点定义：mm, deg
"""

import argparse
import logging
import os
import signal
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.aruco import ArucoDetector
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose, rotation_angle_deg, rotmat_to_euler,
)
from robovision.robot import build_robot
from robovision.io.pose_file import PoseFileWriter
from robovision.io.image_saver import ImageSaver
from robovision.visualization.aruco_overlay import draw_aruco_result

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# ---------------------------------------------------------------------------
# 采样策略参数（单位：mm, deg）
# ---------------------------------------------------------------------------

DEFAULT_XY_SPAN_MM = 30.0
DEFAULT_Z_SPAN_MM = 30.0
DEFAULT_ROT_DEG = 5.0
MAX_DIRECT_STEP_TRANS_MM = 200.0
MAX_DIRECT_STEP_ROT_DEG = 20.0


def build_aruco_waypoints(center_pose, xy_span=DEFAULT_XY_SPAN_MM,
                          z_span=DEFAULT_Z_SPAN_MM, rot_delta=DEFAULT_ROT_DEG):
    """围绕中心 TCP 生成 ArUco 采集用采样位姿列表。

    策略：
    1. 纯平移（姿态不变）：中心 + ±X, ±Y, ±Z + 对角线 → ~9 点
    2. 纯旋转（在中心 + 2 个对角位置做）：±Rx, ±Ry, ±Rz → 每位置 6 个变体 → ~18 点
    总计 ~25 个 waypoint。
    """
    cx, cy, cz, crx, cry, crz = [float(v) for v in center_pose]
    center_T = pose_to_matrix(center_pose)

    # --- Phase 1: 纯平移 ---
    trans_offsets = [
        (0.0, 0.0, 0.0),           # 中心
        ( xy_span, 0.0, 0.0),      # +X
        (-xy_span, 0.0, 0.0),      # -X
        (0.0,  xy_span, 0.0),      # +Y
        (0.0, -xy_span, 0.0),      # -Y
        (0.0, 0.0,  z_span),       # +Z
        (0.0, 0.0, -z_span),       # -Z
        ( xy_span,  xy_span, 0.0), # 对角线 1
        (-xy_span, -xy_span, 0.0), # 对角线 2
    ]

    # --- Phase 2: 纯旋转变体 ---
    rot_variants = [
        ( rot_delta, 0.0, 0.0),    # +Rx
        (-rot_delta, 0.0, 0.0),    # -Rx
        (0.0,  rot_delta, 0.0),    # +Ry
        (0.0, -rot_delta, 0.0),    # -Ry
        (0.0, 0.0,  rot_delta),    # +Rz
        (0.0, 0.0, -rot_delta),    # -Rz
    ]

    # 在这些平移位置上做旋转变体（中心 + 2 个对角位置）
    rotate_position_indices = [0, 7, 8]

    waypoints = []
    seen = set()

    def _add(pose):
        key = tuple(round(float(v), 3) for v in pose)
        if key not in seen:
            seen.add(key)
            waypoints.append([float(v) for v in pose])

    # Phase 1: 所有纯平移点（姿态不变）
    for dx, dy, dz in trans_offsets:
        _add([cx + dx, cy + dy, cz + dz, crx, cry, crz])

    # Phase 2: 在选定位置上做旋转变体
    for pos_idx in rotate_position_indices:
        dx, dy, dz = trans_offsets[pos_idx]
        for drx, dry, drz in rot_variants:
            target_T = center_T.copy()
            target_T[:3, 3] = [cx + dx, cy + dy, cz + dz]
            rot_pose = [0.0, 0.0, 0.0, crx + drx, cry + dry, crz + drz]
            target_T[:3, :3] = pose_to_matrix(rot_pose)[:3, :3]
            _add(matrix_to_pose(target_T))

    # 末尾追加中心点（与第一个点相同位姿），用于首尾一致性验证
    center = [cx, cy, cz, crx, cry, crz]
    waypoints.append([float(v) for v in center])

    return waypoints


# ---------------------------------------------------------------------------
# Raw ArUco 检测（从 collect_aruco_poses_raw.py 复用）
# ---------------------------------------------------------------------------

def detect_raw(gray, aruco_detector, valid_ids, marker_sizes, K, dist):
    """检测：detectMarkers → cornerSubPix → solvePnP(IPPE_SQUARE) + 重投影误差。"""
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return {}

    result = {}
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, mid in enumerate(ids.flatten()):
        mid = int(mid)
        if mid not in valid_ids:
            continue

        c = corners[i].reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray_blur, c, (5, 5), (-1, -1), criteria)
        img_pts = c.reshape(4, 2).astype(np.float64)

        marker_len = marker_sizes.get(mid, 100.0)
        half = marker_len / 2
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue

        # 计算重投影误差
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        reproj_err = float(np.mean(np.linalg.norm(
            proj_pts.reshape(-1, 2) - img_pts, axis=1)))

        R, _ = cv2.Rodrigues(rvec)
        euler_zyx = rotmat_to_euler(R, order='ZYX')
        result[mid] = {
            'raw_corners': corners[i].reshape(4, 2),
            'filtered_corners': img_pts,
            'rvec_m2c': rvec, 'tvec_m2c': tvec, 'R_m2c': R,
            'euler_m2c_zyx': euler_zyx,
            'reproj_err': reproj_err,
            'method': 'IPPE_SQ', 'status': 'OK',
            'marker_length': marker_len,
        }
    return result


def multi_frame_detect(camera, detect_fn, n_frames, show_display=False,
                       win=None, disp_scale=0.4, info_text=""):
    """连采 n_frames 帧检测，选重投影误差最小的那帧结果。

    返回 (best_result, best_frame)，best_result 格式同 detect_raw() 输出。
    """
    all_results = []   # [(result_dict, frame), ...]
    for k in range(n_frames):
        ok_f, frame = camera.read_frame(timeout=1.0)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        result = detect_fn(gray)
        if result:
            all_results.append((result, frame))
        if show_display and frame is not None:
            small = cv2.resize(frame, (int(frame.shape[1] * disp_scale),
                                       int(frame.shape[0] * disp_scale)))
            cv2.putText(small, f"{info_text} capture {k+1}/{n_frames}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow(win, small)
            cv2.waitKey(1)

    if not all_results:
        return {}, None

    # 选平均重投影误差最小的那帧
    best_idx = 0
    best_avg_err = float('inf')
    for idx, (result, _) in enumerate(all_results):
        avg_err = np.mean([d['reproj_err'] for d in result.values()])
        if avg_err < best_avg_err:
            best_avg_err = avg_err
            best_idx = idx

    best_result, best_frame = all_results[best_idx]
    # 标注采集信息
    for mid in best_result:
        best_result[mid]['method'] = f"IPPE_SQ(best/{len(all_results)})"
    logger.debug("multi_frame: %d/%d 帧有效，选第 %d 帧 (avg_reproj=%.3fpx)",
                 len(all_results), n_frames, best_idx + 1, best_avg_err)

    return best_result, best_frame


# ---------------------------------------------------------------------------
# 运动控制
# ---------------------------------------------------------------------------

_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    logger.warning("收到中断信号，将在当前动作完成后安全停止...")


def pose_delta(cur_pose, target_pose):
    """计算两个位姿间的平移距离 (mm) 和旋转角度 (deg)。"""
    cur_T = pose_to_matrix(cur_pose)
    target_T = pose_to_matrix(target_pose)
    delta_T = np.linalg.inv(cur_T) @ target_T
    trans_mm = float(np.linalg.norm(delta_T[:3, 3]))
    rot_deg = rotation_angle_deg(np.eye(3), delta_T[:3, :3])
    return trans_mm, rot_deg


def move_with_display(robot, target_pose, timeout, camera=None,
                      show_display=False, win=None, disp_scale=0.4,
                      info_text="", stop_flag_fn=None):
    """发送运动指令，轮询等待完成，期间持续刷新相机画面避免冻结。"""
    from third_party.robot_driver.robot_driver_interface import CartesianPose
    target = CartesianPose(
        x=target_pose[0], y=target_pose[1], z=target_pose[2],
        rx=target_pose[3], ry=target_pose[4], rz=target_pose[5],
    )
    ret = robot.move_linear(target)
    if ret != 0:
        logger.warning("move_linear 指令下发失败 (ret=%d)", ret)
        return False

    time.sleep(0.1)
    start = time.time()
    while time.time() - start < timeout:
        if stop_flag_fn and stop_flag_fn():
            return False
        status = robot.get_status()
        if status == 0:
            return True
        if show_display and camera is not None:
            ok_f, frame = camera.read_frame(timeout=0)
            if frame is not None:
                small = cv2.resize(frame, (int(frame.shape[1] * disp_scale),
                                           int(frame.shape[0] * disp_scale)))
                cv2.putText(small, info_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.imshow(win, small)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                global _stop_requested
                _stop_requested = True
                return False
        else:
            time.sleep(0.05)
    logger.warning("move_with_display 超时 (%.1fs)", timeout)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='自动 ArUco 位姿采集')
    parser.add_argument('--camera', type=str, default=None,
                        help='相机名称（默认使用 cameras.yaml 中的 default_camera）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--output-dir', type=str, default='data/aruco',
                        help='输出目录')
    parser.add_argument('--start-pose', type=float, nargs=6,
                        metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        default=None,
                        help='采样中心位姿 (mm/deg)；不传则读取当前 TCP')
    parser.add_argument('--xy-span', type=float, default=DEFAULT_XY_SPAN_MM,
                        help=f'XY 方向采样半径 (mm)，默认 {DEFAULT_XY_SPAN_MM:.0f}')
    parser.add_argument('--z-span', type=float, default=DEFAULT_Z_SPAN_MM,
                        help=f'Z 方向采样半径 (mm)，默认 {DEFAULT_Z_SPAN_MM:.0f}')
    parser.add_argument('--rot-delta', type=float, default=DEFAULT_ROT_DEG,
                        help=f'旋转变体幅度 (deg)，默认 {DEFAULT_ROT_DEG:.0f}')
    parser.add_argument('--raw', action='store_true', default=True,
                        help='使用 RAW 检测模式（默认，IPPE_SQUARE，无时序滤波）')
    parser.add_argument('--no-raw', dest='raw', action='store_false',
                        help='使用 ArucoDetector（Kalman + 多方法 PnP）')
    parser.add_argument('--capture-frames', type=int, default=5,
                        help='到位后连采帧数，取 median 降噪（默认 5）')
    parser.add_argument('--reproj-threshold', type=float, default=2.0,
                        help='重投影误差阈值（像素），超过则跳过')
    parser.add_argument('--settle-time', type=float, default=0.5,
                        help='运动后稳定等待时间（秒）')
    parser.add_argument('--move-timeout', type=float, default=30.0,
                        help='单次运动超时（秒）')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示画面（SSH 无头模式）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印轨迹点，不实际运动')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dry_run(waypoints):
    """只打印轨迹点，不执行。"""
    logger.info("=== DRY RUN: %d 个轨迹点 ===", len(waypoints))
    for i, wp in enumerate(waypoints):
        x, y, z, rx, ry, rz = wp
        logger.info("[%02d] pos=(%.1f, %.1f, %.1f)mm  ori=(%.1f, %.1f, %.1f)°",
                    i + 1, x, y, z, rx, ry, rz)
    logger.info("=== DRY RUN 完成 ===")


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    global _stop_requested
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT, _signal_handler)

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    # --- 连接机械臂 ---
    robot = build_robot(robot_cfg)
    logger.info("正在连接机械臂: %s", robot_cfg.ip)
    if not robot.connect():
        logger.error("机械臂连接失败，退出")
        return
    if not hasattr(robot, 'move_and_wait'):
        logger.error("当前驱动不支持 move_and_wait()，自动采集暂不支持该驱动")
        robot.disconnect()
        return

    # --- 连接相机 ---
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    # --- 构建检测器 ---
    if args.raw:
        # Raw 模式：手动构建 cv2.aruco.ArucoDetector
        dict_name = marker_cfg.dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        raw_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        detector = None
        logger.info("检测模式: RAW (IPPE_SQUARE, 无滤波)")
    else:
        detection_cfg = cfg.get_detection()
        detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
        raw_detector = None
        logger.info("检测模式: ArucoDetector (Kalman + PnP 多方法)")

    valid_ids = set(marker_cfg.valid_ids)
    marker_sizes = marker_cfg.marker_sizes

    # --- 可视化窗口 ---
    show_display = not args.no_display
    disp_scale = 0.4
    win = None
    if show_display:
        win = "auto_collect_aruco"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    # --- 记录 home 位姿 ---
    home_pose = robot.get_tcp_pose()
    if home_pose is None:
        logger.error("无法读取当前位姿，退出")
        camera.close()
        robot.disconnect()
        return
    logger.info("当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *home_pose)

    center_pose = [float(v) for v in (args.start_pose if args.start_pose else home_pose)]
    logger.info("采样中心: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *center_pose)

    waypoints = build_aruco_waypoints(
        center_pose=center_pose,
        xy_span=args.xy_span,
        z_span=args.z_span,
        rot_delta=args.rot_delta,
    )
    logger.info("生成 %d 个 waypoints", len(waypoints))

    if args.dry_run:
        dry_run(waypoints)
        camera.close()
        robot.disconnect()
        return

    # --- IO 工具 ---
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    pose_writers = {
        mid: PoseFileWriter(os.path.join(out_dir, fname))
        for mid, fname in marker_cfg.pose_files.items()
    }
    robot_writer = PoseFileWriter(os.path.join(out_dir, 'robot_tcp.txt'))
    tcp_writers = {
        mid: PoseFileWriter(os.path.join(out_dir, f'robot_tcp_id{mid}.txt'))
        for mid in marker_cfg.valid_ids
    }
    image_saver = ImageSaver(os.path.join(out_dir, 'images'))
    saved_counts = {mid: 0 for mid in marker_cfg.valid_ids}
    first_save = True

    def do_save(aruco_result, robot_pose, raw_frame):
        """保存 ArUco 位姿 + robot TCP + 原始帧。"""
        nonlocal first_save
        if not aruco_result:
            return False
        if robot_pose is None:
            return False

        mode_w = 'w' if first_save else 'a'
        if first_save:
            first_save = False

        robot_writer._append = (mode_w == 'a')
        robot_writer.open()
        robot_writer.write(robot_pose)
        robot_writer.close()

        saved_ids = []
        for mid, data in aruco_result.items():
            if mid not in pose_writers:
                continue
            t = data['tvec_m2c'].flatten()
            e = data['euler_m2c_zyx']
            pose_writers[mid]._append = (mode_w == 'a')
            pose_writers[mid].open()
            pose_writers[mid].write([t[0], t[1], t[2], e[0], e[1], e[2]])
            pose_writers[mid].close()
            if mid in tcp_writers:
                tcp_writers[mid]._append = (mode_w == 'a')
                tcp_writers[mid].open()
                tcp_writers[mid].write(robot_pose)
                tcp_writers[mid].close()
            saved_counts[mid] += 1
            saved_ids.append(mid)

        image_saver.save(raw_frame)
        return True

    saved_total = 0
    skipped_detect = 0
    skipped_move = 0
    skipped_reproj = 0
    total = len(waypoints)

    logger.info("开始自动采集，共 %d 个轨迹点...", total)

    try:
        for i, wp in enumerate(waypoints):
            if _stop_requested:
                logger.warning("用户中断，已完成 %d/%d", i, total)
                break

            x, y, z, rx, ry, rz = wp
            logger.info("[%02d/%02d] 移动到 pos=(%.1f,%.1f,%.1f)mm ori=(%.1f,%.1f,%.1f)°",
                        i + 1, total, x, y, z, rx, ry, rz)

            # 安全检查
            current_pose = robot.get_tcp_pose()
            if current_pose is None:
                logger.warning("[%02d] 无法读取当前 TCP，跳过", i + 1)
                skipped_move += 1
                continue
            step_trans, step_rot = pose_delta(current_pose, wp)
            if step_trans > MAX_DIRECT_STEP_TRANS_MM or step_rot > MAX_DIRECT_STEP_ROT_DEG:
                logger.warning("[%02d] 单次运动过大: %.1f mm / %.1f deg，跳过",
                               i + 1, step_trans, step_rot)
                skipped_move += 1
                continue

            # 运动
            ok = move_with_display(
                robot, wp, timeout=args.move_timeout,
                camera=camera, show_display=show_display,
                win=win, disp_scale=disp_scale,
                info_text=f"[{i+1}/{total}] moving...",
                stop_flag_fn=lambda: _stop_requested,
            )
            if not ok:
                logger.warning("[%02d] 运动失败或超时，跳过", i + 1)
                skipped_move += 1
                continue

            # 稳定等待（持续取流刷新显示，丢弃运动中缓冲帧）
            t_end = time.monotonic() + args.settle_time
            frame = None
            while time.monotonic() < t_end:
                ok_frame, frame = camera.read_frame(timeout=0.1)
                if show_display and frame is not None:
                    small = cv2.resize(frame, (int(frame.shape[1] * disp_scale),
                                               int(frame.shape[0] * disp_scale)))
                    cv2.putText(small, f"[{i+1}/{total}] settling...",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    cv2.imshow(win, small)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        _stop_requested = True
                        break
            if _stop_requested:
                logger.warning("用户按 q 退出")
                break

            # 多帧采集 + median 降噪
            if args.raw:
                detect_fn = lambda g: detect_raw(g, raw_detector, valid_ids, marker_sizes, K, dist)
            else:
                detect_fn = lambda g: detector.detect(
                    cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g)

            aruco_result, frame = multi_frame_detect(
                camera, detect_fn, n_frames=args.capture_frames,
                show_display=show_display, win=win, disp_scale=disp_scale,
                info_text=f"[{i+1}/{total}]",
            )

            # 可视化
            if show_display and frame is not None:
                vis = draw_aruco_result(
                    frame, aruco_result, intrinsics,
                    robot_pose=wp,
                    robot_connected=True,
                    saved_counts=saved_counts,
                    saved_images=image_saver.count,
                )
                small = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_LINEAR)
                cv2.putText(small, f"[{i+1}/{total}] saved:{saved_total} skip:{skipped_detect+skipped_move+skipped_reproj}",
                            (10, small.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.imshow(win, small)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    _stop_requested = True
                    logger.warning("用户按 q 退出")
                    break

            if not aruco_result:
                logger.warning("[%02d] ArUco 未检测到（%d 帧均无结果），跳过",
                               i + 1, args.capture_frames)
                skipped_detect += 1
                continue

            # 重投影误差过滤：移除超阈值的 marker
            filtered_result = {}
            for mid, data in aruco_result.items():
                err = data['reproj_err']
                if err <= args.reproj_threshold:
                    filtered_result[mid] = data
                else:
                    logger.warning("[%02d] ID%d 重投影误差 %.2fpx > %.1fpx，剔除",
                                   i + 1, mid, err, args.reproj_threshold)
            if not filtered_result:
                logger.warning("[%02d] 所有 marker 重投影过大，跳过", i + 1)
                skipped_reproj += 1
                continue

            # 读取实际 TCP
            tcp = robot.get_tcp_pose()
            if tcp is None:
                logger.warning("[%02d] 机械臂位姿读取失败，跳过", i + 1)
                skipped_move += 1
                continue

            # 保存
            if do_save(filtered_result, tcp, frame.copy()):
                saved_total += 1
                detected_ids = list(filtered_result.keys())
                errs = [filtered_result[m]['reproj_err'] for m in detected_ids]
                t_first = list(filtered_result.values())[0]['tvec_m2c'].flatten()
                logger.info("[%02d] ✓ 保存 #%d  IDs=%s  reproj=%s  t=(%.1f,%.1f,%.1f)mm  tcp=(%.1f,%.1f,%.1f)",
                            i + 1, saved_total, detected_ids,
                            ['%.2f' % e for e in errs],
                            t_first[0], t_first[1], t_first[2],
                            tcp[0], tcp[1], tcp[2])

    finally:
        # 回 home
        if not _stop_requested and home_pose is not None:
            logger.info("返回 Home 位姿...")
            move_with_display(robot, home_pose, timeout=args.move_timeout)

        if show_display:
            cv2.destroyAllWindows()
        camera.close()
        robot.disconnect()

        logger.info("=" * 50)
        logger.info("采集完成统计:")
        logger.info("  总轨迹点:     %d", total)
        logger.info("  成功保存:     %d", saved_total)
        logger.info("  运动失败:     %d", skipped_move)
        logger.info("  检测失败:     %d", skipped_detect)
        logger.info("  重投影过大:   %d", skipped_reproj)
        logger.info("  每帧采集数:   %d (median)", args.capture_frames)
        logger.info("  重投影阈值:   %.1fpx", args.reproj_threshold)
        logger.info("  输出目录:     %s", args.output_dir)
        logger.info("  累计各 ID:    %s", saved_counts)
        logger.info("=" * 50)


if __name__ == '__main__':
    main()
