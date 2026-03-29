#!/usr/bin/env python3
"""
自动手眼标定数据采集 | Automatic Hand-Eye Calibration Data Collection

围绕给定的起始 TCP 位姿自动生成一组采样位姿，每到一个位姿自动拍照检测棋盘格并保存。
替代手动示教器操作 + 按键保存的流程。

用法：
    python scripts/auto_handeye_collect.py               # 实际运行
    python scripts/auto_handeye_collect.py --dry-run      # 只打印轨迹，不运动
    python scripts/auto_handeye_collect.py --start-pose -350 750 350 -177.5 -1.6 89.5

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
from robovision.detection.chessboard import ChessboardDetector
from robovision.geometry.transforms import pose_to_matrix, matrix_to_pose
from robovision.robot import build_robot
from robovision.io.pose_file import PoseFileWriter

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# ---------------------------------------------------------------------------
# 采样策略定义（单位：毫米, 度）
# ---------------------------------------------------------------------------

DEFAULT_XY_SPAN_MM = 80.0
DEFAULT_Z_DOWN_MM = 40.0
DEFAULT_Z_UP_MM = 60.0
DEFAULT_ROT_DEG = 6.0
DEFAULT_ROT_Z_DEG = 6.0
MAX_DIRECT_STEP_TRANS_MM = 200.0
MAX_DIRECT_STEP_ROT_DEG = 20.0


def _pose_key(pose):
    return tuple(round(float(v), 4) for v in pose)


def build_waypoints(center_pose,
                    xy_span=DEFAULT_XY_SPAN_MM,
                    z_down=DEFAULT_Z_DOWN_MM,
                    z_up=DEFAULT_Z_UP_MM,
                    rot_delta=DEFAULT_ROT_DEG,
                    rot_z_delta=DEFAULT_ROT_Z_DEG):
    """围绕中心 TCP 生成采样位姿列表。"""
    cx, cy, cz, crx, cry, crz = [float(v) for v in center_pose]
    center_T = pose_to_matrix(center_pose)

    offsets = [
        (0.0, 0.0, 0.0),
        (xy_span, 0.0, 0.0),
        (-xy_span, 0.0, 0.0),
        (0.0, xy_span, 0.0),
        (0.0, -xy_span, 0.0),
        (0.0, 0.0, z_up),
        (0.0, 0.0, -z_down),
        (xy_span, xy_span, 0.0),
        (xy_span, -xy_span, 0.0),
        (-xy_span, xy_span, 0.0),
        (-xy_span, -xy_span, 0.0),
        (xy_span, 0.0, z_up * 0.5),
        (-xy_span, 0.0, z_up * 0.5),
        (0.0, xy_span, -z_down * 0.5),
        (0.0, -xy_span, -z_down * 0.5),
    ]

    orientation_variants = [
        (0.0, 0.0, 0.0),
        (rot_delta, 0.0, 0.0),
        (-rot_delta, 0.0, 0.0),
        (0.0, rot_delta, 0.0),
        (0.0, 0.0, -rot_z_delta),
    ]

    rotate_indices = {0, 1, 2, 3, 4, 7, 8, 9, 10}
    waypoints = []
    seen = set()

    for idx, (dx, dy, dz) in enumerate(offsets):
        base_pose = [cx + dx, cy + dy, cz + dz, crx, cry, crz]
        candidates = orientation_variants if idx in rotate_indices else [(0.0, 0.0, 0.0)]
        for drx, dry, drz in candidates:
            if drx == 0.0 and dry == 0.0 and drz == 0.0:
                pose = base_pose
            else:
                target_T = center_T.copy()
                target_T[:3, 3] = [cx + dx, cy + dy, cz + dz]
                rot_pose = [0.0, 0.0, 0.0, crx + drx, cry + dry, crz + drz]
                target_T[:3, :3] = pose_to_matrix(rot_pose)[:3, :3]
                pose = matrix_to_pose(target_T)
            key = _pose_key(pose)
            if key not in seen:
                seen.add(key)
                waypoints.append([float(v) for v in pose])
    return waypoints


def sort_waypoints_nearest(waypoints):
    """最近邻贪心排序，减少连续 waypoint 间的运动距离。"""
    if len(waypoints) <= 2:
        return waypoints
    remaining = list(range(1, len(waypoints)))  # 第 0 个（中心）固定为起点
    ordered = [0]
    while remaining:
        last = waypoints[ordered[-1]]
        best_idx, best_dist = None, float('inf')
        for idx in remaining:
            wp = waypoints[idx]
            dt = sum((a - b)**2 for a, b in zip(last[:3], wp[:3])) ** 0.5
            dr = sum((a - b)**2 for a, b in zip(last[3:], wp[3:])) ** 0.5
            d = dt + dr * 5.0  # 1° ≈ 5mm 权重
            if d < best_dist:
                best_dist, best_idx = d, idx
        ordered.append(best_idx)
        remaining.remove(best_idx)
    return [waypoints[i] for i in ordered]


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    logger.warning("收到中断信号，将在当前动作完成后安全停止...")


def parse_args():
    parser = argparse.ArgumentParser(description='自动手眼标定数据采集')
    parser.add_argument('--camera', type=str, default=None,
                        help='相机名称（默认使用 cameras.yaml 中的 default_camera）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--board-output', type=str, default='data/handeye/board_to_cam.txt',
                        help='棋盘格位姿输出文件')
    parser.add_argument('--robot-output', type=str, default='data/handeye/robot_tcp.txt',
                        help='机械臂位姿输出文件')
    parser.add_argument('--start-pose', type=float, nargs=6, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        default=None,
                        help='采样中心位姿，单位 mm/deg；不传则读取当前 TCP 作为中心')
    parser.add_argument('--xy-span', type=float, default=DEFAULT_XY_SPAN_MM,
                        help=f'中心点两侧的 X/Y 采样范围，默认 {DEFAULT_XY_SPAN_MM:.0f} mm')
    parser.add_argument('--z-down', type=float, default=DEFAULT_Z_DOWN_MM,
                        help=f'相对中心向下采样范围，默认 {DEFAULT_Z_DOWN_MM:.0f} mm')
    parser.add_argument('--z-up', type=float, default=DEFAULT_Z_UP_MM,
                        help=f'相对中心向上采样范围，默认 {DEFAULT_Z_UP_MM:.0f} mm')
    parser.add_argument('--rot-delta', type=float, default=DEFAULT_ROT_DEG,
                        help=f'每个位置附加的小角度旋转幅度，默认 {DEFAULT_ROT_DEG:.0f} deg')
    parser.add_argument('--rot-z-delta', type=float, default=DEFAULT_ROT_Z_DEG,
                        help=f'每个位置附加的 Rz 旋转幅度，默认 {DEFAULT_ROT_Z_DEG:.0f} deg')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印轨迹点，不实际运动')
    parser.add_argument('--settle-time', type=float, default=0.5,
                        help='运动后稳定等待时间（秒）')
    parser.add_argument('--move-timeout', type=float, default=30.0,
                        help='单次运动超时（秒）')
    parser.add_argument('--reproj-threshold', type=float, default=2.0,
                        help='重投影误差阈值（像素），超过则跳过')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示画面（SSH 无头模式）')
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


def pose_delta(cur_pose, target_pose):
    cur_T = pose_to_matrix(cur_pose)
    target_T = pose_to_matrix(target_pose)
    delta_T = np.linalg.inv(cur_T) @ target_T
    trans_mm = float(np.linalg.norm(delta_T[:3, 3]))
    from robovision.geometry.transforms import rotation_angle_deg
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

    time.sleep(0.1)  # 等运动启动
    start = time.time()
    while time.time() - start < timeout:
        if stop_flag_fn and stop_flag_fn():
            return False
        status = robot.get_status()
        if status == 0:
            return True
        # 运动中：读帧刷新显示
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


def main():
    global _stop_requested
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 注册信号处理
    signal.signal(signal.SIGINT, _signal_handler)

    cfg = get_config()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)
    chess_cfg = cfg.get_chessboard()

    # 连接机械臂
    robot = build_robot(robot_cfg)
    logger.info("正在连接机械臂: %s", robot_cfg.ip)
    if not robot.connect():
        logger.error("机械臂连接失败，退出")
        return
    if not hasattr(robot, 'move_and_wait'):
        logger.error("当前机械臂驱动不支持直接 TCP 运动接口 move_and_wait()，自动采集暂不支持该驱动")
        robot.disconnect()
        return

    # 连接相机
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    w, h = intrinsics.width, intrinsics.height
    detect_scale = 0.25 if (w * h > 2_000_000) else 1.0
    detector = ChessboardDetector.from_config(intrinsics, chess_cfg, detect_scale=detect_scale)
    logger.info("棋盘格检测缩放: %.2f (图像 %dx%d)", detect_scale, w, h)

    # 可视化窗口
    show_display = not args.no_display
    disp_scale = 0.4
    if show_display:
        win = "auto_handeye_collect"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        # 预算缩放后的内参，用于在小图上画坐标轴
        K_small = intrinsics.camera_matrix.copy()
        K_small[0, :] *= disp_scale  # fx, cx
        K_small[1, :] *= disp_scale  # fy, cy

    # 记录 home 位姿
    home_pose = robot.get_tcp_pose()
    if home_pose is None:
        logger.error("无法读取当前位姿，退出")
        camera.close()
        robot.disconnect()
        return
    logger.info("当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *home_pose)

    center_pose = [float(v) for v in (args.start_pose if args.start_pose is not None else home_pose)]
    logger.info("采样中心: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *center_pose)

    waypoints = build_waypoints(
        center_pose=center_pose,
        xy_span=args.xy_span,
        z_down=args.z_down,
        z_up=args.z_up,
        rot_delta=args.rot_delta,
        rot_z_delta=args.rot_z_delta,
    )
    waypoints = sort_waypoints_nearest(waypoints)
    logger.info("已对 %d 个 waypoints 做最近邻排序", len(waypoints))

    if args.dry_run:
        dry_run(waypoints)
        camera.close()
        robot.disconnect()
        return

    # 准备输出文件
    os.makedirs(os.path.dirname(args.board_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.robot_output), exist_ok=True)
    board_writer = PoseFileWriter(args.board_output, append=False)
    robot_writer = PoseFileWriter(args.robot_output, append=False)
    board_writer.open()
    robot_writer.open()

    saved_count = 0
    skipped_detect = 0
    skipped_reproj = 0
    skipped_move = 0
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

            ok = move_with_display(
                robot, wp, timeout=args.move_timeout,
                camera=camera, show_display=show_display,
                win=win if show_display else None, disp_scale=disp_scale,
                info_text=f"[{i+1}/{total}] moving...",
                stop_flag_fn=lambda: _stop_requested,
            )
            if not ok:
                logger.warning("[%02d] 运动失败或超时，跳过", i + 1)
                skipped_move += 1
                continue

            # 稳定等待 + 持续取流显示（丢弃运动中的缓冲帧）
            t_end = time.monotonic() + args.settle_time
            frame = None
            while time.monotonic() < t_end:
                ok_frame, frame = camera.read_frame(timeout=0.1)
                if show_display and frame is not None:
                    small = cv2.resize(frame, (int(frame.shape[1]*disp_scale), int(frame.shape[0]*disp_scale)))
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

            # 最终拍照（settle 后的新帧）
            ok_frame, frame = camera.read_frame(timeout=1.0)
            if frame is None:
                logger.warning("[%02d] 拍照失败，跳过", i + 1)
                skipped_detect += 1
                continue

            # 检测棋盘格
            result = detector.detect(frame)

            # --- 可视化（先缩放再绘制，避免在 6MP 图上操作卡顿） ---
            if show_display and frame is not None:
                small = cv2.resize(frame, (int(frame.shape[1]*disp_scale), int(frame.shape[0]*disp_scale)))
                if result is not None:
                    # 角点
                    corners_s = result['corners'] * disp_scale
                    cv2.drawChessboardCorners(small, detector.pattern_size, corners_s, True)
                    # 坐标轴（用缩放后内参）
                    axis_len = detector.square_size * 3
                    cv2.drawFrameAxes(small, K_small, intrinsics.dist_coeffs,
                                      result['rvec'], result['tvec'], axis_len)
                    # 重投影红点
                    proj_pts, _ = cv2.projectPoints(
                        detector.obj_pts,
                        result['rvec'], result['tvec'],
                        K_small, intrinsics.dist_coeffs,
                    )
                    for pt in proj_pts.reshape(-1, 2):
                        cv2.circle(small, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
                    # 文字叠加
                    t = result['tvec'].flatten()
                    e = result['euler_zyx']
                    err = result['reproj_err']
                    err_color = (0, 255, 0) if err < 1.0 else (0, 200, 255) if err < args.reproj_threshold else (0, 0, 255)
                    cv2.putText(small, f"t(mm): X={t[0]:.1f} Y={t[1]:.1f} Z={t[2]:.1f}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(small, f"Euler(ZYX): {e[0]:.2f} {e[1]:.2f} {e[2]:.2f}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(small, f"reproj={err:.2f}px {'OK' if err < args.reproj_threshold else 'TOO HIGH'}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 1)
                else:
                    cv2.putText(small, "NO CHESSBOARD", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tcp_str = f"TCP: {wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f}"
                cv2.putText(small, f"[{i+1}/{total}] saved:{saved_count} skip:{skipped_detect+skipped_reproj+skipped_move} | {tcp_str}",
                            (10, small.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.imshow(win, small)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    _stop_requested = True
                    logger.warning("用户按 q 退出")
                    break

            if result is None:
                logger.warning("[%02d] 棋盘格未检测到，跳过", i + 1)
                skipped_detect += 1
                continue

            # 检查重投影误差
            if result['reproj_err'] > args.reproj_threshold:
                logger.warning("[%02d] 重投影误差 %.2fpx > %.1fpx，跳过",
                               i + 1, result['reproj_err'], args.reproj_threshold)
                skipped_reproj += 1
                continue

            # 读取 TCP 位姿
            tcp = robot.get_tcp_pose()
            if tcp is None:
                logger.warning("[%02d] 机械臂位姿读取失败，跳过", i + 1)
                skipped_move += 1
                continue

            # 保存
            board_writer.write(result['pose_mm'])
            robot_writer.write(tcp)
            saved_count += 1

            t = result['tvec'].flatten()
            logger.info("[%02d] ✓ 保存 #%d  board_t=(%.4f,%.4f,%.4f)  reproj=%.2fpx  tcp=(%.4f,%.4f,%.4f)",
                        i + 1, saved_count,
                        t[0], t[1], t[2], result['reproj_err'],
                        tcp[0], tcp[1], tcp[2])

    finally:
        board_writer.close()
        robot_writer.close()

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
        logger.info("  成功保存:     %d", saved_count)
        logger.info("  运动失败:     %d", skipped_move)
        logger.info("  检测失败:     %d", skipped_detect)
        logger.info("  重投影过大:   %d", skipped_reproj)
        logger.info("  输出文件:     %s, %s", args.board_output, args.robot_output)
        logger.info("=" * 50)


if __name__ == '__main__':
    main()
