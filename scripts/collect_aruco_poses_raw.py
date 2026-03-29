#!/usr/bin/env python3
"""
最简 ArUco 位姿采集脚本 | Raw ArUco Pose Collection (No Filtering)

与 collect_aruco_poses.py 功能一致，但移除时序滤波管线：
- 无 CLAHE / 形态学预处理
- 无 KalmanCornerFilter 角点滤波
- 无 MarkerPoseSmoother (SLERP/EMA)
- 无多方法 PnP 选优
- 无 extrinsic guess
- 显示值用 median(N=3) 平滑，保存数据保持 raw

检测管线：detectMarkers → cornerSubPix → solvePnP(IPPE_SQUARE) → 显示/保存

用法：
    # 海康相机 + 机械臂（默认）
    python scripts/collect_aruco_poses_raw.py --camera hikvision_normal

    # 不连接机械臂
    python scripts/collect_aruco_poses_raw.py --camera hikvision_normal --no-robot

操作键：
    s: 保存当前帧位姿和图片
    c: 清空所有已保存的位姿文件和图片
    r: 重新连接机械臂
    p: 打印当前位姿
    q/ESC: 退出
"""

import argparse
import copy
import logging
import os
import sys
from collections import deque

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.geometry.transforms import rotmat_to_euler
from robovision.robot import build_robot
from robovision.io.pose_file import PoseFileWriter
from robovision.io.image_saver import ImageSaver
from robovision.visualization.aruco_overlay import draw_aruco_result

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)


class DisplayBuffer:
    """Per-marker display-only median buffer for OSD smoothing."""

    def __init__(self, size=3):
        self.euler_buf = deque(maxlen=size)
        self.trans_buf = deque(maxlen=size)

    def update(self, euler, tvec):
        self.euler_buf.append(euler)
        self.trans_buf.append(tvec.flatten())

    def smoothed(self):
        return (np.median(self.euler_buf, axis=0),
                np.median(self.trans_buf, axis=0))


def parse_args():
    parser = argparse.ArgumentParser(description='最简 ArUco 位姿采集（无滤波）')
    parser.add_argument('--camera', type=str, default=None,
                        help='相机名称（默认使用 cameras.yaml 中的 default_camera）')
    parser.add_argument('--robot', action='store_true', default=True,
                        help='连接机械臂（默认）')
    parser.add_argument('--no-robot', dest='robot', action='store_false',
                        help='不连接机械臂')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--output-dir', type=str, default='data/aruco',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    return parser.parse_args()


def detect_raw(gray, aruco_detector, valid_ids, marker_sizes, K, dist):
    """检测：detectMarkers → cornerSubPix → solvePnP，无预处理。"""
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

        # 亚像素精化
        c = corners[i].reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray_blur, c, (5, 5), (-1, -1), criteria)
        img_pts = c.reshape(4, 2).astype(np.float64)

        # PnP (IPPE_SQUARE — 方形 marker 最优解)
        marker_len = marker_sizes.get(mid, 0.10)
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

        R, _ = cv2.Rodrigues(rvec)
        euler_zyx = rotmat_to_euler(R, order='ZYX')
        result[mid] = {
            'raw_corners': corners[i].reshape(4, 2),
            'filtered_corners': img_pts,
            'rvec_m2c': rvec, 'tvec_m2c': tvec, 'R_m2c': R,
            'euler_m2c_zyx': euler_zyx,
            'reproj_err': 0.0,
            'method': 'IPPE_SQ', 'status': 'OK',
            'marker_length': marker_len,
        }
    return result


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    # ——— 构建相机 ———
    camera = build_camera(args.camera, cfg)

    # ——— 构建机械臂 ———
    robot = None
    robot_connected = False
    if args.robot:
        robot = build_robot(robot_cfg)

    # ——— 打开相机，获取内参 ———
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    # ——— 构建 ArUco 检测器（仅用其内部的 cv2.aruco.ArucoDetector）———
    dict_name = marker_cfg.dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    valid_ids = set(marker_cfg.valid_ids)
    marker_sizes = marker_cfg.marker_sizes

    if robot is not None:
        logger.info("正在连接机械臂: %s", robot_cfg.ip)
        robot_connected = robot.connect()
        if not robot_connected:
            logger.warning("机械臂连接失败，以无机械臂模式运行")

    # ——— IO 工具 ———
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
        nonlocal first_save
        if not aruco_result:
            logger.warning("未检测到 ArUco，无法保存")
            return
        if args.robot and robot_pose is None:
            logger.warning("机械臂位姿无效，无法保存")
            return

        mode_w = 'w' if first_save else 'a'
        if first_save:
            first_save = False

        if robot_pose is not None:
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
            if robot_pose is not None and mid in tcp_writers:
                tcp_writers[mid]._append = (mode_w == 'a')
                tcp_writers[mid].open()
                tcp_writers[mid].write(robot_pose)
                tcp_writers[mid].close()
            saved_counts[mid] += 1
            saved_ids.append(mid)
            size_cm = data.get('marker_length', 0.10) * 100
            logger.info("ArUco ID%d(%.1fcm) M->C: X=%.4f Y=%.4f Z=%.4f rx=%.2f ry=%.2f rz=%.2f",
                        mid, size_cm, t[0], t[1], t[2], e[0], e[1], e[2])

        image_saver.save(raw_frame)
        if robot_pose:
            logger.info("TCP: X=%.4f Y=%.4f Z=%.4f rx=%.2f ry=%.2f rz=%.2f",
                        *robot_pose)
        logger.info("已保存 IDs: %s | 累计: %s | 图片: %d",
                    saved_ids, saved_counts, image_saver.count)

    def do_clear():
        nonlocal first_save, saved_counts
        for pw in pose_writers.values():
            pw.clear()
        robot_writer.clear()
        for tw in tcp_writers.values():
            tw.clear()
        image_saver.clear()
        saved_counts = {mid: 0 for mid in marker_cfg.valid_ids}
        first_save = True
        logger.info("已清空所有位姿文件和图片")

    # ——— 实时模式 ———
    cam_name = args.camera or 'default'
    win = f"ArUco RAW [{cam_name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    display_bufs = {}  # {marker_id: DisplayBuffer}

    logger.info("已就绪（RAW 模式，无滤波，显示 median 平滑）。操作: s=保存 c=清空 r=重连机械臂 p=打印 q=退出")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            aruco_result = detect_raw(gray, aruco_detector, valid_ids, marker_sizes, K, dist)
            robot_pose = robot.get_tcp_pose() if robot_connected and robot else None

            # --- Display-only median smoothing ---
            # Clear buffers for markers no longer detected
            for mid in list(display_bufs):
                if mid not in aruco_result:
                    del display_bufs[mid]

            # Build smoothed copy for display
            display_result = copy.deepcopy(aruco_result)
            for mid, data in aruco_result.items():
                if mid not in display_bufs:
                    display_bufs[mid] = DisplayBuffer(size=3)
                display_bufs[mid].update(data['euler_m2c_zyx'], data['tvec_m2c'])
                sm_euler, sm_tvec = display_bufs[mid].smoothed()
                display_result[mid]['euler_m2c_zyx'] = sm_euler
                display_result[mid]['tvec_m2c'] = sm_tvec.reshape(3, 1)

            vis = draw_aruco_result(
                frame, display_result, intrinsics,
                robot_pose=robot_pose,
                robot_connected=robot_connected,
                saved_counts=saved_counts,
                saved_images=image_saver.count,
            )
            cv2.imshow(win, cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_LINEAR))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                do_save(aruco_result, robot_pose, frame.copy())
            elif key == ord('c'):
                do_clear()
            elif key == ord('r'):
                if robot:
                    logger.info("正在重新连接机械臂...")
                    robot.disconnect()
                    robot_connected = robot.connect()
            elif key == ord('p'):
                if not aruco_result:
                    logger.info("未检测到 ArUco")
                else:
                    for mid, data in aruco_result.items():
                        t = data['tvec_m2c'].flatten()
                        e = data['euler_m2c_zyx']
                        print(f"=== ID {mid} {data['marker_length']*100:.1f}cm | "
                              f"IPPE_SQ status=OK")
                        print(f"  t(m): X={t[0]:.4f} Y={t[1]:.4f} Z={t[2]:.4f}")
                        print(f"  Euler(ZYX°): {e[0]:.2f} {e[1]:.2f} {e[2]:.2f}")
                    if robot_pose:
                        print(f"=== TCP: X={robot_pose[0]:.4f} Y={robot_pose[1]:.4f} "
                              f"Z={robot_pose[2]:.4f} | rx={robot_pose[3]:.2f} "
                              f"ry={robot_pose[4]:.2f} rz={robot_pose[5]:.2f}")
    finally:
        cv2.destroyAllWindows()
        camera.close()
        if robot:
            robot.disconnect()
        logger.info("退出完成")


if __name__ == '__main__':
    main()
