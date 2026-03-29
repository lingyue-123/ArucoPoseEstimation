#!/usr/bin/env python3
"""
ArUco 位姿采集入口脚本 | Collect ArUco Poses

替代以下所有旧脚本：
- aruco_pose_from_hkws_cam_multi_aruco.py
- aruco_pose_from_hkws_cam_multi_aruco_without_jkrc.py
- aruco_pose_from_hkws_cam_multi_aruco_black_square_without_jkrc.py
- aruco_pose_from_web_cam_multi_aruco.py
- aruco_pose_from_mech_image_folder_multi_aruco.py
- aruco_pose_from_mech_image_folder.py
- aruco_pose_from_mech_image.py

用法：
    # 海康相机 + 机械臂
    python scripts/collect_aruco_poses.py --camera hikvision_normal --robot

    # RTSP 网络相机，不连接机械臂
    python scripts/collect_aruco_poses.py --camera rtsp_webcam --no-robot

    # Mech-Eye 图片文件夹（离线模式）
    python scripts/collect_aruco_poses.py --camera mecheye --image-folder ./mech_image

操作键：
    s: 保存当前帧位姿和图片
    c: 清空所有已保存的位姿文件和图片
    r: 重新连接机械臂
    p: 打印当前位姿
    q/ESC: 退出
"""

import argparse
import logging
import os
import sys

# 确保 robovision 包可以被导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.cameras.mecheye import MechEyeImageFolderCamera
from robovision.cameras.base import CameraIntrinsics
from robovision.config.loader import get_config
from robovision.detection.aruco import ArucoDetector
from robovision.robot import build_robot
from robovision.io.pose_file import PoseFileWriter
from robovision.io.image_saver import ImageSaver
from robovision.visualization.aruco_overlay import draw_aruco_result

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ARM64 优化
cv2.setNumThreads(1)
cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(description='ArUco 位姿采集')
    parser.add_argument('--camera', type=str, default=None,
                        help='相机名称（默认使用 cameras.yaml 中的 default_camera）')
    parser.add_argument('--robot', action='store_true', default=True,
                        help='连接机械臂（默认）')
    parser.add_argument('--no-robot', dest='robot', action='store_false',
                        help='不连接机械臂')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--image-folder', type=str, default=None,
                        help='Mech-Eye 离线图片文件夹路径（设置后以文件夹模式运行）')
    parser.add_argument('--output-dir', type=str, default='data/aruco',
                        help='输出目录（位姿文件和图片的根目录）')
    parser.add_argument('--no-kalman', action='store_true',
                        help='关闭角点卡尔曼滤波')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)
    if args.robot_ip:
        robot_cfg.ip = args.robot_ip

    # ——— 构建相机 ———
    if args.image_folder:
        logger.info("Mech-Eye 离线图片文件夹模式: %s", args.image_folder)
        cam_cfg_data = cfg.get_camera(args.camera)
        intrinsics = CameraIntrinsics.from_config(cam_cfg_data.intrinsics, name=args.camera)
        camera = MechEyeImageFolderCamera(intrinsics=intrinsics, folder=args.image_folder)
    else:
        camera = build_camera(args.camera, cfg)

    # ——— 构建机械臂接口（先创建以触发 SDK 路径检查，避免 os.execv 在 camera.open 后重启）———
    robot = None
    robot_connected = False
    if args.robot:
        robot = build_robot(robot_cfg)

    # ——— 构建 ArUco 检测器 ———
    camera.open()
    intrinsics = camera.get_intrinsics()
    detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
    if args.no_kalman:
        detector._use_kalman = False

    if robot is not None:
        logger.info("正在连接机械臂: %s", robot_cfg.ip)
        robot_connected = robot.connect()
        if not robot_connected:
            logger.warning("机械臂连接失败，以无机械臂模式运行")

    # ——— 构建 IO 工具 ———
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

    # ——— 离线文件夹模式 ———
    if args.image_folder:
        detector._use_kalman = False
        detector._use_smoother = False
        logger.info("离线模式：已自动禁用卡尔曼滤波和位姿平滑")
        logger.info("开始处理图片文件夹...")
        vis_saver = ImageSaver(os.path.join(out_dir, 'vis'))
        idx = 0
        while True:
            ok, frame = camera.read_frame()
            if not ok:
                break
            raw_frame = frame.copy()
            aruco_result = detector.detect(frame)
            robot_pose = robot.get_tcp_pose() if robot_connected else None

            # 可视化 + 保存
            vis = draw_aruco_result(
                frame, aruco_result, intrinsics,
                robot_pose=robot_pose,
                robot_connected=robot_connected,
                saved_counts=saved_counts,
                saved_images=image_saver.count,
            )
            vis_saver.save(vis)

            do_save(aruco_result, robot_pose, raw_frame)
            idx += 1
            logger.info("已处理 %d 帧", idx)
        logger.info("文件夹处理完成，共 %d 帧，可视化图片保存至 %s", idx, vis_saver.output_dir)
        camera.close()
        if robot:
            robot.disconnect()
        return

    # ——— 实时模式 ———
    win = f"ArUco 采集 [{args.camera}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    logger.info("已就绪。操作: s=保存 c=清空 r=重连机械臂 p=打印 q=退出")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            aruco_result = detector.detect(frame)
            robot_pose = robot.get_tcp_pose() if robot_connected and robot else None

            vis = draw_aruco_result(
                frame, aruco_result, intrinsics,
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
                        print(f"=== ID {mid} {data.get('marker_length',100.0)/10:.1f}cm | "
                              f"{data.get('method','')} reproj={data['reproj_err']:.2f}px")
                        print(f"  t(mm): X={t[0]:.2f} Y={t[1]:.2f} Z={t[2]:.2f}")
                        print(f"  Euler(ZYX°): {e[0]:.2f} {e[1]:.2f} {e[2]:.2f}")
                    if robot_pose:
                        print(f"=== TCP: X={robot_pose[0]:.2f} Y={robot_pose[1]:.2f} "
                              f"Z={robot_pose[2]:.2f} | rx={robot_pose[3]:.2f} "
                              f"ry={robot_pose[4]:.2f} rz={robot_pose[5]:.2f}")
    finally:
        cv2.destroyAllWindows()
        camera.close()
        if robot:
            robot.disconnect()
        logger.info("退出完成")


if __name__ == '__main__':
    main()
