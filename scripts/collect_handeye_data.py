#!/usr/bin/env python3
"""
手眼标定数据采集入口脚本 | Collect Hand-Eye Calibration Data

替代以下旧脚本：
- Handeyecalib/chessboard_est_hkws_cam_0225.py
- Handeyecalib/chessboard_est_usb_cam_0206.py
- Handeyecalib/chessboard_est_web_cam_0212.py
- Handeyecalib/chessboard_est.py

用法：
    python scripts/collect_handeye_data.py --camera hikvision_normal

操作键：
    s: 保存当前棋盘格位姿 + 机械臂位姿
    c: 清空已保存数据
    q/ESC: 退出
"""

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.chessboard import ChessboardDetector
from robovision.robot import build_robot
from robovision.io.pose_file import PoseFileWriter

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(description='手眼标定数据采集')
    parser.add_argument('--camera', type=str, default=None,
                        help='相机名称（默认使用 cameras.yaml 中的 default_camera）')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--board-output', type=str, default='data/handeye/board_to_cam.txt',
                        help='棋盘格位姿输出文件')
    parser.add_argument('--robot-output', type=str, default='data/handeye/robot_tcp.txt',
                        help='机械臂位姿输出文件')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)
    chess_cfg = cfg.get_chessboard()

    # 先创建机械臂（触发 SDK 路径检查，若需 os.execv 重启在此发生）
    if args.robot_ip:
        robot_cfg.ip = args.robot_ip
    robot = build_robot(robot_cfg)

    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    # 高分辨率相机（>2MP）使用 0.25 缩放粗检测，subpix 仍在全分辨率精化
    w, h = intrinsics.width, intrinsics.height
    detect_scale = 0.25 if (w * h > 2_000_000) else 1.0
    detector = ChessboardDetector.from_config(intrinsics, chess_cfg, detect_scale=detect_scale)
    logger.info("棋盘格检测缩放比例: %.2f (图像 %dx%d)", detect_scale, w, h)

    logger.info("正在连接机械臂: %s", robot_cfg.ip)
    robot_connected = robot.connect()
    if not robot_connected:
        logger.warning("机械臂连接失败，将以无机械臂模式运行")

    os.makedirs(os.path.dirname(args.board_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.robot_output), exist_ok=True)
    board_writer = PoseFileWriter(args.board_output)
    robot_writer = PoseFileWriter(args.robot_output)
    saved_count = 0
    first_save = True
    REPROJ_WARN = 1.0   # px：超过此值显示黄色
    REPROJ_BLOCK = 2.0  # px：超过此值显示红色，禁止保存

    win = f"棋盘格采集 [{args.camera}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。s=保存 c=清空 r=重连机械臂 q=退出")

    try:
        while True:
            ok, frame = camera.read_frame(timeout=0.1)
            if not ok or frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            result = detector.detect(frame)
            vis = detector.draw_result(frame, result)

            reproj_ok = False
            if result:
                t = result['tvec'].flatten()
                e = result['euler_zyx']
                err = result['reproj_err']

                # 投影点可视化：把 3D 角点用当前位姿投回图像，与检测点对比
                proj_pts, _ = cv2.projectPoints(
                    detector.obj_pts,
                    result['rvec'], result['tvec'],
                    intrinsics.camera_matrix, intrinsics.dist_coeffs,
                )
                for pt in proj_pts.reshape(-1, 2):
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)  # 红点=投影点

                # 颜色编码：绿(<1px) 黄(1~2px) 红(>2px)
                if err < REPROJ_WARN:
                    err_color = (0, 255, 0)
                    reproj_ok = True
                elif err < REPROJ_BLOCK:
                    err_color = (0, 200, 255)
                    reproj_ok = True
                else:
                    err_color = (0, 0, 255)
                    reproj_ok = False

                cv2.putText(vis, f"t(mm): X={t[0]:.4f} Y={t[1]:.4f} Z={t[2]:.4f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis, f"Euler(ZYX): {e[0]:.2f} {e[1]:.2f} {e[2]:.2f}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis, f"reproj={err:.2f}px {'OK' if reproj_ok else 'TOO HIGH'}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, err_color, 2)
            else:
                cv2.putText(vis, "棋盘格未检测到", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            img_h = vis.shape[0]
            robot_status = "已连接" if robot_connected else "未连接"
            cv2.putText(vis, f"已保存: {saved_count} 组 | 机械臂: {robot_status} | s=保存 c=清空 r=重连 q=退出",
                        (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # 缩放显示
            scale = 0.4
            disp = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                if result is None:
                    logger.warning("棋盘格未检测到，跳过")
                    continue
                if not reproj_ok:
                    logger.warning("重投影误差 %.2fpx 过大（阈值 %.1fpx），跳过",
                                   result['reproj_err'], REPROJ_BLOCK)
                    continue
                tcp = robot.get_tcp_pose() if robot_connected else None
                if tcp is None and robot_connected:
                    logger.warning("机械臂位姿无效，跳过")
                    continue
                mode = 'w' if first_save else 'a'
                board_writer._append = (mode == 'a')
                board_writer.open()
                board_writer.write(result['pose_mm'])
                board_writer.close()
                if tcp is not None:
                    robot_writer._append = (mode == 'a')
                    robot_writer.open()
                    robot_writer.write(tcp)
                    robot_writer.close()
                saved_count += 1
                first_save = False
                t = result['tvec'].flatten()
                e = result['euler_zyx']
                logger.info("[%d] 棋盘格 t=%.4f %.4f %.4f e=%.2f %.2f %.2f",
                            saved_count, t[0], t[1], t[2], e[0], e[1], e[2])
                if tcp:
                    logger.info("[%d] TCP X=%.4f Y=%.4f Z=%.4f", saved_count, *tcp[:3])
            elif key == ord('c'):
                board_writer.clear()
                robot_writer.clear()
                saved_count = 0
                first_save = True
                logger.info("已清空数据")
            elif key == ord('r'):
                logger.info("正在重新连接机械臂...")
                robot.disconnect()
                robot_connected = robot.connect()
                logger.info("机械臂重连%s", "成功" if robot_connected else "失败")
    finally:
        cv2.destroyAllWindows()
        camera.close()
        robot.disconnect()
        logger.info("退出，共保存 %d 组数据", saved_count)


if __name__ == '__main__':
    main()
