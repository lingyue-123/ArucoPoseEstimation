#!/usr/bin/env python3
"""
在线目标位姿估计 | Online Target Pose Estimation

实时从相机检测 ArUco，结合手眼标定结果，估算参考位置（如充电口正前方）的机械臂 TCP。

原理：
    ArUco 固定在世界中，在任意机械臂位姿下其基座坐标不变：
        T_g2b_ref = T_g2b_cur × T_c2g × T_aruco2cam_cur × T_aruco2cam_ref⁻¹ × T_c2g⁻¹

    注意：估算结果的一致性（方差）由以下两者共同决定：
        1. 手眼标定精度（T_c2g 的误差）
        2. 相机估计 ArUco 的精度（光照、角度、内参误差等）
    方差大时需分别排查，不能单独归因。

工作流：
    1. 将机械臂移到参考位置（充电口正前方），按 r 保存当前 ArUco 位姿为参考
    2. 移动机械臂到不同位置，观察屏幕上估算的目标 TCP 是否稳定
    3. 按 s 采集当前估算值（用于统计一致性），按 q 退出后打印汇总

用法：
    python scripts/estimate_target_pose.py --camera hikvision_normal

    # 加载已有参考位姿（跳过步骤 1）
    python scripts/estimate_target_pose.py \\
        --camera hikvision_normal \\
        --aruco-ref aruco_pose_ref.txt
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.aruco import ArucoDetector
from robovision.robot import build_robot
from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file, save_pose_file
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose, compute_new_tool_pose,
)
from robovision.visualization.aruco_overlay import draw_aruco_result

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(description='在线目标位姿估计')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--hand-eye', type=str, default='data/handeye/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考 ArUco 位姿文件（不存在则按 r 键现场保存）')
    parser.add_argument('--target-marker', type=int, default=1,
                        help='用于估计的 ArUco Marker ID（默认 1）')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型（覆盖 config/robot.yaml）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _put(img, text, y, color=(200, 200, 200)):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _aruco_to_matrix(data: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data['R_m2c']
    T[:3, 3] = data['tvec_m2c'].flatten()
    return T


def _print_summary(samples):
    if not samples:
        return
    arr = np.array(samples)  # (N, 6)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    pos_std_mm = float(np.linalg.norm(std[:3]))
    print("\n========== 目标 TCP 估算汇总 ==========")
    print(f"采样次数: {len(arr)}")
    print(f"均值:   X={mean[0]:.4f}  Y={mean[1]:.4f}  Z={mean[2]:.4f}"
          f"  Rx={mean[3]:.3f}°  Ry={mean[4]:.3f}°  Rz={mean[5]:.3f}°")
    print(f"标准差: X={std[0]:.4f}  Y={std[1]:.4f}  Z={std[2]:.4f}"
          f"  Rx={std[3]:.3f}°  Ry={std[4]:.3f}°  Rz={std[5]:.3f}°")
    print(f"位置一致性(std): {pos_std_mm:.2f} mm")
    print("  （反映手眼标定 + 相机 ArUco 估计的联合精度）")
    print("=======================================\n")


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    # ——— 加载手眼标定结果 ———
    T_c2g = load_hand_eye_result(args.hand_eye)
    logger.info("手眼标定加载成功: t=[%.2f, %.2f, %.2f]",
                T_c2g[0, 3], T_c2g[1, 3], T_c2g[2, 3])

    # ——— 加载参考 ArUco 位姿（如果文件已存在）———
    T_aruco2cam_ref = None
    if os.path.isfile(args.aruco_ref):
        ref_pose = load_pose_file(args.aruco_ref)[-1]
        T_aruco2cam_ref = pose_to_matrix(ref_pose)
        logger.info("参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref, *ref_pose[:3])
    else:
        logger.info("参考位姿文件不存在，将机械臂移到参考位置后按 r 键保存")

    # ——— 机械臂（先于相机，因 build_robot 可能触发 os.execv 重启进程）———
    if args.robot_ip:
        robot_cfg.ip = args.robot_ip
    robot = build_robot(robot_cfg)

    # ——— 相机与检测器 ———
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
    robot_connected = robot.connect()
    if not robot_connected:
        logger.warning("机械臂连接失败，无法读取当前 TCP，估算功能不可用")

    target_id = args.target_marker
    samples = []  # 采集的目标 TCP 估算值，用于一致性统计

    win = f"目标位姿估计 [{args.camera}] Marker ID={target_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。r=保存参考位姿  s=采集样本  q=退出")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            aruco_result = detector.detect(frame)
            target_data = aruco_result.get(target_id)

            # ——— ArUco 可视化（角点、坐标轴、位姿文字）———
            tcp = None
            if robot_connected:
                tcp = robot.get_tcp_pose()
            vis = draw_aruco_result(
                frame, aruco_result, intrinsics,
                robot_pose=tcp,
                use_kalman=False,
                robot_connected=robot_connected,
            )
            h, w = vis.shape[:2]
            y = 35 + len(aruco_result) * 100 + 80  # 跳过 draw_aruco_result 已绘制的区域

            if not target_data:
                _put(vis, f"ID{target_id} NOT DETECTED", y, (0, 0, 255))
                y += 40

            # ——— 估算目标 TCP ———
            estimated = None
            if T_aruco2cam_ref is not None and target_data is not None and robot_connected:
                if tcp is not None:
                    T_g2b_cur = pose_to_matrix(tcp)
                    T_aruco2cam_cur = _aruco_to_matrix(target_data)
                    T_g2b_ref = compute_new_tool_pose(
                        base_from_tool_old=T_g2b_cur,
                        cam_from_target_old=T_aruco2cam_cur,
                        tool_from_cam=T_c2g,
                        cam_from_target_new=T_aruco2cam_ref,
                    )
                    estimated = matrix_to_pose(T_g2b_ref)

            if estimated:
                _put(vis, "Target TCP:", y, (255, 200, 0))
                y += 28
                _put(vis,
                     f"  X={estimated[0]:.4f}  Y={estimated[1]:.4f}  Z={estimated[2]:.4f}",
                     y, (255, 200, 0))
                y += 28
                _put(vis,
                     f"  Rx={estimated[3]:.3f}°  Ry={estimated[4]:.3f}°  Rz={estimated[5]:.3f}°",
                     y, (255, 200, 0))
                y += 35
                if samples:
                    arr = np.array(samples)
                    std_mm = float(np.linalg.norm(arr[:, :3].std(axis=0)))
                    _put(vis, f"  一致性(std): {std_mm:.2f} mm  [n={len(samples)}]",
                         y, (180, 180, 180))
                    y += 28
            elif T_aruco2cam_ref is None:
                _put(vis, "Move to ref pos, press r to save ref", y, (0, 165, 255))
                y += 30

            # ——— 状态栏 ———
            ref_str = "Ref:SET" if T_aruco2cam_ref is not None else "Ref:NONE"
            robot_str = "Robot:ON" if robot_connected else "Robot:OFF"
            _put(vis, f"{ref_str} | {robot_str} | r=SetRef  s=Sample  q=Quit",
                 h - 20, (140, 140, 140))

            scale = 0.4
            disp = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                # 保存当前 ArUco 位姿为参考
                if target_data is None:
                    logger.warning("未检测到 ID%d，无法保存参考", target_id)
                    continue
                T_aruco2cam_ref = _aruco_to_matrix(target_data)
                ref_pose_vec = matrix_to_pose(T_aruco2cam_ref)
                os.makedirs(os.path.dirname(args.aruco_ref) or '.', exist_ok=True)
                save_pose_file(args.aruco_ref, np.array([ref_pose_vec]))
                samples.clear()
                logger.info("参考位姿已保存到 %s  t=[%.2f, %.2f, %.2f] mm",
                            args.aruco_ref, *ref_pose_vec[:3])

            elif key == ord('s'):
                if estimated is None:
                    logger.warning("当前无有效估算（检查参考位姿/机械臂连接/ArUco 检测）")
                    continue
                samples.append(estimated)
                logger.info("[%d] 目标 TCP: X=%.4f Y=%.4f Z=%.4f",
                            len(samples), *estimated[:3])

    finally:
        cv2.destroyAllWindows()
        camera.close()
        robot.disconnect()
        _print_summary(samples)


if __name__ == '__main__':
    main()
