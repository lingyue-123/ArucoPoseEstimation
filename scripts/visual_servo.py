#!/usr/bin/env python3
"""
视觉伺服 Stage 1：按键步进 | Visual Servo Stage 1: Key-step

实时检测 ArUco，结合手眼标定，按键驱动机械臂逐步逼近参考位置。

工作流：
    1. 按 'r'：记录当前 ArUco 位姿为参考（基准位置）
    2. 手动移开机械臂
    3. 按 'm'：计算目标 TCP，执行一步运动（gain 控制步幅）
    4. 重复按 'm' 直到误差收敛，按 'q' 退出

安全机制：
    - 单次移动超 50mm 或 5° 时拒绝执行
    - gain 默认 0.3，可用 +/- 调节
    - move_and_wait 30s 超时
    - 运动中不接受新指令

用法：
    python scripts/visual_servo.py --camera hikvision_normal

    # 不连机械臂，仅打印计算结果
    python scripts/visual_servo.py --camera hikvision_normal --no-robot

    # 加载已有参考位姿
    python scripts/visual_servo.py --camera hikvision_normal \\
        --aruco-ref data/aruco/aruco_pose_ref.txt
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2

from robovision.geometry.transforms import matrix_to_pose
from robovision.servo.core import (
    DEFAULT_GAIN, GAIN_MIN, GAIN_MAX, GAIN_STEP,
    compute_step_pose, check_step_safety, execute_move,
)
from robovision.servo.osd import put_text, draw_servo_status, draw_status_bar
from robovision.servo.session import ServoSession

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)


def parse_args():
    parser = argparse.ArgumentParser(description='视觉伺服 Stage 1：按键步进')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--hand-eye', type=str, default='data/handeye/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考 ArUco 位姿文件')
    parser.add_argument('--target-marker', type=int, default=1,
                        help='ArUco Marker ID（默认 1）')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'],
                        help='机械臂驱动类型')
    parser.add_argument('--no-robot', action='store_true',
                        help='不连接机械臂，仅打印计算结果')
    parser.add_argument('--gain', type=float, default=DEFAULT_GAIN,
                        help=f'初始 gain 系数（默认 {DEFAULT_GAIN}）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    session = ServoSession(args)
    session.setup()

    gain = max(GAIN_MIN, min(GAIN_MAX, args.gain))
    moving = False
    step_history = []

    win = f"视觉伺服 [{args.camera}] ID={session.target_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。r=设参考  m=步进  +/-=调gain  q=退出")

    try:
        while True:
            frame, aruco_result, target_data, tcp, T_g2b_cur = session.read_and_detect()
            if frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            vis = session.draw_aruco(frame, aruco_result, tcp)
            h, w = vis.shape[:2]
            y = 35 + len(aruco_result) * 100 + 80

            if not target_data:
                put_text(vis, f"ID{session.target_id} NOT DETECTED", y, (0, 0, 255))
                y += 40

            # 计算目标和误差
            T_g2b_target, trans_err, rot_err = session.compute_target(target_data, tcp, T_g2b_cur)
            target_pose = matrix_to_pose(T_g2b_target) if T_g2b_target is not None else None

            if T_g2b_target is None and target_data is not None and args.no_robot:
                put_text(vis, "ArUco detected (no-robot mode)", y, (0, 200, 0))
                y += 30

            # 优先用参考 TCP 误差显示
            ref_trans_xyz, ref_trans_err, ref_rot_err = session.compute_ref_error(T_g2b_cur)
            display_trans = ref_trans_err if ref_trans_err is not None else trans_err
            display_rot = ref_rot_err if ref_rot_err is not None else rot_err

            # OSD
            y = draw_servo_status(vis, y, gain, display_trans, display_rot, target_pose,
                                  moving, step_history, ref_set=session.ref_set,
                                  trans_xyz=ref_trans_xyz)
            draw_status_bar(vis, session.ref_set, session.robot_status_str, gain,
                            "r=Ref m=Move +/-=Gain q=Quit")

            # 显示
            scale = 0.4
            disp = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                if session.set_reference(target_data, T_g2b_cur):
                    step_history.clear()

            elif key == ord('+') or key == ord('='):
                gain = min(GAIN_MAX, gain + GAIN_STEP)
                logger.info("Gain → %.2f", gain)

            elif key == ord('-'):
                gain = max(GAIN_MIN, gain - GAIN_STEP)
                logger.info("Gain → %.2f", gain)

            elif key == ord('m'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if T_g2b_target is None:
                    logger.warning("无法计算目标（检查参考位姿/ArUco检测/机械臂连接）")
                    continue

                T_step, step_trans, step_rot = compute_step_pose(T_g2b_cur, T_g2b_target, gain)
                step_pose = matrix_to_pose(T_step)

                logger.info("步进计算: 平移=%.2f mm, 旋转=%.2f deg (gain=%.2f)",
                            step_trans, step_rot, gain)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *step_pose)

                safe, reason = check_step_safety(step_trans, step_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行运动")
                    step_history.append((trans_err, rot_err))
                else:
                    moving = True
                    ok = execute_move(session.robot, step_pose)
                    moving = False
                    if ok:
                        step_history.append((trans_err, rot_err))

    finally:
        session.close()
        if step_history:
            print("\n========== 步进历史 ==========")
            print(f"{'#':>3}  {'平移误差(mm)':>12}  {'旋转误差(deg)':>13}")
            for i, (t_e, r_e) in enumerate(step_history, 1):
                print(f"{i:3d}  {t_e:12.2f}  {r_e:13.2f}")
            print("==============================\n")


if __name__ == '__main__':
    main()
