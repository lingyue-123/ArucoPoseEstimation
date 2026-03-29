#!/usr/bin/env python3
"""
视觉伺服 Stage 2：PBVS 自动伺服 | Visual Servo Stage 2: Auto PBVS

在 Stage 1（按键步进）基础上增加自动伺服循环：
按 'a' 启动后，机械臂自动迭代逼近参考位置直到收敛。

工作流：
    1. 按 'r'：记录当前 ArUco 位姿为参考
    2. 手动移开机械臂
    3. 按 'a'：启动自动伺服（自适应 gain，自动收敛）
    4. 收敛后自动停止，或按 'a' 手动停止

自动伺服循环：
    detect → 误差 < 阈值? → 停止(成功)
                 ↓ 否
    计算步进(adaptive gain) → 安全检查 → 执行运动 → 重复

安全机制：
    - 单步安全阈值（50mm / 5°）
    - 最大迭代次数（默认 50）
    - 总超时（默认 120s）
    - 连续检测丢失暂停（默认 10 帧后停止）
    - 运动失败立即停止

用法：
    python scripts/visual_servo_auto.py --camera hikvision_normal

    # dry-run 模式
    python scripts/visual_servo_auto.py --camera hikvision_normal --no-robot
"""

import argparse
import logging
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2

from robovision.geometry.transforms import matrix_to_pose
from robovision.servo.core import (
    DEFAULT_GAIN, GAIN_MIN, GAIN_MAX, GAIN_STEP,
    compute_step_pose, check_step_safety, execute_move,
    adaptive_gain,
)
from robovision.servo.osd import put_text, draw_servo_status, draw_status_bar
from robovision.servo.session import ServoSession

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# 自动伺服参数
DEFAULT_CONV_TRANS_MM = 1.0
DEFAULT_CONV_ROT_DEG = 0.5
DEFAULT_MAX_STEPS = 50
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_LOST_LIMIT = 10


def parse_args():
    parser = argparse.ArgumentParser(description='视觉伺服 Stage 2：PBVS 自动伺服')
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
                        help=f'初始 base gain（默认 {DEFAULT_GAIN}）')
    parser.add_argument('--conv-trans', type=float, default=DEFAULT_CONV_TRANS_MM,
                        help=f'收敛平移阈值 mm（默认 {DEFAULT_CONV_TRANS_MM}）')
    parser.add_argument('--conv-rot', type=float, default=DEFAULT_CONV_ROT_DEG,
                        help=f'收敛旋转阈值 deg（默认 {DEFAULT_CONV_ROT_DEG}）')
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS,
                        help=f'最大迭代步数（默认 {DEFAULT_MAX_STEPS}）')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_S,
                        help=f'总超时秒数（默认 {DEFAULT_TIMEOUT_S}）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    session = ServoSession(args)
    session.setup()

    base_gain = max(GAIN_MIN, min(GAIN_MAX, args.gain))
    gain = base_gain
    moving = False
    step_history = []

    # 自动模式状态
    auto_mode = False
    auto_step = 0
    auto_start_time = 0.0
    lost_count = 0  # 连续检测丢失帧数

    win = f"视觉伺服 Auto [{args.camera}] ID={session.target_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。r=设参考  a=自动  m=手动步进  +/-=调gain  q=退出")

    def stop_auto(reason):
        nonlocal auto_mode, auto_step, lost_count
        auto_mode = False
        logger.info("自动伺服停止: %s (共 %d 步)", reason, auto_step)
        auto_step = 0
        lost_count = 0

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
                                  auto_mode=auto_mode, auto_step=auto_step,
                                  auto_max_steps=args.max_steps, trans_xyz=ref_trans_xyz)
            keys_help = "a=Auto m=Move r=Ref +/-=Gain q=Quit"
            draw_status_bar(vis, session.ref_set, session.robot_status_str, gain,
                            keys_help, auto_mode=auto_mode)

            # 显示
            scale = 0.4
            disp = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow(win, disp)

            # === 自动伺服循环 ===
            if auto_mode and not moving:
                # 超时检查
                if time.time() - auto_start_time > args.timeout:
                    stop_auto(f"总超时 {args.timeout:.0f}s")
                elif auto_step >= args.max_steps:
                    stop_auto(f"达到最大步数 {args.max_steps}")
                elif T_g2b_target is None:
                    # 检测丢失
                    lost_count += 1
                    if lost_count >= DEFAULT_LOST_LIMIT:
                        stop_auto(f"连续 {DEFAULT_LOST_LIMIT} 帧未检测到目标")
                    else:
                        logger.debug("检测丢失 %d/%d", lost_count, DEFAULT_LOST_LIMIT)
                else:
                    lost_count = 0
                    # 收敛检查
                    if trans_err < args.conv_trans and rot_err < args.conv_rot:
                        stop_auto(f"收敛! 误差={trans_err:.2f}mm/{rot_err:.2f}deg")
                    else:
                        # 自适应 gain
                        gain = adaptive_gain(trans_err, base_gain)

                        T_step, step_trans, step_rot = compute_step_pose(
                            T_g2b_cur, T_g2b_target, gain)
                        step_pose = matrix_to_pose(T_step)

                        logger.info("[AUTO #%d] 误差=%.2f mm/%.2f deg, gain=%.2f, "
                                    "步进=%.2f mm/%.2f deg",
                                    auto_step + 1, trans_err, rot_err, gain,
                                    step_trans, step_rot)

                        safe, reason = check_step_safety(step_trans, step_rot)
                        if not safe:
                            stop_auto(f"安全检查失败: {reason}")
                        elif args.no_robot:
                            logger.info("[DRY RUN] 不执行运动")
                            step_history.append((trans_err, rot_err))
                            auto_step += 1
                        else:
                            moving = True
                            ok = execute_move(session.robot, step_pose)
                            moving = False
                            if ok:
                                step_history.append((trans_err, rot_err))
                                auto_step += 1
                            else:
                                stop_auto("运动失败")

            # === 按键处理 ===
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                if auto_mode:
                    stop_auto("用户退出")
                break

            elif key == ord('a'):
                if auto_mode:
                    stop_auto("用户手动停止")
                else:
                    if not session.ref_set:
                        logger.warning("请先按 r 设定参考位姿")
                    elif T_g2b_target is None:
                        logger.warning("无法启动自动（检查 ArUco 检测/机械臂连接）")
                    else:
                        auto_mode = True
                        auto_step = 0
                        auto_start_time = time.time()
                        lost_count = 0
                        step_history.clear()
                        logger.info("自动伺服启动 (base_gain=%.2f, 收敛=%.1fmm/%.1fdeg, "
                                    "max=%d步, timeout=%.0fs)",
                                    base_gain, args.conv_trans, args.conv_rot,
                                    args.max_steps, args.timeout)

            elif key == ord('r'):
                if auto_mode:
                    continue  # 自动模式中忽略
                if session.set_reference(target_data, T_g2b_cur):
                    step_history.clear()

            elif key == ord('+') or key == ord('='):
                base_gain = min(GAIN_MAX, base_gain + GAIN_STEP)
                gain = base_gain
                logger.info("Base Gain → %.2f", base_gain)

            elif key == ord('-'):
                base_gain = max(GAIN_MIN, base_gain - GAIN_STEP)
                gain = base_gain
                logger.info("Base Gain → %.2f", base_gain)

            elif key == ord('m'):
                if auto_mode or moving:
                    continue
                if T_g2b_target is None:
                    logger.warning("无法计算目标（检查参考位姿/ArUco检测/机械臂连接）")
                    continue

                T_step, step_trans, step_rot = compute_step_pose(
                    T_g2b_cur, T_g2b_target, gain)
                step_pose = matrix_to_pose(T_step)

                logger.info("步进计算: 平移=%.2f mm, 旋转=%.2f deg (gain=%.2f)",
                            step_trans, step_rot, gain)

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
            if auto_step > 0 and len(step_history) >= 2:
                print(f"\n初始误差: {step_history[0][0]:.2f} mm / {step_history[0][1]:.2f} deg")
                print(f"最终误差: {step_history[-1][0]:.2f} mm / {step_history[-1][1]:.2f} deg")
            print("==============================\n")


if __name__ == '__main__':
    main()
