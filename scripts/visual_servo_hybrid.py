#!/usr/bin/env python3
"""
视觉伺服 Stage 3：Hybrid Servo | PBVS Approach + Fine Align State Machine

渐进式设计：
1. 远距离用 PBVS 做粗逼近
2. 进入近距离后切到精对齐状态
3. 当前先实现 pbvs-fine，对齐状态的控制接口为后续 IBVS 预留
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

from robovision.servo.core import DEFAULT_GAIN, GAIN_MIN, GAIN_MAX, adaptive_gain, check_step_safety, execute_move
from robovision.servo.ibvs import compute_ibvs_step, ibvs_cam_vel_to_step_pose
from robovision.servo.runner import build_frame_state, render_frame, run_pbvs_step
from robovision.servo.session import ServoSession

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

DEFAULT_APPROACH_TRANS_MM = 15.0
DEFAULT_APPROACH_ROT_DEG = 2.0
DEFAULT_FINE_TRANS_MM = 2.0
DEFAULT_FINE_ROT_DEG = 0.5
DEFAULT_MAX_STEPS = 80
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_LOST_LIMIT = 10


def parse_args():
    parser = argparse.ArgumentParser(description='视觉伺服 Stage 3：Hybrid Servo')
    parser.add_argument('--camera', type=str, default=None, help='cameras.yaml 中的相机名称')
    parser.add_argument('--hand-eye', type=str, default='data/handeye/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考 ArUco 位姿文件')
    parser.add_argument('--target-marker', type=int, default=1, help='ArUco Marker ID（默认 1）')
    parser.add_argument('--robot-ip', type=str, default=None, help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp'], help='机械臂驱动类型')
    parser.add_argument('--no-robot', action='store_true', help='不连接机械臂，仅打印计算结果')
    parser.add_argument('--gain', type=float, default=DEFAULT_GAIN, help=f'PBVS 粗逼近 base gain（默认 {DEFAULT_GAIN}）')
    parser.add_argument('--fine-gain', type=float, default=0.12, help='精对齐阶段 gain（默认 0.12）')
    parser.add_argument('--approach-trans', type=float, default=DEFAULT_APPROACH_TRANS_MM,
                        help=f'进入精对齐的平移阈值 mm（默认 {DEFAULT_APPROACH_TRANS_MM}）')
    parser.add_argument('--approach-rot', type=float, default=DEFAULT_APPROACH_ROT_DEG,
                        help=f'进入精对齐的旋转阈值 deg（默认 {DEFAULT_APPROACH_ROT_DEG}）')
    parser.add_argument('--fine-trans', type=float, default=DEFAULT_FINE_TRANS_MM,
                        help=f'最终完成的平移阈值 mm（默认 {DEFAULT_FINE_TRANS_MM}）')
    parser.add_argument('--fine-rot', type=float, default=DEFAULT_FINE_ROT_DEG,
                        help=f'最终完成的旋转阈值 deg（默认 {DEFAULT_FINE_ROT_DEG}）')
    parser.add_argument('--align-mode', choices=['pbvs-fine', 'ibvs-corners'], default='pbvs-fine',
                        help='精对齐阶段控制器；当前先稳定支持 pbvs-fine')
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

    approach_gain = max(GAIN_MIN, min(GAIN_MAX, args.gain))
    fine_gain = max(GAIN_MIN, min(GAIN_MAX, args.fine_gain))
    gain = approach_gain

    auto_mode = False
    mode = 'SEARCH'
    moving = False
    step_history = []
    auto_step = 0
    auto_start_time = 0.0
    lost_count = 0
    win = f"视觉伺服 Hybrid [{args.camera}] ID={session.target_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。r=设参考  a=自动  m=手动一步  q=退出")

    def stop_auto(reason):
        nonlocal auto_mode, mode, auto_step, lost_count
        auto_mode = False
        mode = 'SEARCH'
        logger.info("Hybrid 停止: %s (共 %d 步)", reason, auto_step)
        auto_step = 0
        lost_count = 0

    try:
        while True:
            state = build_frame_state(session)
            if state.frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            keys = f"a=Auto m=Move r=Ref q=Quit | Mode={mode}"
            disp = render_frame(session, state, gain, moving, step_history, keys,
                                auto_mode=auto_mode, auto_step=auto_step, auto_max_steps=args.max_steps)
            cv2.imshow(win, disp)

            if auto_mode and not moving:
                if time.time() - auto_start_time > args.timeout:
                    stop_auto(f"总超时 {args.timeout:.0f}s")
                elif auto_step >= args.max_steps:
                    stop_auto(f"达到最大步数 {args.max_steps}")
                elif state.T_g2b_target is None:
                    lost_count += 1
                    if lost_count >= DEFAULT_LOST_LIMIT:
                        stop_auto(f"连续 {DEFAULT_LOST_LIMIT} 帧未检测到目标")
                elif not state.quality['allow_auto_move']:
                    lost_count += 1
                    if lost_count >= DEFAULT_LOST_LIMIT:
                        stop_auto(f"目标质量不足，连续 {DEFAULT_LOST_LIMIT} 帧未满足自动条件")
                else:
                    lost_count = 0
                    if mode == 'SEARCH':
                        mode = 'PBVS_APPROACH'

                    if mode == 'PBVS_APPROACH':
                        gain = adaptive_gain(state.trans_err, approach_gain)
                        if state.trans_err <= args.approach_trans and state.rot_err <= args.approach_rot:
                            mode = 'ALIGN'
                            logger.info("进入精对齐阶段: %.2fmm / %.2fdeg", state.trans_err, state.rot_err)
                            continue

                    elif mode == 'ALIGN':
                        gain = fine_gain
                        if state.trans_err <= args.fine_trans and state.rot_err <= args.fine_rot:
                            stop_auto(f"完成! 误差={state.trans_err:.2f}mm/{state.rot_err:.2f}deg")
                            continue
                        if args.align_mode == 'ibvs-corners':
                            ibvs_result = compute_ibvs_step(
                                current_corners=state.target_data.get('filtered_corners'),
                                ref_corners=session.ref_corners,
                                camera_matrix=session.intrinsics.camera_matrix,
                                depth_mm=float(state.target_data['tvec_m2c'].flatten()[2]),
                                gain=gain,
                            )
                            if not ibvs_result['ok']:
                                stop_auto(ibvs_result['reason'])
                                continue
                            mapped = ibvs_cam_vel_to_step_pose(
                                state.T_g2b_cur, session.T_c2g, ibvs_result['cam_vel'])
                            safe, reason = check_step_safety(mapped['step_trans_mm'], mapped['step_rot_deg'])
                            if not safe:
                                stop_auto(reason)
                                continue
                            if args.no_robot:
                                logger.info("[DRY RUN][IBVS] err=%.2fpx step=(%.2fmm, %.2fdeg)",
                                            ibvs_result['error_norm_px'],
                                            mapped['step_trans_mm'], mapped['step_rot_deg'])
                                step_history.append((state.trans_err, state.rot_err))
                                auto_step += 1
                                continue
                            moving = True
                            ok = execute_move(session.robot, mapped['step_pose'])
                            moving = False
                            if ok:
                                step_history.append((state.trans_err, state.rot_err))
                                auto_step += 1
                            else:
                                stop_auto("IBVS 运动失败")
                            continue

                    moving = True
                    ok, reason, trans_err, rot_err = run_pbvs_step(
                        session, state, gain, no_robot=args.no_robot)
                    moving = False
                    if ok:
                        if trans_err is not None and rot_err is not None:
                            step_history.append((trans_err, rot_err))
                        auto_step += 1
                    else:
                        stop_auto(reason)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                if auto_mode:
                    stop_auto("用户退出")
                break
            elif key == ord('r'):
                if not auto_mode and session.set_reference(state.target_data):
                    step_history.clear()
                    mode = 'SEARCH'
            elif key == ord('a'):
                if auto_mode:
                    stop_auto("用户手动停止")
                else:
                    if not session.ref_set:
                        logger.warning("请先按 r 设定参考位姿")
                    elif state.T_g2b_target is None:
                        logger.warning("无法启动 Hybrid（检查 ArUco 检测/机械臂连接）")
                    elif not state.quality['allow_auto_move']:
                        logger.warning("当前目标质量不允许启动 Hybrid: %s", state.quality['reason'])
                    elif args.align_mode == 'ibvs-corners' and session.ref_corners is None:
                        logger.warning("当前未保存参考角点，请重新按 r 记录参考")
                    else:
                        auto_mode = True
                        mode = 'PBVS_APPROACH'
                        auto_step = 0
                        auto_start_time = time.time()
                        lost_count = 0
                        step_history.clear()
                        logger.info("Hybrid 启动: approach=PBVS, align=%s", args.align_mode)
            elif key == ord('m'):
                if auto_mode or moving:
                    continue
                moving = True
                ok, reason, trans_err, rot_err = run_pbvs_step(
                    session, state, gain, no_robot=args.no_robot)
                moving = False
                if not ok:
                    logger.warning(reason)
                elif trans_err is not None and rot_err is not None:
                    step_history.append((trans_err, rot_err))

    finally:
        session.close()


if __name__ == '__main__':
    main()
