#!/usr/bin/env python3
"""
Demo: 沿法兰 Z 轴直线运动 + 异步夹爪开合

按 a: 机械臂沿工具 Z 轴前进 50mm（非阻塞），同时夹爪异步张开
按 v: 机械臂沿工具 Z 轴后退 50mm（非阻塞），同时夹爪异步闭合

用法: python scripts/demo_async_gripper.py --camera mecheye
      python scripts/demo_async_gripper.py --no-robot
"""

import argparse
import logging
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.geometry.transforms import offset_pose_along_tool_axis

from third_party.force_control_crp import BridgeCRobotAdapter
from gripper_controller import GripperController

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, y, color=(200, 200, 200)):
    cv2.putText(img, text, (20, y), _FONT, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (20, y), _FONT, 0.5, color, 1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Async Arm Z-axis + Gripper Demo')
    p.add_argument('--camera', type=str, default=None, help='相机名称')
    p.add_argument('--robot-ip', type=str, default=None, help='机械臂 IP')
    p.add_argument('--no-robot', action='store_true', help='dry-run')
    p.add_argument('--speed', type=int, default=30, help='运动速度 %%')
    p.add_argument('--z-mm', type=float, default=50, help='Z 轴移动距离 mm')
    return p.parse_args(argv)


def main():
    args = parse_args()
    cfg = get_config()
    robot_cfg = cfg.get_robot()

    # ---- 连接 ----
    robot, robot_ok = None, False
    if not args.no_robot:
        ip = args.robot_ip or getattr(robot_cfg, 'ip', None) or '192.168.1.12'
        so = getattr(robot_cfg, 'so_path', 'third_party/crp_robot_sdk/libRobotService.so')
        if not os.path.isabs(so):
            so = os.path.join(_ROOT, so)
        bridge = BridgeCRobotAdapter(ip=ip, so_path=so)
        robot_ok = bridge.connect()
        if robot_ok:
            robot = bridge
            robot.set_speed(args.speed)
            logger.info("Robot connected: %s", ip)

    gripper = GripperController(port='/dev/ttysWK3', baudrate=115200, slave_id=4)
    gripper_ok = not args.no_robot and gripper.connect()
    if gripper_ok:
        logger.info("Gripper connected (heartbeat active)")

    # ---- 相机 ----
    camera = None
    if args.camera:
        camera = build_camera(args.camera, cfg)
        camera.open()

    # ---- 异步状态 ----
    grp_pending = False       # 夹爪动作进行中
    grp_deadline = 0.0        # 夹爪超时时刻

    win = "Async Demo [a=advance v=retract q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    logger.info("a=前进+%dmm并打开夹爪  v=后退%dmm并闭合夹爪  q=退出", args.z_mm, args.z_mm)

    try:
        while True:
            vis = None
            if camera is not None:
                ok, frame = camera.read_frame()
                if ok and frame is not None:
                    vis = cv2.resize(frame, (960, 540)) if max(frame.shape) > 960 else frame.copy()
            if vis is None:
                vis = np.zeros((540, 960, 3), dtype=np.uint8)

            y = 25

            # ---- 检查夹爪是否完成 ----
            if grp_pending:
                gs = gripper.latest_grip_status
                if gs in (1, 2) or time.time() > grp_deadline:
                    if time.time() > grp_deadline:
                        logger.warning("Gripper timeout (status=%s)", gs)
                    else:
                        logger.info("Gripper complete (status=%d)", gs)
                    grp_pending = False

            # ---- 显示 ----
            is_moving_flag = robot.is_moving() if robot_ok else False
            put_text(vis, f"Robot:{'ON' if robot_ok else 'OFF'}  Gripper:{'ON' if gripper_ok else 'OFF'}", y)
            y += 30

            if robot_ok:
                tcp = robot.get_tcp_pose()
                if tcp:
                    put_text(vis, f"TCP: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}", y, (200, 200, 200))
                    y += 30

            gs = gripper.latest_grip_status if gripper_ok else None
            gs_map = {0: "MOVING", 1: "REACHED", 2: "GRASPED", 3: "DROPPED"}
            put_text(vis, f"Gripper: {gs_map.get(gs, str(gs)) if gs is not None else 'N/A'}", y)
            y += 30

            if is_moving_flag:
                put_text(vis, "ARM: MOVING...", y, (0, 100, 255))
                y += 30
            if grp_pending:
                put_text(vis, "GRIPPER: MOVING...", y, (0, 200, 255))
                y += 30

            put_text(vis, "a=Advance+Open  v=Retract+Close  q=Quit", y, (120, 120, 120))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key in (ord('a'), ord('v')):
                if not robot_ok:
                    logger.warning("Robot not connected")
                    continue
                if not gripper_ok:
                    logger.warning("Gripper not connected")
                    continue

                direction = 1 if key == ord('a') else -1
                tcp = robot.get_tcp_pose()
                if tcp is None:
                    logger.warning("Failed to read TCP")
                    continue

                target = offset_pose_along_tool_axis(
                    list(tcp),
                    float(args.z_mm * direction),
                    axis='z'
                )
                from crobot_driver_interface import CartesianPose
                cart = CartesianPose(*target)

                # 机械臂: 非阻塞运动（end=False，发送后立即返回）
                ok = robot.move_linear(cart, speed=args.speed, start=True, end=False)
                if ok:
                    logger.info("Arm: moving Z=%+.1fmm (non-blocking)", args.z_mm * direction)

                # 夹爪: 异步开/合（写入寄存器后立即返回，心跳线程持续更新状态）
                if direction == 1:
                    gripper.set_speed(30)
                    gripper.set_position(0)   # 张开
                    logger.info("Gripper: open started (non-blocking)")
                else:
                    gripper.set_speed(30)
                    gripper.set_force(75)
                    gripper.set_position(32)  # 闭合
                    logger.info("Gripper: close started (non-blocking)")

                grp_pending = True
                grp_deadline = time.time() + 5.0
                logger.info(">>> ARM + GRIPPER 并发执行中")

    finally:
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()
        if gripper_ok:
            gripper.disconnect()
        if robot_ok:
            robot.disconnect()


if __name__ == '__main__':
    main()
