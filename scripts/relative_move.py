#!/usr/bin/env python3
"""
相对位移复制工具 | Relative Offset Copier

记录两个TCP位置的相对位移，然后从任意位置应用相同的位移。

工作流：
    1. 按 '1'：记录并保存起点位置
    2. 按 '2'：记录并保存终点位置
    3. 按 's'：计算相对位移 delta = end - start 并保存
    4. 移动到任意位置
    5. 按 'a'：应用相对位移，移动到 当前位置 + delta
    6. 按 'l'：加载之前保存的数据

用法：
    python scripts/relative_move.py
    python scripts/relative_move.py --robot-ip 192.168.1.133
    python scripts/relative_move.py --no-robot
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

from robovision.config.loader import get_config
from robovision.robot import build_robot
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_MAX_TRANS_MM = 500.0
DEFAULT_MAX_ROT_DEG = 90.0
DEFAULT_SPEED = 4
MOVE_TIMEOUT = 30.0
BASE_TOOL_ID = 0

_OFFSET_FILE = "data/relative_offset.txt"
_POSES_FILE = "data/relative_poses.txt"  # 保存 pos1 和 pos2

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1


def put_text(img, text, y, color=(200, 200, 200)):
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def put_text_center(img, text, y, color=(200, 200, 200)):
    """在图像中心显示文本"""
    w = img.shape[1]
    text_size = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICK)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def tcp_pose_to_mm(tcp_pose, robot_cfg):
    """把驱动返回的 TCP 位姿转换为本脚本内部统一使用的 mm/deg。"""
    if tcp_pose is None:
        return None
    pose = [float(v) for v in tcp_pose]
    unit = getattr(robot_cfg, 'tcp_position_unit', 'mm')
    if unit == 'm':
        pose[:3] = [v * 1000.0 for v in pose[:3]]
    elif unit != 'mm':
        raise ValueError(f"不支持的 TCP 位置单位: {unit}")
    return pose


def get_robot_tcp_pose_mm(robot, robot_cfg):
    try:
        return tcp_pose_to_mm(robot.get_tcp_pose(), robot_cfg)
    except Exception as exc:
        logger.debug("读取 TCP 失败: %s", exc)
        return None


def compute_relative_pose(pos1, pos2):
    """计算两个位姿的相对位移 (基坐标系下)"""
    # 对于平移，直接相减即可
    # 对于旋转，使用ZYX欧拉角
    delta = [
        pos2[0] - pos1[0],  # dx
        pos2[1] - pos1[1],  # dy
        pos2[2] - pos1[2],  # dz
        pos2[3] - pos1[3],  # drx
        pos2[4] - pos1[4],  # dry
        pos2[5] - pos1[5],  # drz
    ]
    return delta


def apply_relative_pose(current, delta):
    """应用相对位移到当前位姿"""
    result = [
        current[0] + delta[0],  # x
        current[1] + delta[1],  # y
        current[2] + delta[2],  # z
        current[3] + delta[3],  # rx
        current[4] + delta[4],  # ry
        current[5] + delta[5],  # rz
    ]
    return result


def compute_distance(pose1, pose2):
    """计算两个位姿的距离（用于安全检查）"""
    trans_dist = np.sqrt(
        (pose1[0] - pose2[0])**2 +
        (pose1[1] - pose2[1])**2 +
        (pose1[2] - pose2[2])**2
    )
    rot_dist = np.sqrt(
        (pose1[3] - pose2[3])**2 +
        (pose1[4] - pose2[4])**2 +
        (pose1[5] - pose2[5])**2
    )
    return trans_dist, rot_dist


def save_offset_to_file(offset, filepath):
    """保存相对位移到文件（txt 格式：dx dy dz drx dry drz）"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f} "
                f"{offset[3]:.6f} {offset[4]:.6f} {offset[5]:.6f}\n")
    logger.info("相对位移已保存到 %s", filepath)


def load_offset_from_file(filepath):
    """从文件加载相对位移（txt 格式）"""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
        values = [float(x) for x in line.split()]
        if len(values) != 6:
            logger.warning("文件格式错误: 期望6个数值")
            return None
        logger.info("从文件加载相对位移: dx=%.2f dy=%.2f dz=%.2f mm, drx=%.2f dry=%.2f drz=%.2f deg",
                    *values)
        return values
    except Exception as e:
        logger.warning("加载文件失败: %s", e)
        return None


def save_poses_to_file(pos1, pos2, filepath):
    """保存位置1和位置2到文件（每行6个数值：x y z rx ry rz）"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        if pos1 is not None:
            f.write(f"pos1: {pos1[0]:.6f} {pos1[1]:.6f} {pos1[2]:.6f} "
                    f"{pos1[3]:.6f} {pos1[4]:.6f} {pos1[5]:.6f}\n")
        else:
            f.write("pos1: NONE\n")
        if pos2 is not None:
            f.write(f"pos2: {pos2[0]:.6f} {pos2[1]:.6f} {pos2[2]:.6f} "
                    f"{pos2[3]:.6f} {pos2[4]:.6f} {pos2[5]:.6f}\n")
        else:
            f.write("pos2: NONE\n")
    logger.info("位置1和位置2已保存到 %s", filepath)


def load_poses_from_file(filepath):
    """从文件加载位置1和位置2"""
    if not os.path.isfile(filepath):
        return None, None
    try:
        pos1, pos2 = None, None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("pos1:"):
                    parts = line[5:].strip().split()
                    if parts[0] != "NONE":
                        pos1 = [float(x) for x in parts]
                elif line.startswith("pos2:"):
                    parts = line[5:].strip().split()
                    if parts[0] != "NONE":
                        pos2 = [float(x) for x in parts]
        if pos1 is not None:
            logger.info("加载位置1: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *pos1)
        if pos2 is not None:
            logger.info("加载位置2: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *pos2)
        return pos1, pos2
    except Exception as e:
        logger.warning("加载位置文件失败: %s", e)
        return None, None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='相对位移复制工具')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='机械臂 IP（覆盖 config/robot.yaml）')
    parser.add_argument('--robot-driver', type=str, default=None,
                        choices=['jaka', 'modbus', 'crp', 'keba'],
                        help='机械臂驱动类型')
    parser.add_argument('--no-robot', action='store_true',
                        help='不连接机械臂，仅打印计算结果')
    parser.add_argument('--max-trans', type=float, default=DEFAULT_MAX_TRANS_MM,
                        help=f'最大允许平移 mm（默认 {DEFAULT_MAX_TRANS_MM}）')
    parser.add_argument('--max-rot', type=float, default=DEFAULT_MAX_ROT_DEG,
                        help=f'最大允许旋转 deg（默认 {DEFAULT_MAX_ROT_DEG}）')
    parser.add_argument('--speed', type=int, default=DEFAULT_SPEED,
                        help=f'运动速度 %%（默认 {DEFAULT_SPEED}）')
    parser.add_argument('--move-timeout', type=float, default=MOVE_TIMEOUT,
                        help=f'运动等待超时秒数（默认 {MOVE_TIMEOUT}）')
    parser.add_argument('--offset-file', type=str, default=_OFFSET_FILE,
                        help=f'相对位移保存路径（默认 {_OFFSET_FILE}）')
    parser.add_argument('--poses-file', type=str, default=_POSES_FILE,
                        help=f'位置1和2保存路径（默认 {_POSES_FILE}）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    robot = None
    robot_connected = False
    if not args.no_robot:
        if args.robot_ip:
            robot_cfg.ip = args.robot_ip
        robot = build_robot(robot_cfg)
        robot_connected = robot.connect()
        if robot_connected:
            if hasattr(robot, 'set_speed'):
                ret = robot.set_speed(args.speed)
                logger.info("速度设置 → %d%%: %s", args.speed,
                            "OK" if ret == 0 else f"失败({ret})")
            else:
                logger.info("当前机械臂驱动不支持速度设置")
        else:
            logger.warning("机械臂连接失败")
    else:
        logger.info("--no-robot 模式：不连接机械臂，仅计算")

    # 导入 CartesianPose（延迟导入以避免不依赖时的问题）
    try:
        from third_party.robot_driver.robot_driver_interface import CartesianPose
    except ImportError:
        # 如果导入失败，定义一个简单的类
        class CartesianPose:
            def __init__(self, x, y, z, rx, ry, rz):
                self.x, self.y, self.z = x, y, z
                self.rx, self.ry, self.rz = rx, ry, rz

    # 状态变量
    pos1 = None  # 位置1
    pos2 = None  # 位置2
    offset = None  # 相对位移 delta
    current_pose = None  # 当前位姿
    target_pose = None  # 目标位姿
    moving = False

    # 尝试加载之前保存的数据
    loaded_offset = load_offset_from_file(args.offset_file)
    if loaded_offset is not None:
        offset = loaded_offset

    loaded_pos1, loaded_pos2 = load_poses_from_file(args.poses_file)
    if loaded_pos1 is not None:
        pos1 = loaded_pos1
    if loaded_pos2 is not None:
        pos2 = loaded_pos2

    # 如果加载了pos1和pos2但没有offset，自动计算
    if pos1 is not None and pos2 is not None and offset is None:
        offset = compute_relative_pose(pos1, pos2)
        logger.info("从加载的位置自动计算相对位移")

    win = "RelativeOffsetCopier"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    logger.info("=" * 60)
    logger.info("相对位移复制工具已启动")
    logger.info("=" * 60)
    logger.info("按键说明:")
    logger.info("  1 - 记录并保存起点位置")
    logger.info("  2 - 记录并保存终点位置")
    logger.info("  s - 计算并保存相对位移 (需要先按1和2)")
    logger.info("  a - 应用相对位移 (当前位置 + delta)")
    logger.info("  l - 加载之前保存的数据")
    logger.info("  c - 清除所有记录")
    logger.info("  q - 退出")
    logger.info("=" * 60)

    try:
        while True:
            # 读取当前位姿
            if robot_connected:
                current_pose = get_robot_tcp_pose_mm(robot, robot_cfg)

            # 创建显示画面
            vis = np.zeros((720, 1280, 3), dtype=np.uint8)

            y = 50

            # 标题
            put_text(vis, "Relative Offset Copier | 相对位移复制工具", y, (0, 200, 255))
            y += 50

            # 当前位姿
            if current_pose is not None:
                put_text(vis, f"Current: X={current_pose[0]:.2f} Y={current_pose[1]:.2f} Z={current_pose[2]:.2f} mm",
                        y, (200, 200, 200))
                y += 30
                put_text(vis, f"         Rx={current_pose[3]:.2f} Ry={current_pose[4]:.2f} Rz={current_pose[5]:.2f} deg",
                        y, (200, 200, 200))
                y += 50
            else:
                put_text(vis, "Current: NOT CONNECTED", y, (0, 0, 255))
                y += 50

            # 位置1 (起点)
            if pos1 is not None:
                put_text(vis, f"[1] Start: X={pos1[0]:.2f} Y={pos1[1]:.2f} Z={pos1[2]:.2f} mm",
                        y, (0, 255, 0))
                y += 30
                put_text(vis, f"         Rx={pos1[3]:.2f} Ry={pos1[4]:.2f} Rz={pos1[5]:.2f} deg",
                        y, (0, 255, 0))
                y += 40
            else:
                put_text(vis, "[1] Start: NOT SET (press 1 to record)", y, (0, 165, 255))
                y += 40

            # 位置2 (终点)
            if pos2 is not None:
                put_text(vis, f"[2] End:   X={pos2[0]:.2f} Y={pos2[1]:.2f} Z={pos2[2]:.2f} mm",
                        y, (0, 255, 0))
                y += 30
                put_text(vis, f"         Rx={pos2[3]:.2f} Ry={pos2[4]:.2f} Rz={pos2[5]:.2f} deg",
                        y, (0, 255, 0))
                y += 50
            else:
                put_text(vis, "[2] End:   NOT SET (press 2 to record)", y, (0, 165, 255))
                y += 50

            # 相对位移
            if offset is not None:
                put_text(vis, f"[D] Delta(End-Start): dX={offset[0]:.2f} dY={offset[1]:.2f} dZ={offset[2]:.2f} mm",
                        y, (255, 200, 0))
                y += 30
                put_text(vis, f"                     dRx={offset[3]:.2f} dRy={offset[4]:.2f} dRz={offset[5]:.2f} deg",
                        y, (255, 200, 0))
                y += 40

                # 显示距离
                dist_trans = np.sqrt(offset[0]**2 + offset[1]**2 + offset[2]**2)
                dist_rot = np.sqrt(offset[3]**2 + offset[4]**2 + offset[5]**2)
                put_text(vis, f"    Total Distance: {dist_trans:.2f} mm, {dist_rot:.2f} deg",
                        y, (255, 200, 0))
                y += 40
            else:
                put_text(vis, "[D] Delta: NOT SET (press 1, 2, then s)", y, (0, 165, 255))
                y += 40

            # 目标位姿预览
            if offset is not None and current_pose is not None:
                target_pose = apply_relative_pose(current_pose, offset)
                put_text(vis, f"[T] Target:      X={target_pose[0]:.2f} Y={target_pose[1]:.2f} Z={target_pose[2]:.2f} mm",
                        y, (150, 255, 255))
                y += 30
                put_text(vis, f"                  Rx={target_pose[3]:.2f} Ry={target_pose[4]:.2f} Rz={target_pose[5]:.2f} deg",
                        y, (150, 255, 255))
                y += 40

            # 运动中提示
            if moving:
                put_text(vis, "MOVING... PLEASE WAIT", y, (0, 100, 255))
                y += 40

            # 按键提示
            y = 680
            help_text = "1=Start 2=End s=Calc&Save a=Apply l=Load c=Clear q=Quit"
            put_text(vis, help_text, y, (140, 140, 140))

            # 连接状态
            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            put_text(vis, robot_str, 705, (140, 140, 140))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('1'):
                if current_pose is None:
                    logger.warning("无法读取当前位姿，请检查机械臂连接")
                    continue
                pos1 = current_pose.copy()
                save_poses_to_file(pos1, pos2, args.poses_file)
                logger.info("记录并保存起点(位置1): X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *pos1)

            elif key == ord('2'):
                if current_pose is None:
                    logger.warning("无法读取当前位姿，请检查机械臂连接")
                    continue
                pos2 = current_pose.copy()
                save_poses_to_file(pos1, pos2, args.poses_file)
                logger.info("记录并保存终点(位置2): X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *pos2)

            elif key == ord('s'):
                # 计算相对位移
                if pos1 is None or pos2 is None:
                    logger.warning("请先记录起点和终点位置（按1和2）")
                    continue
                offset = compute_relative_pose(pos1, pos2)
                dist_trans = np.sqrt(offset[0]**2 + offset[1]**2 + offset[2]**2)
                dist_rot = np.sqrt(offset[3]**2 + offset[4]**2 + offset[5]**2)
                logger.info("计算并保存相对位移: dX=%.2f dY=%.2f dZ=%.2f mm, dRx=%.2f dRy=%.2f dRz=%.2f deg",
                            *offset)
                logger.info("  总距离: %.2f mm, %.2f deg", dist_trans, dist_rot)
                save_offset_to_file(offset, args.offset_file)

            elif key == ord('a'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if offset is None:
                    logger.warning("请先设置相对位移（记录位置1和2，或加载文件）")
                    continue
                if current_pose is None:
                    logger.warning("无法读取当前位姿，请检查机械臂连接")
                    continue

                # 计算目标位姿
                target_pose = apply_relative_pose(current_pose, offset)

                # 安全检查
                dist_trans, dist_rot = compute_distance(current_pose, target_pose)
                if dist_trans > args.max_trans or dist_rot > args.max_rot:
                    logger.warning("运动距离过大: 平移=%.2f mm (max %.2f), 旋转=%.2f deg (max %.2f)",
                                dist_trans, args.max_trans, dist_rot, args.max_rot)
                    response = input("是否继续? (y/n): ")
                    if response.lower() != 'y':
                        continue

                logger.info("应用相对位移:")
                logger.info("  当前: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *current_pose)
                logger.info("  增量: dX=%.2f dY=%.2f dZ=%.2f mm, dRx=%.2f dRy=%.2f dRz=%.2f deg", *offset)
                logger.info("  目标: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *target_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行运动")
                else:
                    moving = True
                    cart = CartesianPose(
                        x=target_pose[0], y=target_pose[1], z=target_pose[2],
                        rx=target_pose[3], ry=target_pose[4], rz=target_pose[5],
                    )
                    ok = robot.move_and_wait(cart, timeout=args.move_timeout)
                    moving = False

                    if ok:
                        logger.info("运动完成")
                        # 读取运动后的位姿
                        final_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if final_pose is not None:
                            logger.info("  实际: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *final_pose)
                    else:
                        logger.warning("运动超时或失败")

            elif key == ord('s'):
                # 保存位置1、位置2、相对位移
                if pos1 is not None or pos2 is not None:
                    save_poses_to_file(pos1, pos2, args.poses_file)
                if offset is not None:
                    save_offset_to_file(offset, args.offset_file)

            elif key == ord('l'):
                # 加载之前保存的位置和相对位移
                loaded_pos1, loaded_pos2 = load_poses_from_file(args.poses_file)
                if loaded_pos1 is not None:
                    pos1 = loaded_pos1
                if loaded_pos2 is not None:
                    pos2 = loaded_pos2
                loaded_offset = load_offset_from_file(args.offset_file)
                if loaded_offset is not None:
                    offset = loaded_offset
                if loaded_pos1 is not None or loaded_pos2 is not None or loaded_offset is not None:
                    logger.info("数据已加载")

            elif key == ord('c'):
                pos1 = None
                pos2 = None
                offset = None
                logger.info("已清除所有记录")

    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.disconnect()


if __name__ == '__main__':
    main()
