#!/usr/bin/env python3
"""
充电枪插入/拔出全流程自动化 | Full Auto Insert/Extract Pipeline (v4.7)

通过后台 Unix Domain Socket 接收外部触发信号 ({"auto_insert_gun": 1})，
自动按序执行 18 步状态机，完成充电枪插入→拔出→归枪→取小盖→放回小盖的完整流程。
同时保留手动按键操作作为调试/回退手段。

状态机流程 (State Machine Flow):
    STATE_1  → 机械臂回初始关节位 + 双目粗定位充电口盖 → 粗定位运动
    STATE_2  → 力控按压开盖
    STATE_3  → 拨盖运动 (CoverActionFlow)
    STATE_4  → 自动多次视觉对准 (插枪 ArUco)
    STATE_5  → 对准后固定偏移运动
    STATE_6  → 夹住小盖并放小盖 → 运动到取枪初始点
    STATE_7  → 自动多次视觉对准 (取枪 ArUco)
    STATE_8  → 固定偏移运动 + 沿法兰z向直线运动 → 触发夹爪和舵机
    STATE_9  → 沿法兰z向退出 → 取出充电枪并复位舵机
    STATE_10 → 插枪前运动
    STATE_11  → 力控插枪
    STATE_11B → 开夹爪+沿法兰z退150mm+记驻停位姿+去FINALL_POINT → 等12s → 回取枪逼近位姿→视觉对准(同STATE_7)→STATE_8取枪逻辑
    STATE_12  → 力控拔枪
    STATE_13 → 归枪运动
    STATE_14 → 移动至取小盖前点位
    STATE_15 → 自动多次视觉对准 (取小盖 ArUco)
    STATE_16 → 对准后固定偏移运动
    STATE_17 → 夹住小盖放回充电口 + 关大盖 → 回到初始点

    IDLE → 等待 UDS 触发或手动按键操作
    任何步骤失败或按 'q'/ESC 中止自动流程 → 回到 IDLE

光照鲁棒: --lighting-robust 启用后台线程，增益优先→曝光兜底，按 ref ROI 亮度自动调节。
  - 按 'r' 保存当前 marker ROI 亮度为参考（同时持久化到文件）
  - 启动时自动从文件加载参考亮度，有值则 LR 线程生效

手动按键 (Manual Keys, IDLE 状态下):
    按键 'r': 记录当前 ArUco 位姿为参考（自动识别插枪/取枪 marker），同步保存法兰坐标系下的参考 TCP
    按键 '3','d','g','m','b','h','a','c','j','e','s','k','l','p': 单独执行对应步骤（调试用）
    按键 'q' / ESC: 退出

用法 (Usage):
    python scripts/oneshot_move_crobot_insert_auto_brightness_v4.7.py --camera mecheye --insert-cm-insert 7 --insert-cm-take 12
    python scripts/oneshot_move_crobot_insert_auto_brightness_v4.7.py --camera mecheye --no-robot --align-max-attempts 4
"""

import argparse
import logging
import os
import sys
import threading
import time
import fcntl
import json
import socket
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from enum import IntEnum

import cv2
import numpy as np

# --- robovision 核心 ---
from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.aruco import (
    ArucoDetector, build_raw_aruco_detector, detect_raw_frame,
)
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose, compute_new_tool_pose,
    offset_pose_along_tool_axis,
)
from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file, save_pose_file
from robovision.visualization.aruco_overlay import (
    draw_aruco_result_for_display, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT,
)
from robovision.servo.core import (
    aruco_to_matrix, compute_pose_error, compute_pose_error_in_frame, execute_move,
)

# --- 第三方驱动 ---
from crobot_driver_interface import CartesianPose
from scripts.cover_main_bak_run import CoverActionFlow
from gripper_controller import GripperController
from relative_move import apply_relative_pose, load_offset_from_file, compute_distance
from Intergration.stereo_camera_calib.yrq.pose_estimation.cover_pose_estimator import CoverPoseEstimator
from Intergration.FTServo_Linux_main.examples.sms_sts_driver import SMSSTSController

from third_party.force_control_crp import RobotController, BridgeCRobotAdapter


# ── Unix Domain Socket 服务（接收外部进程自动化控制信号） ──
# 监听 /tmp/auto_gun.sock，外部进程通过 Unix Domain Socket
# 发送 JSON 报文 {"auto_insert_gun": 1} 触发自动化流程。
SOCK = "/tmp/auto_gun.sock"

async def to_thread(func,/,*args,**kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None,func,*args,**kwargs)

# ── 自动化状态机 | Auto State Machine ──
class AutoState(IntEnum):
    IDLE = -1
    STATE_1_INIT_COVER = 1
    STATE_2_FORCE_OPEN = 2
    STATE_3_COVER_ACTION = 3
    STATE_4_ALIGN_INSERT = 4
    STATE_5_OFFSET_B = 5
    STATE_6_GRAB_INNER = 6
    STATE_7_ALIGN_TAKE = 7
    STATE_8_OFFSET_A = 8
    STATE_9_RETRACT_C = 9
    STATE_10_PRE_INSERT = 10
    STATE_11_FORCE_IN = 11
    STATE_11B_PARK_RETURN = 110
    STATE_12_FORCE_OUT = 12
    STATE_13_RETURN_GUN = 13
    STATE_14_ALIGN_POINT = 14
    STATE_15_ALIGN_INNER = 15
    STATE_16_OFFSET_B2 = 16
    STATE_17_CLOSE_COVER = 17

# ── UDS 触发事件（线程安全） ──
auto_trigger_event = threading.Event()
trigger_count = 0
auto_trigger_time = None  # 接收到触发信号的时间戳（用于统计全流程总耗时）
global fullworkflow

def trigger_control(auto_gun: int):
    """收到 auto_gun=1 信号后设置触发事件，主循环检测到后启动自动化流程。"""
    global trigger_count, auto_trigger_time
    trigger_count += 1
    logger.info("[trigger_control] auto_insert_gun=%s (第%d次)", auto_gun, trigger_count)
    if auto_gun == 1:
        auto_trigger_time = time.time()
        auto_trigger_event.set()
        fullworkflow()

def _socket_server():
    """后台 UDS 监听线程：接收外部进程发来的 JSON 控制信号。

    外部进程连接 /tmp/auto_gun.sock 并发送 JSON 行
    {"auto_insert_gun": 1} 触发自动化流程。
    """
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    logger.info("[UDS] 监听 %s", SOCK)
    while True:
        conn, _ = srv.accept()
        with conn:
            data = conn.recv(4096)
            if data:
                for line in data.decode().split("\n"):
                    if line.strip():
                        try:
                            payload = json.loads(line)
                            logger.info("[UDS] 收到信号: %s", payload)
                            trigger_control(payload.get("auto_insert_gun", 0))
                        except json.JSONDecodeError:
                            logger.warning("[UDS] 收到无效 JSON: %s", line.strip())


# --- 本地辅助 (替代 robovision.robot.tool_coord，适配 CRP 驱动) ---
def check_oneshot_safety(trans_mm, rot_deg, max_trans, max_rot):
    if trans_mm > max_trans:
        return False, (f"平移 {trans_mm:.1f} mm 超过阈值 {max_trans:.1f} mm，请手动移近后重试或增大 --max-trans")
    if rot_deg > max_rot:
        return False, (f"旋转 {rot_deg:.2f} deg 超过阈值 {max_rot:.1f} deg，请手动调整姿态后重试或增大 --max-rot")
    return True, ""

def _marker_roi_brightness(frame, target_data):
    """从 ArUco 检测结果提取 marker ROI 并返回中值亮度。失败返回 None。"""
    corners = target_data.get('filtered_corners')
    if corners is None:
        corners = target_data.get('raw_corners')
    if corners is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    x1 = max(0, int(corners[:, 0].min() - 20))
    y1 = max(0, int(corners[:, 1].min() - 20))
    x2 = min(w, int(corners[:, 0].max() + 20))
    y2 = min(h, int(corners[:, 1].max() + 20))
    if x2 <= x1 or y2 <= y1:
        return None
    return float(np.median(gray[y1:y2, x1:x2]))


def pick_stable_alignment_context(contexts, max_spread_mm):
    """从多帧对准上下文中选取中位数帧，仅在波动 <= max_spread_mm 时返回。"""
    if not contexts:
        return None, None
    ordered = sorted(contexts, key=lambda item: float(item["aruco_norm"]))
    spread = float(ordered[-1]["aruco_norm"] - ordered[0]["aruco_norm"])
    if spread > float(max_spread_mm):
        return None, spread
    return ordered[len(ordered) // 2], spread


def should_finish_alignment(attempt, aruco_norm_mm, success_mm=1.0, min_attempts=2):
    """判断是否可以提前结束对准循环。"""
    return int(attempt) >= int(min_attempts) and float(aruco_norm_mm) <= float(success_mm)


def ensure_tool_id(robot, tool_id, label=None):
    return True

def get_tcp_pose_in_tool(robot, tool_id, label=None):
    return robot.get_tcp_pose()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# ============================================================
# 常量 | Constants
# ============================================================

DEFAULT_MAX_TRANS_MM = 420.0   # 一次性运动允许的最大平移 (mm)
DEFAULT_MAX_ROT_DEG = 50.0     # 一次性运动允许的最大旋转 (deg)
DEFAULT_SPEED = 100             # 默认运动速度 (%)
MOVE_TIMEOUT = 30.0            # 运动超时 (s)
INSERT_TOOL_ID = 0             # 工具坐标系 ID (0 = 法兰，用于沿工具轴执行插入/拔出)
BASE_TOOL_ID = 0               # 参考点保存时使用的法兰坐标系 ID

# 关节角 (deg) | Joint poses
INIT_JOINT = [112.196, 99.884, -56.131, 128.446, -5.912, -17.849]            # 初始关节位 step_1
STEREO_IK_REF_JOINT = [118.086, 30.858, 5.595, 129.986, -9.015, -17.671]     # 双目逆解参考位姿

# 笛卡尔偏移位姿 [x, y, z, rx, ry, rz] (mm/deg) | Cartesian offset poses
STEREO_DETECT_OFFSET_POSE = [52.306, 11.298, -287.556, 0.744, 2.402, -1.644]  # 双目识别 offset
INSERT_BEFORE_OFFSET_POSE = [1.699, -226.26, -41.071, -6.271, 2.87, -1.234]   # 插枪前模板偏移 原按键‘m’
INNER_COVER_OFFSET_POSE = [15.786, -187.891, 263.694, -9.333, 2.806, -4.281]  # 取小盖偏移 原按键‘b’
TAKEGUN_OFFSET_POSE = [96.934, -163.317, 242.083, -4.635, -4.807, -1.68]      # 取枪偏移 原按键‘a’
PARKED_OFFSET = [-78.873, 153.876, -254.146, 0, 0, 0]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICK = 1
PROMPT_TEXT_COLOR = (255, 255, 255)
_TAKEGUN_OFFSET_FILE = "data/relative_offset_takegun.txt"
_POSES_FILE = "data/relative_poses_takegun.txt"
_CHARGING_OFFSET_FILE =  "data/relative_offset_charging.txt"

GAIN_LIMIT_DB = 12.0

# 光照鲁棒线程参数
LR_DEADBAND = 1
LR_GAIN_STEP = 0.4
LR_EXP_LIMIT_US = 100000
LR_EXP_MIN_US = 100
LR_EXP_RATIO = 0.05

MARKER_DISPLAY_NAMES = {
    "插枪": "Insert",
    "取枪": "Take",
}


# ============================================================
# 细粒度运动分段计时器 | Motion Segment Timer
# ============================================================

class MotionSegmentTimer:
    """记录每段机械臂/夹爪/舵机运动的耗时与TCP位姿，最终导出txt汇总。"""

    def __init__(self, robot, robot_cfg, robot_connected, output_dir="data"):
        self._segments = []
        self._robot = robot
        self._robot_cfg = robot_cfg
        self._robot_connected = robot_connected
        self._output_dir = output_dir
        self._step_name = "UNKNOWN"
        self._seg_idx = 0
        self._start_time = time.time()

    def set_step(self, name):
        self._step_name = name
        self._seg_idx = 0

    @staticmethod
    def _format_pose(pose):
        if pose is None:
            return "N/A"
        return "X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f" % tuple(pose)

    @staticmethod
    def _pose_csv(pose):
        if pose is None:
            return "N/A"
        return ", ".join("%.2f" % v for v in pose)

    def _read_tcp(self):
        if not self._robot_connected or self._robot is None:
            return None
        try:
            tcp = self._robot.get_tcp_pose()
            if tcp is None:
                return None
            pose = [float(v) for v in tcp]
            unit = getattr(self._robot_cfg, 'tcp_position_unit', 'mm')
            if unit == 'm':
                pose[:3] = [v * 1000.0 for v in pose[:3]]
            elif unit != 'mm':
                raise ValueError(f"Unsupported TCP position unit: {unit}")
            return pose
        except Exception:
            return None

    @contextmanager
    def segment(self, label):
        seg_idx = self._seg_idx + 1
        self._seg_idx = seg_idx
        tcp_before = self._read_tcp()
        logger.info("[计时] %s | #%d %s 开始", self._step_name, seg_idx, label)
        if tcp_before is not None:
            logger.info("[计时]  %s | #%d 位姿(前): %s", self._step_name, seg_idx, self._format_pose(tcp_before))
        t0 = time.time()
        yield
        dt = time.time() - t0
        tcp_after = self._read_tcp()
        logger.info("[计时] %s | #%d %s 完成 耗时=%.3fs", self._step_name, seg_idx, label, dt)
        if tcp_after is not None:
            logger.info("[计时]  %s | #%d 位姿(后): %s", self._step_name, seg_idx, self._format_pose(tcp_after))
        self._segments.append({
            'step': self._step_name,
            'idx': seg_idx,
            'label': label,
            'dt': dt,
            'tcp_before': list(tcp_before) if tcp_before is not None else None,
            'tcp_after': list(tcp_after) if tcp_after is not None else None,
        })

    def export_txt(self, path=None):
        if not self._segments:
            return

        if path is None:
            os.makedirs(self._output_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._output_dir, f"timing_log_{ts}.txt")

        total_time = time.time() - self._start_time

        lines = []
        lines.append("=" * 100)
        lines.append("  细粒度运动计时汇总")
        lines.append(f"  工作时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  总耗时: {total_time:.3f}s")
        lines.append(f"  总运动段数: {len(self._segments)}")
        lines.append("=" * 100)
        lines.append("")

        step_times = {}
        for seg in self._segments:
            sn = seg['step']
            if sn not in step_times:
                step_times[sn] = {'total': 0.0, 'count': 0}
            step_times[sn]['total'] += seg['dt']
            step_times[sn]['count'] += 1

        step_order = list(dict.fromkeys(s['step'] for s in self._segments))
        for sn in step_order:
            info = step_times[sn]
            lines.append("-" * 100)
            lines.append(f"  Step: {sn} | 耗时: {info['total']:.3f}s | {info['count']}段")
            lines.append("-" * 100)
            for seg in self._segments:
                if seg['step'] != sn:
                    continue
                lines.append(f"  #{seg['idx']:<2} {seg['label']:<50} {seg['dt']:.3f}s")
                lines.append(f"      前: {self._pose_csv(seg['tcp_before'])}")
                lines.append(f"      后: {self._pose_csv(seg['tcp_after'])}")

        lines.append("")
        lines.append("=" * 100)
        lines.append("  Step 耗时汇总")
        lines.append("=" * 100)
        for sn in step_order:
            info = step_times[sn]
            lines.append(f"  {sn:<12}: {info['total']:.3f}s ({info['count']}段)")
        lines.append("-" * 100)
        lines.append(f"  {'Total':<12}: {total_time:.3f}s ({len(self._segments)}段)")
        lines.append("=" * 100)

        content = "\n".join(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("[计时] 汇总已导出: %s", path)
        return path


# ============================================================
# 本地辅助函数 | Local Helpers
# ============================================================

def put_text(img, text, y, color=(200, 200, 200)):
    """在图像上绘制带黑色描边的文本。"""
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2)
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def marker_display_name(marker_type):
    """将内部 marker 类型标识转为英文显示名。"""
    return MARKER_DISPLAY_NAMES.get(marker_type, "Unknown")


def tcp_pose_to_mm(tcp_pose, robot_cfg):
    """将驱动返回的 TCP 位姿转换为脚本内部统一使用的 mm/deg 格式。"""
    if tcp_pose is None:
        return None
    pose = [float(v) for v in tcp_pose]
    unit = getattr(robot_cfg, 'tcp_position_unit', 'mm')
    if unit == 'm':
        pose[:3] = [v * 1000.0 for v in pose[:3]]
    elif unit != 'mm':
        raise ValueError(f"Unsupported TCP position unit: {unit}")
    return pose


def get_robot_tcp_pose_mm(robot, robot_cfg):
    """从机器人读取法兰坐标系 TCP 位姿 (mm/deg)。"""
    try:
        return tcp_pose_to_mm(robot.get_tcp_pose(), robot_cfg)
    except Exception as exc:
        logger.debug("Failed to read TCP: %s", exc)
        return None


def get_tcp_pose_in_tool_mm(robot, robot_cfg, tool_id, label=None):
    """从机器人读取指定工具坐标系 TCP 位姿 (mm/deg)。"""
    try:
        return tcp_pose_to_mm(get_tcp_pose_in_tool(robot, tool_id, label=label), robot_cfg)
    except Exception as exc:
        logger.debug("Failed to read tool %s TCP: %s", label or tool_id, exc)
        return None


def execute_keba_force_mode(robot, mode, displace_target=0):
    """执行 KEBA mode_1 (力控插枪) / mode_3 (力控拔枪) 动作。"""
    move_force = getattr(robot, 'move_force', None)
    if not callable(move_force):
        logger.warning("Current robot driver does not support KEBA force mode")
        return False

    logger.info("Executing KEBA force mode %d...", int(mode))
    ok = move_force(int(mode), displace_target=int(displace_target))
    if ok:
        logger.info("KEBA force mode %d dispatched", int(mode))
    else:
        logger.warning("KEBA force mode %d dispatch failed", int(mode))
    return bool(ok)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='充电枪全流程自动化 v4.7')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--hand-eye', type=str, default='/home/nvidia/Downloads/HD/HD_0323/data/handeye_eye_in_hand/hand_eye_result_mono.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考插枪 ArUco 位姿文件')
    parser.add_argument('--aruco-ref-takegun', type=str, default='data/aruco/aruco_pose_ref_takegun.txt',
                        help='参考取枪 ArUco 位姿文件')
    parser.add_argument('--target-marker', type=int, default=1,
                        help='插枪 ArUco Marker ID')
    parser.add_argument('--takegun-marker', type=int, default=0,
                        help='取枪 ArUco Marker ID')
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
    parser.add_argument('--insert-cm-insert', type=float, default=7,
                        help='插枪时沿工具 z 轴前进距离（cm）')
    parser.add_argument('--insert-cm-take', type=float, default=12,
                        help='取枪时沿工具 z 轴前进距离（cm）')
    parser.add_argument('--out-mm', type=float, default=35,
                        help='拔枪时沿工具 z 轴后退距离（注意：内部会乘以 10 转换为实际移动量）')
    parser.add_argument('--move-timeout', type=float, default=MOVE_TIMEOUT,
                        help=f'普通运动等待超时秒数（默认 {MOVE_TIMEOUT}）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（固定 IPPE_SQUARE，且不使用时序滤波）')
    parser.add_argument('--no-temporal-filter', action='store_true',
                        help='关闭 Kalman 角点滤波和位姿时序平滑（仅影响标准检测模式）')
    parser.add_argument('--align-min-attempts', type=int, default=3,
                        help='自动对准最少执行次数（默认 3）')
    parser.add_argument('--align-max-attempts', type=int, default=6,
                        help='自动对准最大执行次数（默认 6）')
    parser.add_argument('--align-success-mm', type=float, default=1.0,
                        help='ArUco 平移误差成功阈值 mm（默认 1.0）')
    parser.add_argument('--align-settle-s', type=float, default=0.5,
                        help='每次运动后等待画面稳定秒数（默认 0.5）')
    parser.add_argument('--align-stable-samples', type=int, default=5,
                        help='稳定窗口帧数（默认 5）')
    parser.add_argument('--align-stable-timeout-s', type=float, default=5.0,
                        help='等待稳定窗口超时秒数（默认 5.0）')
    parser.add_argument('--align-stable-spread-mm', type=float, default=0.5,
                        help='窗口内误差最大波动 mm（默认 0.5）')
    parser.add_argument('--lighting-robust', action='store_true',
                        help=f'启用光照鲁棒线程：增益优先→曝光兜底（死区±{LR_DEADBAND}，增益步长{LR_GAIN_STEP}dB，曝光步长{LR_EXP_RATIO*100:.0f}%）')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--step-mode', action='store_true',
                        help='单步调试模式：每完成一个 step 暂停，按 ENTER 继续下一步')
    parser.add_argument('--debug-start-step', type=int, default=None,
                        help='配合 --step-mode，从指定步骤 N 开始执行（1-17）')
    parser.add_argument('--takegun-offset-file', type=str, default=_TAKEGUN_OFFSET_FILE,
                        help=f'相对位移保存路径（默认 {_TAKEGUN_OFFSET_FILE}）')
    parser.add_argument('--charging-offset-file', type=str, default=_CHARGING_OFFSET_FILE,
                        help=f'相对位移保存路径（默认 {_CHARGING_OFFSET_FILE}）')
    parser.add_argument('--poses-file', type=str, default=_POSES_FILE,
                        help=f'位置1和2保存路径（默认 {_POSES_FILE}）')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    step_mode = args.step_mode
    debug_start_step = args.debug_start_step

    _STEP_EXEC_ORDER = [
        AutoState.STATE_1_INIT_COVER,
        AutoState.STATE_2_FORCE_OPEN,
        AutoState.STATE_3_COVER_ACTION,
        AutoState.STATE_4_ALIGN_INSERT,
        AutoState.STATE_5_OFFSET_B,
        AutoState.STATE_6_GRAB_INNER,
        AutoState.STATE_7_ALIGN_TAKE,
        AutoState.STATE_8_OFFSET_A,
        AutoState.STATE_9_RETRACT_C,
        AutoState.STATE_10_PRE_INSERT,
        AutoState.STATE_11_FORCE_IN,
        AutoState.STATE_11B_PARK_RETURN,
        AutoState.STATE_12_FORCE_OUT,
        AutoState.STATE_13_RETURN_GUN,
        AutoState.STATE_14_ALIGN_POINT,
        AutoState.STATE_15_ALIGN_INNER,
        AutoState.STATE_16_OFFSET_B2,
        AutoState.STATE_17_CLOSE_COVER,
    ]

    # 启动 Unix Domain Socket 监听线程（接收外部进程控制信号）
    threading.Thread(target=_socket_server, daemon=True, name="uds-server").start()


    # 运动距离：命令行单位为 cm，内部统一为 mm
    insert_offset_mm = args.insert_cm_insert * 10.0   # 插枪时的插入距离
    take_offset_mm = args.insert_cm_take * 10.0       # 取枪时的插入距离
    retract_offset_mm = args.out_mm * 10              # 拔枪后退距离

    takegun_offset = None  # 取枪对准后保存的相对位移 (后续由文件加载或运行时计算)

    charging_offset = None

    # 大盖是否打开标志位
    outer_cover_is_opened = False

    # 是否归枪标志位
    already_return_gun = False

    # 是否夹完小盖标志位
    already_take_inner_cover = False

    # 加载全局配置
    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    # 加载手眼标定
    T_c2g = load_hand_eye_result(args.hand_eye)
    logger.info("Hand-eye calibration loaded: t=[%.2f, %.2f, %.2f]",
                T_c2g[0, 3], T_c2g[1, 3], T_c2g[2, 3])

    # --- 加载插枪参考位姿 (ArUco + TCP) ---
    T_aruco2cam_ref_insert = None
    if os.path.isfile(args.aruco_ref):
        ref_pose = load_pose_file(args.aruco_ref)[-1]
        T_aruco2cam_ref_insert = pose_to_matrix(ref_pose)
        logger.info("Insert ref pose loaded: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref, *ref_pose[:3])
    else:
        logger.info("Insert ref pose file not found; move to reference position and press 'r'")

    tcp_ref_path_insert = args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
    T_g2b_ref_insert = None
    if os.path.isfile(tcp_ref_path_insert):
        tcp_ref_pose = load_pose_file(tcp_ref_path_insert)[-1]
        T_g2b_ref_insert = pose_to_matrix(tcp_ref_pose)
        logger.info("Insert ref TCP (tool %d) loaded: t=[%.2f, %.2f, %.2f] mm",
                    BASE_TOOL_ID, *tcp_ref_pose[:3])

    # --- 加载取枪参考位姿 (ArUco + TCP) ---
    T_aruco2cam_ref_takegun = None
    if os.path.isfile(args.aruco_ref_takegun):
        ref_pose_take = load_pose_file(args.aruco_ref_takegun)[-1]
        T_aruco2cam_ref_takegun = pose_to_matrix(ref_pose_take)
        logger.info("Take ref pose loaded: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref_takegun, *ref_pose_take[:3])
    else:
        logger.info("Take ref pose file not found; move to reference position and press 'r'")

    tcp_ref_path_takegun = args.aruco_ref_takegun.replace('aruco_pose_ref_takegun', 'tcp_ref_takegun')
    T_g2b_ref_takegun = None
    if os.path.isfile(tcp_ref_path_takegun):
        tcp_ref_pose_take = load_pose_file(tcp_ref_path_takegun)[-1]
        T_g2b_ref_takegun = pose_to_matrix(tcp_ref_pose_take)
        logger.info("Take ref TCP (tool %d) loaded: t=[%.2f, %.2f, %.2f] mm",
                    BASE_TOOL_ID, *tcp_ref_pose_take[:3])

    # 加载取枪相对位移偏移量
    loaded_offset = load_offset_from_file(args.takegun_offset_file)
    if loaded_offset is not None:
        takegun_offset = loaded_offset
    
    charging_offset = load_offset_from_file(args.charging_offset_file)

    # 加载 ROI 亮度参考（光照鲁棒线程用）
    exp_ref_path_insert = args.aruco_ref.replace('aruco_pose_ref', 'aruco_exp_ref')
    exp_ref_path_takegun = args.aruco_ref_takegun.replace('aruco_pose_ref_takegun', 'aruco_exp_ref_takegun')
    ref_bri_insert = None
    if os.path.isfile(exp_ref_path_insert):
        with open(exp_ref_path_insert, 'r') as f:
            try:
                val = f.read().strip().split(',')[0]
                ref_bri_insert = float(val)
                logger.info("Insert bri ref loaded: %.0f", ref_bri_insert)
            except (ValueError, IndexError):
                pass
    ref_bri_takegun = None
    if os.path.isfile(exp_ref_path_takegun):
        with open(exp_ref_path_takegun, 'r') as f:
            try:
                val = f.read().strip().split(',')[0]
                ref_bri_takegun = float(val)
                logger.info("Take bri ref loaded: %.0f", ref_bri_takegun)
            except (ValueError, IndexError):
                pass


    # --- 初始化机器人流程壳对象，避免先构造旧 CRobot 会话 ---
    flow = CoverActionFlow.__new__(CoverActionFlow)
    flow._connected = False
    flow.gripper = GripperController(port='/dev/ttysWK3', baudrate=115200, slave_id=4)
    with open("/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json", "r", encoding="utf-8") as f:
        flow.CFG = json.load(f)

    robot = None
    robot_connected = False
    if not args.no_robot:
        crp_ip = args.robot_ip or getattr(robot_cfg, 'ip', None) or '192.168.1.12'
        crp_so_path = getattr(robot_cfg, 'so_path', 'third_party/crp_robot_sdk/libRobotService.so')
        if not os.path.isabs(crp_so_path):
            crp_so_path = os.path.join(_ROOT, crp_so_path)
        bridge_robot = BridgeCRobotAdapter(ip=crp_ip, so_path=crp_so_path)
        robot_connected = bridge_robot.connect()
        if robot_connected:
            flow.arm = bridge_robot
            robot = bridge_robot
            flow.gripper.connect()
            flow._connected = True
            robot.set_speed(args.speed)
            logger.info("Speed set to %d%%: OK", args.speed)
        else:
            logger.warning("Robot connection failed")
    else:
        logger.info("--no-robot mode: no robot connection, calculation only")

    # --- 初始化力控制器 (六维力传感器) | Force controller init ---
    _force_so = os.path.join(_ROOT, 'third_party', 'force_control_crp', 'libforcecontrol_crp.so')
    force_ctrl = RobotController(so_path=_force_so)
    force_ctrl.load()
    if robot_connected:
        force_ctrl.attach_crp_robot(robot.bridge_robot)
    logger.info("Force controller loaded")

    # --- 初始化相机和检测器 | Camera & detector init ---
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    use_raw = args.raw
    use_temporal_filter = not args.no_temporal_filter
    if use_raw:
        raw_detector = build_raw_aruco_detector(marker_cfg.dictionary)
        valid_ids = set(marker_cfg.valid_ids)
        marker_sizes = marker_cfg.marker_sizes
        detector = None
        logger.info("Detection mode: RAW (IPPE_SQUARE, no filter)")
    else:
        detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
        detector.set_temporal_filter(use_temporal_filter)
        raw_detector = None
        logger.info("Detection mode: ArucoDetector (%s, multi-method PnP)",
                    "temporal filter ON" if use_temporal_filter else "temporal filter OFF")

    # --- Marker ID configuration ---
    insert_marker_id = args.target_marker
    takegun_marker_id = args.takegun_marker
    # Reference pose status
    ref_set_insert = T_aruco2cam_ref_insert is not None
    ref_set_takegun = T_aruco2cam_ref_takegun is not None
    moving = False

    win = f"OneShotInsert [{args.camera}] InsertID={insert_marker_id} TakeID={takegun_marker_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    det_lock = threading.Lock()
    latest_det = {}
    latest_det_frame = None
    stop_event = threading.Event()
    _lr_ref_bris = {}  # 光照鲁棒线程参考亮度 {marker_id: bri}
    if ref_bri_insert is not None:
        _lr_ref_bris[insert_marker_id] = ref_bri_insert
    if ref_bri_takegun is not None:
        _lr_ref_bris[takegun_marker_id] = ref_bri_takegun

    def _detection_loop():
        nonlocal latest_det, latest_det_frame
        while not stop_event.is_set():
            ok_d, frm = camera.read_frame()
            if not ok_d or frm is None:
                time.sleep(0.005)
                continue
            frm = frm.copy()
            if use_raw:
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY) if frm.ndim == 3 else frm
                result = detect_raw_frame(gray, raw_detector, valid_ids, marker_sizes, K, dist)
            else:
                result = detector.detect(frm)
            with det_lock:
                latest_det = result
                latest_det_frame = frm

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()
    logger.info("Ready: UDS trigger or manual keys r=Ref q=Quit (detection thread started)")

    # --- 光照鲁棒线程（增益优先 → 曝光兜底） ---
    if args.lighting_robust:
        def _lighting_robust_loop():
            while not stop_event.is_set():
                if not _lr_ref_bris:
                    time.sleep(1.0)
                    continue
                with det_lock:
                    cur_frame = latest_det_frame
                    cur_det = latest_det
                if cur_frame is None or not cur_det:
                    time.sleep(1.0)
                    continue

                deviations = []
                for mid, ref_bri in _lr_ref_bris.items():
                    if mid in cur_det:
                        cur_bri = _marker_roi_brightness(cur_frame, cur_det[mid])
                        if cur_bri is not None:
                            deviations.append(cur_bri - ref_bri)

                if not deviations:
                    time.sleep(0.15)
                    continue

                avg_dev = sum(deviations) / len(deviations)

                if abs(avg_dev) <= LR_DEADBAND:
                    time.sleep(0.15)
                    continue

                try:
                    cur_gain = camera.get_gain()
                    if cur_gain is None:
                        cur_gain = 0.0
                    cur_exp = camera.get_exposure_time()
                    if cur_exp is None:
                        cur_exp = 5000.0

                    if avg_dev < 0:
                        if cur_gain < GAIN_LIMIT_DB - LR_GAIN_STEP:
                            new_gain = min(cur_gain + LR_GAIN_STEP, GAIN_LIMIT_DB)
                            camera.set_gain(new_gain)
                            logger.debug("Lighting adj: dev=%.1f  gain %.1f→%.1f dB",
                                         avg_dev, cur_gain, new_gain)
                        else:
                            new_exp = min(cur_exp * (1.0 + LR_EXP_RATIO), LR_EXP_LIMIT_US)
                            if new_exp / cur_exp - 1.0 > 0.001:
                                camera.set_exposure_time(new_exp)
                                logger.debug("Lighting adj: dev=%.1f  exp %.0f→%.0f us",
                                             avg_dev, cur_exp, new_exp)
                    else:
                        if cur_gain > LR_GAIN_STEP:
                            new_gain = max(cur_gain - LR_GAIN_STEP, 0.0)
                            if cur_gain - new_gain > 0.001:
                                camera.set_gain(new_gain)
                                logger.debug("Lighting adj: dev=%.1f  gain %.1f→%.1f dB",
                                             avg_dev, cur_gain, new_gain)
                        else:
                            new_exp = max(cur_exp * (1.0 - LR_EXP_RATIO), LR_EXP_MIN_US)
                            if 1.0 - new_exp / cur_exp > 0.001:
                                camera.set_exposure_time(new_exp)
                                logger.debug("Lighting adj: dev=%.1f  exp %.0f→%.0f us",
                                             avg_dev, cur_exp, new_exp)
                except Exception:
                    pass

                time.sleep(0.15)

        lr_thread = threading.Thread(target=_lighting_robust_loop, daemon=True)
        lr_thread.start()
        logger.info("Lighting robustness thread started (gain-first then exp, deadband=%d)", LR_DEADBAND)

    # --- 细粒度运动计时器 ---
    timer = MotionSegmentTimer(robot, robot_cfg, robot_connected, output_dir="data")

    # --- v4.5: 自动对准闭包 ---
    def _read_alignment_context(marker_type):
        """读取一帧对准上下文（ArUco 误差 + 目标位姿），返回 dict 或 None。"""
        with det_lock:
            result = latest_det
        mid = insert_marker_id if marker_type == "插枪" else takegun_marker_id
        if mid not in result:
            return None
        td = result[mid]
        ref = T_aruco2cam_ref_insert if marker_type == "插枪" else T_aruco2cam_ref_takegun
        if ref is None:
            return None
        T_cur = aruco_to_matrix(td)
        xyz, norm, rot = compute_pose_error_in_frame(T_cur, ref)
        cur_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
        if cur_tcp is None:
            return None
        T_g2b_now = pose_to_matrix(cur_tcp)
        T_target = compute_new_tool_pose(
            base_from_tool_old=T_g2b_now,
            cam_from_target_old=T_cur,
            tool_from_cam=T_c2g,
            cam_from_target_new=ref,
        )
        t_err, r_err = compute_pose_error(T_g2b_now, T_target)
        return {
            "aruco_norm": norm,
            "aruco_rot": rot,
            "trans_err": t_err,
            "rot_err": r_err,
            "target_pose": matrix_to_pose(T_target),
        }

    def _wait_alignment_context(marker_type):
        """等待画面稳定后返回对准上下文，超时返回 None。"""
        time.sleep(args.align_settle_s)
        samples = []
        t0 = time.time()
        while len(samples) < args.align_stable_samples:
            if time.time() - t0 > args.align_stable_timeout_s:
                logger.debug("Stable wait timeout (%d/%d samples)",
                             len(samples), args.align_stable_samples)
                break
            ctx = _read_alignment_context(marker_type)
            if ctx is not None:
                samples.append(ctx)
            time.sleep(0.05)
        if not samples:
            return None
        chosen, spread = pick_stable_alignment_context(
            samples, args.align_stable_spread_mm)
        if chosen is None:
            logger.debug("Alignment unstable: spread=%.2f mm > %.2f mm",
                         spread or 0, args.align_stable_spread_mm)
            return samples[len(samples) // 2]
        logger.debug("Stable alignment: spread=%.2f mm, n=%d", spread, len(samples))
        return chosen

    # 双目相机及检测模型初始化 | Stereo cover pose estimator
    cover_pose_estimator = CoverPoseEstimator()

    # ── 自动化步骤函数 (Auto Step Functions) ──

    def _check_abort():
        """检查是否有用户中止按键 (q/ESC)，有则返回 True。"""
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            logger.info("Auto flow aborted by user")
            return True
        return False

    def _execute_auto_align(marker_label):
        """执行一次完整的 ArUco 自动对准循环。返回 True 表示对准成功。"""
        nonlocal moving
        with det_lock:
            result = latest_det
        # 根据传入的 marker_label 优先匹配对应 marker
        detected_type = None
        ref = None
        if marker_label == "取枪":
            if takegun_marker_id in result:
                detected_type = "取枪"
                ref = T_aruco2cam_ref_takegun
            elif insert_marker_id in result:
                detected_type = "插枪"
                ref = T_aruco2cam_ref_insert
        else:
            if insert_marker_id in result:
                detected_type = "插枪"
                ref = T_aruco2cam_ref_insert
            elif takegun_marker_id in result:
                detected_type = "取枪"
                ref = T_aruco2cam_ref_takegun
        if detected_type is None:
            logger.warning("Auto-align: no marker detected")
            return False

        if ref is None:
            logger.warning("Auto-align: reference not set for %s", detected_type)
            return False

        moving = True
        align_success = False
        logger.info("=== 自动对准 %s (max=%d, success<%.1fmm, min=%d) ===",
                    detected_type, args.align_max_attempts,
                    args.align_success_mm, args.align_min_attempts)

        for attempt in range(1, args.align_max_attempts + 1):
            if _check_abort():
                moving = False
                return False
            ctx = _wait_alignment_context(detected_type)
            if ctx is None:
                logger.warning("对准 #%d: 未检测到 marker 或无法稳定", attempt)
                break
            logger.info("对准 #%d/%d: ArUco误差=%.2f mm %.2f deg | 运动误差=%.2f mm %.2f deg",
                        attempt, args.align_max_attempts,
                        ctx["aruco_norm"], ctx["aruco_rot"],
                        ctx["trans_err"], ctx["rot_err"])
            if should_finish_alignment(attempt, ctx["aruco_norm"],
                                       args.align_success_mm, args.align_min_attempts):
                logger.info("对准成功: 误差 %.2f mm <= %.2f mm (完成 %d 次)",
                            ctx["aruco_norm"], args.align_success_mm, attempt)
                align_success = True
                break
            safe, reason = check_oneshot_safety(ctx["trans_err"], ctx["rot_err"],
                                                args.max_trans, args.max_rot)
            if not safe:
                logger.warning("对准 #%d 安全检查失败: %s", attempt, reason)
                break
            if args.no_robot:
                logger.info("[DRY RUN] 跳过对准运动 #%d", attempt)
                align_success = True
                break
            with timer.segment("对准%s #%d/%d" % (detected_type, attempt, args.align_max_attempts)):
                ok = execute_move(robot, ctx["target_pose"], timeout=args.move_timeout)
            if not ok:
                logger.warning("对准 #%d 运动失败", attempt)
                break
            new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
            if new_tcp is not None:
                logger.info("  对准 #%d 运动后 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            attempt, *new_tcp)

        if not align_success:
            ctx = _wait_alignment_context(detected_type)
            if ctx is not None and ctx["aruco_norm"] <= args.align_success_mm:
                logger.info("最终检查通过: 误差 %.2f mm <= %.2f mm",
                            ctx["aruco_norm"], args.align_success_mm)
                align_success = True

        moving = False
        logger.info("=== 自动对准结束: %s ===", "成功" if align_success else "未达标")

        if robot_connected:
            new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
            if new_tcp is not None:
                T_g2b_after = pose_to_matrix(new_tcp)
                ref_for_log = T_g2b_ref_insert if detected_type == "插枪" else T_g2b_ref_takegun
                if ref_for_log is not None:
                    ref_xyz, ref_res_t, ref_res_r = compute_pose_error_in_frame(T_g2b_after, ref_for_log)
                    logger.info("%s residual (vs ref): trans=%.2f mm, rot=%.2f deg",
                                detected_type, ref_res_t, ref_res_r)
                    logger.info("  ref axes: dX=%.2f dY=%.2f dZ=%.2f mm", *ref_xyz)
                logger.info("  Actual TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *new_tcp)

        return align_success

    def _step_1_init_cover():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_1")
        logger.info("=== STATE_1: 初始关节运动 + 双目粗定位 ===")
        if robot_connected:
            # 夹爪机械臂异步运动
            with timer.segment("夹爪(45)+机械臂(INIT_JOINT)(异步)"):
                async def parallel_task():
                    await asyncio.gather(to_thread(flow.gripper.set_position, 45),
                                    to_thread(robot.move_joint, INIT_JOINT, 60))
                asyncio.run(parallel_task())

            cover_3D_pose = cover_pose_estimator.pose_estimation()
            if cover_3D_pose is not None:
                logger.info("Cover 3D pose: %s", cover_3D_pose)
                cover_3D_matrix = pose_to_matrix(cover_3D_pose)
                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_tool_pose is None:
                    logger.error("STATE_1 失败: 无法读取 TCP")
                    return AutoState.IDLE
                cur_tcp_matrix = pose_to_matrix(current_tool_pose)
                cover_offset_pose = STEREO_DETECT_OFFSET_POSE
                cover_offset_matrix = pose_to_matrix(cover_offset_pose)
                target_matrix = cur_tcp_matrix @ (cover_3D_matrix @ cover_offset_matrix)
                target_pose = matrix_to_pose(target_matrix)
                logger.info("  Target TCP for coarse alignment: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *target_pose)
                # 参考关节角（某个停车位置）
                ref_joint = STEREO_IK_REF_JOINT
                cur_joint = robot.get_joint_pose()
                target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=ref_joint)
                with timer.segment("双目粗定位运动"):
                    robot.move_by_joint_list(joints=[target_joint], speeds=[60])
                # robot.move_linear(CartesianPose(*target_pose))
            else:
                logger.error("STATE_1 失败: 双目姿态估计返回 None")
                return AutoState.IDLE
        else:
            logger.info("[DRY RUN] STATE_1 skipped")
        logger.info("=== STATE_1 完成 ===")
        return AutoState.STATE_2_FORCE_OPEN

    def _step_2_force_open():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_2")
        logger.info("=== STATE_2: 力控按压开盖 ===")
        if robot_connected:
            with timer.segment("力控按压开盖"):
                force_ctrl.run_ForceControl_OpenCover()
        logger.info("=== STATE_2 完成 ===")
        return AutoState.STATE_3_COVER_ACTION

    def _step_3_cover_action():
        nonlocal outer_cover_is_opened
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_3")
        logger.info("=== STATE_3: 拨盖运动 ===")
        if not outer_cover_is_opened:
            flow.run(1, timer=timer)
        else:
            flow.run(8, timer=timer)
        logger.info("=== STATE_3 完成 ===")
        return AutoState.STATE_4_ALIGN_INSERT

    def _step_4_align_insert():
        nonlocal already_return_gun
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_4")
        logger.info("=== STATE_4: 自动对准插枪 ArUco ===")
        if robot_connected:
            success = _execute_auto_align("插枪")
            if not success:
                logger.error("STATE_4 失败: 自动对准未达标")
                return AutoState.IDLE
            if not already_return_gun:
                current_tcp = robot.get_tcp_pose()
                current_matrix = pose_to_matrix(current_tcp)
                offset_pose = INSERT_BEFORE_OFFSET_POSE
                offset_matrix = pose_to_matrix(offset_pose)
                target_matrix = current_matrix @ offset_matrix
                target_pose = matrix_to_pose(target_matrix)
                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data["INSERT_BEFORE_TEMPLATE"] = target_pose
                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                flow.update_CFG()
        logger.info("=== STATE_4 完成 ===")
        return AutoState.STATE_5_OFFSET_B

    def _step_5_offset_b():
        nonlocal already_take_inner_cover
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_5")
        logger.info("=== STATE_5: 对准后固定偏移运动 ===")
        if robot_connected:
            current_pose = robot.get_tcp_pose()
            current_matrix = pose_to_matrix(current_pose)
            offset_pose_cover = INNER_COVER_OFFSET_POSE
            offset_matrix = pose_to_matrix(offset_pose_cover)
            target_matrix = current_matrix @ offset_matrix
            target_pose = matrix_to_pose(target_matrix)
            take_cover_pose = robot.relative_tool_pose(dz=50, init_pose=target_pose).to_list()
            current_joint = robot.get_joint_pose()
            target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=current_joint)
            take_cover_joint = robot.inverse_kinematics(target_pose=take_cover_pose, initial_joints=current_joint)
            with timer.segment("固定偏移运动(两点轨迹)"):
                robot.move_by_joint_list(joints=[target_joint, take_cover_joint], speeds=[25, 15])
            if not already_take_inner_cover:
                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data["PUSH_POINT1"] = target_pose
                    data["PUSH_POINT2"] = take_cover_pose
                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                flow.update_CFG()
                logger.info('PUSH POINT1 POINT2已写入json.')
        logger.info("=== STATE_5 完成 ===")
        return AutoState.STATE_6_GRAB_INNER

    def _step_6_grab_inner():
        nonlocal already_take_inner_cover
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_6")
        logger.info("=== STATE_6: 夹住小盖并放小盖 → 运动到取枪初始点 ===")
        flow.run(2, timer=timer)
        already_take_inner_cover = True
        logger.info("=== STATE_6 完成 ===")
        return AutoState.STATE_7_ALIGN_TAKE

    def _step_7_align_take():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_7")
        logger.info("=== STATE_7: 自动对准取枪 ArUco ===")
        if robot_connected:
            time.sleep(0.2)
            success = _execute_auto_align("取枪")
            if not success:
                logger.error("STATE_7 失败: 自动对准未达标")
                return AutoState.IDLE
        logger.info("=== STATE_7 完成 ===")
        return AutoState.STATE_8_OFFSET_A

    def _step_8_offset_a():
        nonlocal moving
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_8")
        logger.info("=== STATE_8: 固定偏移 + 沿法兰z前进 + 夹爪舵机 ===")
        if robot_connected:
            current_pose = get_tcp_pose_in_tool_mm(
                robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            if current_pose is None:
                logger.error("STATE_8 失败: 无法读取 TCP")
                return AutoState.IDLE
            current_matrix = pose_to_matrix(current_pose)
            offset_pose = TAKEGUN_OFFSET_POSE
            offset_matrix = pose_to_matrix(offset_pose)
            target_matrix = current_matrix @ offset_matrix
            target_pose = matrix_to_pose(target_matrix)
            logger.info("Applying relative offset:")
            logger.info("  Current:  X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *current_pose)
            logger.info("  Target:   X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *target_pose)
            if not args.no_robot:
                moving = True
                motion_success = True
                robot.set_speed(100)
                cart = CartesianPose(
                    x=target_pose[0], y=target_pose[1], z=target_pose[2],
                    rx=target_pose[3], ry=target_pose[4], rz=target_pose[5],
                )
                with timer.segment("固定偏移运动"):
                    ok = robot.move_joint_and_wait(cart, speed=50, timeout=args.move_timeout)
                if ok:
                    logger.info("Relative offset move complete")
                    current_tool_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if current_tool_pose is not None:
                        advance_pose = offset_pose_along_tool_axis(current_tool_pose, 121.0, axis='z')
                        logger.info("Take gun: advance 121 mm along tool Z-axis")
                        logger.info("  Target: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *advance_pose)
                        robot.set_speed(50) # 直线插入运动速度
                        with timer.segment("沿法兰z前进121mm"):
                            ok = robot.move_and_wait(CartesianPose(
                                x=advance_pose[0], y=advance_pose[1], z=advance_pose[2],
                                rx=advance_pose[3], ry=advance_pose[4], rz=advance_pose[5],
                            ), timeout=args.move_timeout)
                        if ok:
                            logger.info("Take-gun advance 121mm complete")
                        else:
                            logger.warning("Take-gun advance 121mm failed")
                            motion_success = False
                    else:
                        logger.warning("Failed to read TCP for 120mm advance")
                        motion_success = False
                else:
                    logger.warning("Relative offset move timed out or failed")
                    motion_success = False
                moving = False
                final_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                if final_pose is not None:
                    logger.info("  Final: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *final_pose)
                robot.set_speed(DEFAULT_SPEED)
                if motion_success:
                    while robot.is_moving():
                        time.sleep(0.1)
                    logger.info("Auto-trigger gripper close + servo press")
                    with timer.segment("夹爪闭合"):
                        flow.gripper.set_speed(30)
                        flow.gripper.set_force(100)
                        flow.gripper.set_position(32)
                        timeout_s = 5.0
                        t0 = time.time()
                        while True:
                            status = flow.gripper.get_grip_status()
                            if status in (1, 2):
                                logger.info("Gripper action complete (status=%d)", status)
                                break
                            if time.time() - t0 > timeout_s:
                                logger.warning("Gripper action timeout after %.1fs (status=%s)", timeout_s, status)
                                break
                            time.sleep(0.1)
                    with timer.segment("舵机按压"):
                        arm_controller = SMSSTSController("/dev/ttysWK1")
                        arm_controller.connect()
                        arm_controller.press_trigger()
                        arm_controller.disconnect()
                else:
                    logger.error("STATE_8 失败: 运动失败")
                    return AutoState.IDLE
        logger.info("=== STATE_8 完成 ===")
        return AutoState.STATE_9_RETRACT_C

    def _step_9_retract_c():
        nonlocal moving
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_9")
        logger.info("=== STATE_9: 沿法兰z退出 + 复位舵机 ===")
        if robot_connected:
            current_tool_pose = get_tcp_pose_in_tool_mm(
                robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            if current_tool_pose is None:
                logger.error("STATE_9 失败: 无法读取 TCP")
                return AutoState.IDLE
            before_retract_joint = robot.get_joint_pose()
            retract_pose = offset_pose_along_tool_axis(current_tool_pose, -100.0, axis='z')
            retract_joint = robot.inverse_kinematics(target_pose=retract_pose, initial_joints=before_retract_joint)
            logger.info("Retract 100 mm along tool Z-axis")
            logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                        *current_tool_pose)
            logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                        *retract_pose)
            if not args.no_robot:
                moving = True
                motion_success = True
                robot.set_speed(100)
                with timer.segment("沿法兰z退出100mm"):
                    ok = execute_move(robot, retract_pose, timeout=args.move_timeout)
                if ok:
                    logger.info("Retract 100mm complete")
                else:
                    logger.warning("Retract 100mm failed")
                    motion_success = False
                moving = False
                try:
                    tcp_after = get_robot_tcp_pose_mm(robot, robot_cfg)
                except Exception:
                    tcp_after = None
                if tcp_after is not None:
                    logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                *tcp_after)
                robot.set_speed(DEFAULT_SPEED)
        logger.info("=== STATE_9 完成 ===")
        return AutoState.STATE_10_PRE_INSERT

    def _step_10_pre_insert():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_10")
        logger.info("=== STATE_10: 插枪前运动 ===")
        logger.info("Auto-trigger servo reset")
        flow.run(3, timer=timer)
        logger.info("=== STATE_10 完成 ===")
        return AutoState.STATE_11_FORCE_IN

    def _step_11_force_in():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_11")
        logger.info("=== STATE_11: 力控插枪 ===")
        with timer.segment("力控插枪"):
            force_ctrl.run_forcecontrol_charge_in()
        logger.info("=== STATE_11 完成 ===")
        return AutoState.STATE_11B_PARK_RETURN

    def _step_11b_park_return():
        nonlocal moving
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_11B")
        logger.info("=== STATE_11B: 开夹爪→退150mm→记驻停位姿→去FINALL_POINT→等12s→回逼近位姿→视觉对准→STATE_8取枪逻辑 ===")
        if robot_connected:
            # 力控插枪后当前 TCP (tool 0 = 法兰)
            force_in_tcp = get_tcp_pose_in_tool_mm(
                robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            if force_in_tcp is None:
                logger.error("STATE_11B 失败: 无法读取 TCP")
                return AutoState.IDLE
            # 1) 开夹爪 (释放枪)
            with timer.segment("开夹爪"):
                flow.gripper.open()
                timeout_s = 5.0
                t0 = time.time()
                while True:
                    status = flow.gripper.get_grip_status()
                    if status in (1,):
                        logger.info("夹爪打开完成 (status=%d)", status)
                        break
                    if time.time() - t0 > timeout_s:
                        logger.warning("夹爪打开超时 %.1fs (status=%s)", timeout_s, status)
                        break
                    time.sleep(0.1)
            moving = True
            # 2) 沿法兰 z 轴退出 150 mm (关节运动)
            exit_pose = offset_pose_along_tool_axis(force_in_tcp, -150.0, axis='z')
            cur_joint = robot.get_joint_pose()
            exit_joint = robot.inverse_kinematics(target_pose=exit_pose, initial_joints=cur_joint)
            logger.info("沿法兰z退出150mm: 目标 X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *exit_pose)
            with timer.segment("沿法兰z退出150mm"):
                robot.move_by_joint_list(joints=[exit_joint], speeds=[30])
            # 3) 记录驻停位姿 (关节角 + 笛卡尔)
            parked_joint = robot.get_joint_pose()
            parked_cart = get_tcp_pose_in_tool_mm(
                robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            logger.info("驻停位姿已记录: joint=%s", parked_joint)
            if parked_cart is not None:
                logger.info("  驻停位姿 tcp: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f", *parked_cart)
            # 4) 移动至 FINALL_POINT (关节运动)
            final_joint = flow.CFG["FINALL_POINT"]
            logger.info("移动至 FINALL_POINT: %s", final_joint)
            with timer.segment("移动至FINALL_POINT"):
                robot.move_by_joint_list(joints=[final_joint], speeds=[60])
            moving = False
            # 5) 充电等待 12 秒
            logger.info("=== 充电等待 12s ===")
            time.sleep(12)
            moving = True
            # 6) 计算取枪逼近位姿 (驻停位姿 ) 并用关节运动回去
            if parked_cart is None:
                logger.error("STATE_11B 失败: 驻停笛卡尔位姿为空，无法计算逼近位姿")
                return AutoState.IDLE
            parked_matrix = pose_to_matrix(parked_cart)
            parked_offset_matrix = pose_to_matrix(PARKED_OFFSET)
            takegun_approach_matrix = parked_matrix @ parked_offset_matrix
            takegun_approach_cart = matrix_to_pose(takegun_approach_matrix)
            cur_joint = robot.get_joint_pose()
            takegun_approach_joint = robot.inverse_kinematics(
                target_pose=takegun_approach_cart, initial_joints=cur_joint)
            logger.info("返回取枪逼近位姿: 目标 X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                        *takegun_approach_cart)
            with timer.segment("返回取枪逼近位姿"):
                robot.move_by_joint_list(joints=[takegun_approach_joint], speeds=[60])
            moving = False
            # 6.5) 自动对准取枪 ArUco (同 STATE_7)
            time.sleep(0.2)
            logger.info("=== STATE_11B 内自动对准取枪 ArUco ===")
            success = _execute_auto_align("取枪")
            if not success:
                logger.error("STATE_11B 失败: 取枪视觉对准未达标")
                return AutoState.IDLE
            # 7) 执行与 STATE_8 相同的取枪逻辑：固定偏移 + 沿法兰z前进 + 夹爪舵机
            current_pose = get_tcp_pose_in_tool_mm(
                robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            if current_pose is None:
                logger.error("STATE_11B 失败: 无法读取 TCP")
                return AutoState.IDLE
            current_matrix = pose_to_matrix(current_pose)
            offset_pose = TAKEGUN_OFFSET_POSE
            offset_matrix = pose_to_matrix(offset_pose)
            target_matrix = current_matrix @ offset_matrix
            target_pose = matrix_to_pose(target_matrix)
            logger.info("Applying relative offset:")
            logger.info("  Current:  X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *current_pose)
            logger.info("  Target:   X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *target_pose)
            if not args.no_robot:
                moving = True
                motion_success = True
                robot.set_speed(100)
                cart = CartesianPose(
                    x=target_pose[0], y=target_pose[1], z=target_pose[2],
                    rx=target_pose[3], ry=target_pose[4], rz=target_pose[5],
                )
                with timer.segment("固定偏移运动"):
                    ok = robot.move_joint_and_wait(cart, speed=50, timeout=args.move_timeout)
                if ok:
                    logger.info("Relative offset move complete")
                    current_tool_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if current_tool_pose is not None:
                        advance_pose = offset_pose_along_tool_axis(current_tool_pose, 121.0, axis='z')
                        logger.info("Take gun: advance 121 mm along tool Z-axis")
                        logger.info("  Target: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *advance_pose)
                        robot.set_speed(50)
                        with timer.segment("沿法兰z前进121mm"):
                            ok = robot.move_and_wait(CartesianPose(
                                x=advance_pose[0], y=advance_pose[1], z=advance_pose[2],
                                rx=advance_pose[3], ry=advance_pose[4], rz=advance_pose[5],
                            ), timeout=args.move_timeout)
                        if ok:
                            logger.info("Take-gun advance 121mm complete")
                        else:
                            logger.warning("Take-gun advance 121mm failed")
                            motion_success = False
                    else:
                        logger.warning("Failed to read TCP for 121mm advance")
                        motion_success = False
                else:
                    logger.warning("Relative offset move timed out or failed")
                    motion_success = False
                moving = False
                final_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                if final_pose is not None:
                    logger.info("  Final: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *final_pose)
                robot.set_speed(DEFAULT_SPEED)
                if motion_success:
                    while robot.is_moving():
                        time.sleep(0.1)
                    logger.info("Auto-trigger gripper close + servo press")
                    with timer.segment("夹爪闭合"):
                        flow.gripper.set_speed(30)
                        flow.gripper.set_force(100)
                        flow.gripper.set_position(32)
                        timeout_s = 5.0
                        t0 = time.time()
                        while True:
                            status = flow.gripper.get_grip_status()
                            if status in (1, 2):
                                logger.info("Gripper action complete (status=%d)", status)
                                break
                            if time.time() - t0 > timeout_s:
                                logger.warning("Gripper action timeout after %.1fs (status=%s)", timeout_s, status)
                                break
                            time.sleep(0.1)
                    with timer.segment("舵机按压"):
                        arm_controller = SMSSTSController("/dev/ttysWK1")
                        arm_controller.connect()
                        arm_controller.press_trigger()
                        arm_controller.disconnect()
                else:
                    logger.error("STATE_11B 失败: 运动失败")
                    return AutoState.IDLE
        else:
            logger.info("[DRY RUN] STATE_11B skipped")
        logger.info("=== STATE_11B 完成 ===")
        return AutoState.STATE_12_FORCE_OUT

    def _step_12_force_out():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_12")
        logger.info("=== STATE_12: 力控拔枪 ===")
        if robot_connected:
            with timer.segment("力控拔枪"):
                force_ctrl.run_forcecontrol_charge_out()
        logger.info("=== STATE_12 完成 ===")
        return AutoState.STATE_13_RETURN_GUN

    def _step_13_return_gun():
        nonlocal already_return_gun
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_13")
        logger.info("=== STATE_13: 归枪运动 ===")
        flow.run(4, timer=timer)
        already_return_gun = True
        logger.info("=== STATE_13 完成 ===")
        return AutoState.STATE_14_ALIGN_POINT

    def _step_14_align_point():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_14")
        logger.info("=== STATE_14: 移动至取小盖前点位 ===")
        flow.run(5, timer=timer)
        logger.info("=== STATE_14 完成 ===")
        return AutoState.STATE_15_ALIGN_INNER

    def _step_15_align_inner():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_15")
        logger.info("=== STATE_15: 自动对准取小盖 ArUco ===")
        if robot_connected:
            time.sleep(0.2)
            success = _execute_auto_align("取小盖")
            if not success:
                logger.error("STATE_15 失败: 自动对准未达标")
                return AutoState.IDLE
        logger.info("=== STATE_15 完成 ===")
        return AutoState.STATE_16_OFFSET_B2

    def _step_16_offset_b2():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_16")
        logger.info("=== STATE_16: 对准后固定偏移运动 ===")
        if robot_connected:
            current_pose = robot.get_tcp_pose()
            current_matrix = pose_to_matrix(current_pose)
            offset_pose_cover = INNER_COVER_OFFSET_POSE
            offset_matrix = pose_to_matrix(offset_pose_cover)
            target_matrix = current_matrix @ offset_matrix
            target_pose = matrix_to_pose(target_matrix)
            take_cover_pose = robot.relative_tool_pose(dz=50, init_pose=target_pose).to_list()
            current_joint = robot.get_joint_pose()
            target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=current_joint)
            take_cover_joint = robot.inverse_kinematics(target_pose=take_cover_pose, initial_joints=current_joint)
            with timer.segment("固定偏移运动(两点轨迹)"):
                robot.move_by_joint_list(joints=[target_joint, take_cover_joint], speeds=[30, 20])
        logger.info("=== STATE_16 完成 ===")
        return AutoState.STATE_17_CLOSE_COVER

    def _step_17_close_cover():
        if _check_abort(): return AutoState.IDLE
        timer.set_step("STATE_17")
        logger.info("=== STATE_17: 夹住小盖放回充电口 + 关大盖 → 回到初始点 ===")
        flow.run(7, timer=timer)
        flow.run(6, timer=timer)
        logger.info("=== STATE_17 完成 === 全流程结束")
        return AutoState.IDLE

    _STEP_DISPATCH = {
        AutoState.STATE_1_INIT_COVER: _step_1_init_cover,
        AutoState.STATE_2_FORCE_OPEN: _step_2_force_open,
        AutoState.STATE_3_COVER_ACTION: _step_3_cover_action,
        AutoState.STATE_4_ALIGN_INSERT: _step_4_align_insert,
        AutoState.STATE_5_OFFSET_B: _step_5_offset_b,
        AutoState.STATE_6_GRAB_INNER: _step_6_grab_inner,
        AutoState.STATE_7_ALIGN_TAKE: _step_7_align_take,
        AutoState.STATE_8_OFFSET_A: _step_8_offset_a,
        AutoState.STATE_9_RETRACT_C: _step_9_retract_c,
        AutoState.STATE_10_PRE_INSERT: _step_10_pre_insert,
        AutoState.STATE_11_FORCE_IN: _step_11_force_in,
        AutoState.STATE_11B_PARK_RETURN: _step_11b_park_return,
        AutoState.STATE_12_FORCE_OUT: _step_12_force_out,
        AutoState.STATE_13_RETURN_GUN: _step_13_return_gun,
        AutoState.STATE_14_ALIGN_POINT: _step_14_align_point,
        AutoState.STATE_15_ALIGN_INNER: _step_15_align_inner,
        AutoState.STATE_16_OFFSET_B2: _step_16_offset_b2,
        AutoState.STATE_17_CLOSE_COVER: _step_17_close_cover,
    }


    def _execute_auto_step(state):
        fn = _STEP_DISPATCH.get(state)
        if fn is None:
            logger.error("Unknown auto state: %s", state)
            return AutoState.IDLE
        _step_t0 = time.time()
        next_state = fn()
        _step_dt = time.time() - _step_t0
        logger.info("[计时] %s 耗时 %.3f 秒", state.name, _step_dt)
        return next_state

    try:
        _start_state = AutoState.STATE_1_INIT_COVER
        if debug_start_step is not None:
            if 1 <= debug_start_step <= len(_STEP_EXEC_ORDER):
                _start_state = _STEP_EXEC_ORDER[debug_start_step - 1]
                logger.info("[STEP MODE] 从第 %d 步 %s 开始执行",
                            debug_start_step, _start_state.name)
            else:
                logger.warning("--debug-start-step %d 无效（有效范围 1-%d），从第1步开始",
                               debug_start_step, len(_STEP_EXEC_ORDER))
        fullworkflow = _step_1_init_cover
        first_move = 0 # 随动标志位
        auto_state = AutoState.IDLE
        auto_flow_start_time = None  # 本轮自动流程的起始时间（从触发信号算起）
        while True:
            with open('/home/nvidia/Downloads/HD/HD_0323/Intergration/stereo_camera_calib/yrq/first_move.json', 'r', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                first_move = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
                
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            with det_lock:
                aruco_result = latest_det
                det_frame = latest_det_frame
            # 检测当前有效的 marker (优先插枪，其次取枪)
            current_marker_id = None
            current_marker_type = None
            target_data = None
            
            if insert_marker_id in aruco_result:
                current_marker_id = insert_marker_id
                current_marker_type = "插枪"
                target_data = aruco_result[insert_marker_id]
            elif takegun_marker_id in aruco_result:
                current_marker_id = takegun_marker_id
                current_marker_type = "取枪"
                target_data = aruco_result[takegun_marker_id]

            tcp = None
            T_g2b_cur = None
            if robot_connected:
                tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                if tcp is not None:
                    T_g2b_cur = pose_to_matrix(tcp)

            # Scale overlay TCP to meters for display
            overlay_tcp = None
            if tcp is not None:
                overlay_tcp = [tcp[0] / 1000.0, tcp[1] / 1000.0, tcp[2] / 1000.0, *tcp[3:]]
            vis = draw_aruco_result_for_display(
                frame, aruco_result, intrinsics,
                robot_pose=overlay_tcp,
                use_kalman=(not use_raw and use_temporal_filter),
                robot_connected=robot_connected,
                max_width=DISPLAY_MAX_WIDTH,
                max_height=DISPLAY_MAX_HEIGHT,
            )
            h_disp, w_disp = vis.shape[:2]
            y = 35 + len(aruco_result) * 60 + 40

            # 显示当前检测到的marker类型
            if current_marker_id is None:
                put_text(vis, f"Insert ID{insert_marker_id} / Take ID{takegun_marker_id} Marker not detected", y, (0, 0, 255))
                y += 40
            else:
                current_marker_label = marker_display_name(current_marker_type)
                put_text(vis, f"Detected: {current_marker_label} (ID{current_marker_id})", y, (0, 255, 0))
                y += 40

            # Select reference poses based on detected marker type
            T_aruco2cam_ref = None
            T_g2b_ref = None
            if current_marker_type == "插枪":
                T_aruco2cam_ref = T_aruco2cam_ref_insert
                T_g2b_ref = T_g2b_ref_insert
            elif current_marker_type == "取枪":
                T_aruco2cam_ref = T_aruco2cam_ref_takegun
                T_g2b_ref = T_g2b_ref_takegun

            T_g2b_target = None
            trans_err = None
            rot_err = None
            target_pose = None

            if T_aruco2cam_ref is not None and target_data is not None:
                T_aruco2cam_cur = aruco_to_matrix(target_data)
                if T_g2b_cur is not None:
                    T_g2b_target = compute_new_tool_pose(
                        base_from_tool_old=T_g2b_cur,
                        cam_from_target_old=T_aruco2cam_cur,
                        tool_from_cam=T_c2g,
                        cam_from_target_new=T_aruco2cam_ref,
                    )
                    trans_err, rot_err = compute_pose_error(T_g2b_cur, T_g2b_target)
                    target_pose = matrix_to_pose(T_g2b_target)

            # ArUco error: current detection vs reference (independent of robot connection)
            aruco_trans_xyz = None
            aruco_trans_norm = None
            aruco_rot_err = None
            if T_aruco2cam_ref is not None and target_data is not None:
                T_aruco2cam_cur = aruco_to_matrix(target_data)
                aruco_trans_xyz, aruco_trans_norm, aruco_rot_err = compute_pose_error_in_frame(
                    T_aruco2cam_cur, T_aruco2cam_ref)

            if aruco_trans_norm is not None:
                if aruco_trans_norm < 2.0:
                    err_color = (0, 255, 0)
                elif aruco_trans_norm < 10.0:
                    err_color = (0, 200, 255)
                else:
                    err_color = (0, 0, 255)
                current_marker_label = marker_display_name(current_marker_type)
                put_text(vis, f"{current_marker_label} ArUco Error: {aruco_trans_norm:.2f} mm  {aruco_rot_err:.2f} deg", y, err_color)
                y += 30
                put_text(vis, f"  dX={aruco_trans_xyz[0]:+.2f}  dY={aruco_trans_xyz[1]:+.2f}  dZ={aruco_trans_xyz[2]:+.2f} mm",
                         y, err_color)
                y += 28

                if target_pose is not None:
                    put_text(vis, f"{current_marker_label} Base Target(0): X={target_pose[0]:.1f} Y={target_pose[1]:.1f} Z={target_pose[2]:.1f}",
                             y, (255, 200, 0))
                    y += 28
                    put_text(vis, f"                Rx={target_pose[3]:.2f} Ry={target_pose[4]:.2f} Rz={target_pose[5]:.2f}",
                             y, (255, 200, 0))
                    y += 30
            elif current_marker_id is not None:
                current_marker_label = marker_display_name(current_marker_type)
                put_text(vis, f"Press r at {current_marker_label} ref position to set reference", y, PROMPT_TEXT_COLOR)
                y += 30

            # Display insert distance info
            if current_marker_type is not None:
                insert_cm = args.insert_cm_insert if current_marker_type == "插枪" else args.insert_cm_take
                put_text(vis, f"{current_marker_type} insert distance: {insert_cm} cm", y, PROMPT_TEXT_COLOR)
                y += 28

            if moving:
                put_text(vis, "MOVING...", y, (0, 100, 255))
                y += 30

            roi_bri = _marker_roi_brightness(frame, target_data) if target_data is not None else None
            roi_str = f"ROI Bri: {roi_bri:.0f}" if roi_bri is not None else "ROI Bri: None"
            roi_color = (0, 255, 128) if roi_bri is not None else (100, 100, 100)
            put_text(vis, roi_str, y, roi_color)
            y += 30

            # 参考亮度对比（Bri Ref）
            if roi_bri is not None and current_marker_type is not None:
                ref_bri = ref_bri_insert if current_marker_type == "插枪" else ref_bri_takegun
                if ref_bri is not None:
                    bri_ref_str = f"Bri Ref: {ref_bri:.0f} d={roi_bri - ref_bri:+.0f}"
                    put_text(vis, bri_ref_str, y, (0, 200, 100))
                    y += 30

            try:
                exp = camera.get_exposure_time()
                gain_db = camera.get_gain()
                if exp is not None:
                    exp_str = f"Exposure: {exp/1000:.2f}ms  Gain: {gain_db:.1f}dB" if gain_db is not None else f"Exposure: {exp/1000:.2f}ms"
                    put_text(vis, exp_str, y, (180, 180, 180))
                    y += 30
            except Exception:
                pass

            lr_flag = " LR" if args.lighting_robust else ""
            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            ref_str = f"Ref:Insert{'SET' if ref_set_insert else 'NONE'} Take{'SET' if ref_set_takegun else 'NONE'}"
            auto_state_str = f"State:{auto_state.name}" if auto_state != AutoState.IDLE else ""
            status = (f"V4.7 AUTO{lr_flag} | {ref_str} | {robot_str} | {auto_state_str} | "
                      "f=Run r=Ref q=Quit")
            put_text(vis, status, h_disp - 15, (140, 140, 140))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if first_move == 1:
                robot.move_joint(INIT_JOINT, speed=50)

                first_move = 0
                with open('/home/nvidia/Downloads/HD/HD_0323/Intergration/stereo_camera_calib/yrq/first_move.json', 'w', encoding='utf-8') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(0, f)
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f, fcntl.LOCK_UN)

            # ── 自动化流程状态机 | Auto Flow State Machine ──
            if auto_state > AutoState.IDLE:
                if key in (ord('q'), 27):
                    logger.info("Auto flow aborted by user")
                    moving = False
                    auto_state = AutoState.IDLE
                    if auto_flow_start_time is not None:
                        logger.info("[计时] 自动流程总耗时（从触发到中止）: %.3f 秒",
                                    time.time() - auto_flow_start_time)
                        auto_flow_start_time = None
                    timer.export_txt()
                    continue
                prev_state = auto_state
                auto_state = _execute_auto_step(auto_state)
                if auto_state == AutoState.IDLE:
                    if auto_flow_start_time is not None:
                        logger.info("[计时] 自动流程总耗时（从触发到17步全部结束）: %.3f 秒",
                                    time.time() - auto_flow_start_time)
                        auto_flow_start_time = None
                    timer.export_txt()
                    logger.info("Auto flow terminated (back to IDLE)")
                elif step_mode:
                    # 单步模式：暂停等待用户确认
                    logger.info("[STEP MODE] %s 完成, 下一步: %s | ENTER=继续 q=退出",
                                prev_state.name, auto_state.name)
                    while True:
                        ok_p, frame_p = camera.read_frame()
                        if ok_p and frame_p is not None:
                            vis_p = frame_p.copy()
                            cv2.putText(vis_p, "[STEP MODE] Step done, next: %s" % auto_state.name,
                                        (20, 60), _FONT, 0.7, (0, 255, 255), 2)
                            cv2.putText(vis_p, "  ENTER / SPACE = continue   q = quit",
                                        (20, 95), _FONT, 0.6, (0, 255, 255), 2)
                            cv2.imshow(win, vis_p)
                        key_p = cv2.waitKey(30) & 0xFF
                        if key_p in (13, 32):
                            break
                        if key_p in (ord('q'), 27):
                            logger.info("[STEP MODE] Aborted by user")
                            auto_state = AutoState.IDLE
                            moving = False
                            break
                continue

            # ── IDLE 模式: UDS 触发 + 手动按键 ──
            if key in (ord('q'), 27):
                break

            # 检查 UDS 自动触发信号
            if auto_trigger_event.is_set():
                auto_trigger_event.clear()
                auto_state = _start_state
                auto_flow_start_time = auto_trigger_time if auto_trigger_time is not None else time.time()
                logger.info("Auto flow triggered by UDS signal")
                continue

            # 手动启动全流程（等效 UDS 触发，从 STATE_1 顺序执行 17 步）
            if key == ord('f'):
                auto_state = _start_state
                auto_flow_start_time = time.time()
                logger.info("Auto flow started by manual key 'f'")
                continue

            # --- 手动按键（IDLE 状态下） | Manual Keys ---
            if key == ord('r'):
                if current_marker_id is None:
                    logger.warning("No marker detected, cannot save reference")
                    continue

                if current_marker_type == "插枪":
                    T_aruco2cam_ref_insert = aruco_to_matrix(target_data)
                    ref_pose_vec = matrix_to_pose(T_aruco2cam_ref_insert)
                    os.makedirs(os.path.dirname(args.aruco_ref) or '.', exist_ok=True)
                    save_pose_file(args.aruco_ref, np.array([ref_pose_vec]))
                    ref_set_insert = True
                    logger.info("Insert ref pose saved: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

                    if T_g2b_cur is not None:
                        T_g2b_ref_insert = T_g2b_cur
                        save_pose_file(tcp_ref_path_insert, np.array([matrix_to_pose(T_g2b_cur)]))
                        logger.info("Insert ref TCP (tool %d) saved: t=[%.2f, %.2f, %.2f] mm",
                                    BASE_TOOL_ID, T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])

                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            if bri is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_insert) or '.', exist_ok=True)
                                with open(exp_ref_path_insert, 'w') as f:
                                    f.write(f"{bri:.1f}\n")
                                ref_bri_insert = bri
                                _lr_ref_bris[insert_marker_id] = bri
                                logger.info("Insert bri ref saved: %.0f", bri)
                        except Exception:
                            logger.warning("Failed to save insert bri ref", exc_info=True)
                else:
                    T_aruco2cam_ref_takegun = aruco_to_matrix(target_data)
                    ref_pose_vec = matrix_to_pose(T_aruco2cam_ref_takegun)
                    os.makedirs(os.path.dirname(args.aruco_ref_takegun) or '.', exist_ok=True)
                    save_pose_file(args.aruco_ref_takegun, np.array([ref_pose_vec]))
                    ref_set_takegun = True
                    logger.info("Take ref pose saved: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

                    if T_g2b_cur is not None:
                        T_g2b_ref_takegun = T_g2b_cur
                        save_pose_file(tcp_ref_path_takegun, np.array([matrix_to_pose(T_g2b_cur)]))
                        logger.info("Take ref TCP (tool %d) saved: t=[%.2f, %.2f, %.2f] mm",
                                    BASE_TOOL_ID, T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])

                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            if bri is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_takegun) or '.', exist_ok=True)
                                with open(exp_ref_path_takegun, 'w') as f:
                                    f.write(f"{bri:.1f}\n")
                                ref_bri_takegun = bri
                                _lr_ref_bris[takegun_marker_id] = bri
                                logger.info("Take bri ref saved: %.0f", bri)
                        except Exception:
                            logger.warning("Failed to save take bri ref", exc_info=True)

            elif key == ord('m'):
                # 手动自动多次视觉对准
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if current_marker_type is None:
                    logger.warning("No marker detected, cannot align")
                    continue
                if T_aruco2cam_ref is None:
                    logger.warning("Reference pose not set for %s; press 'r' first",
                                   current_marker_type)
                    continue
                if not robot_connected and not args.no_robot:
                    logger.warning("Robot not connected, cannot align")
                    continue

                moving = True
                align_success = False
                logger.info("=== 开始自动对准 %s (max=%d, success<%.1fmm, min=%d) ===",
                            current_marker_type, args.align_max_attempts,
                            args.align_success_mm, args.align_min_attempts)

                for attempt in range(1, args.align_max_attempts + 1):
                    ctx = _wait_alignment_context(current_marker_type)
                    if ctx is None:
                        logger.warning("对准 #%d: 未检测到 marker 或无法稳定", attempt)
                        break

                    logger.info("对准 #%d/%d: ArUco误差=%.2f mm %.2f deg | 运动误差=%.2f mm %.2f deg",
                                attempt, args.align_max_attempts,
                                ctx["aruco_norm"], ctx["aruco_rot"],
                                ctx["trans_err"], ctx["rot_err"])

                    if should_finish_alignment(attempt, ctx["aruco_norm"],
                                               args.align_success_mm, args.align_min_attempts):
                        logger.info("对准成功: 误差 %.2f mm <= %.2f mm (完成 %d 次)",
                                    ctx["aruco_norm"], args.align_success_mm, attempt)
                        align_success = True
                        break

                    safe, reason = check_oneshot_safety(ctx["trans_err"], ctx["rot_err"],
                                                        args.max_trans, args.max_rot)
                    if not safe:
                        logger.warning("对准 #%d 安全检查失败: %s", attempt, reason)
                        break

                    if args.no_robot:
                        logger.info("[DRY RUN] 跳过对准运动 #%d", attempt)
                        align_success = True
                        break

                    ok = execute_move(robot, ctx["target_pose"], timeout=args.move_timeout)
                    if not ok:
                        logger.warning("对准 #%d 运动失败", attempt)
                        break

                    new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if new_tcp is not None:
                        logger.info("  对准 #%d 运动后 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    attempt, *new_tcp)

                if not align_success:
                    ctx = _wait_alignment_context(current_marker_type)
                    if ctx is not None and ctx["aruco_norm"] <= args.align_success_mm:
                        logger.info("最终检查通过: 误差 %.2f mm <= %.2f mm",
                                    ctx["aruco_norm"], args.align_success_mm)
                        align_success = True
                    else:
                        last_err = "unknown" if ctx is None else f"{ctx['aruco_norm']:.2f} mm"
                        logger.warning("自动对准未达标，最终误差=%s (可重新按 m 重试)", last_err)

                moving = False
                logger.info("=== 自动对准结束: %s ===", "成功" if align_success else "未达标")

                if robot_connected:
                    new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if new_tcp is not None:
                        T_g2b_after = pose_to_matrix(new_tcp)
                        if T_g2b_ref is not None:
                            ref_xyz, ref_res_t, ref_res_r = compute_pose_error_in_frame(
                                T_g2b_after, T_g2b_ref)
                            logger.info("%s residual (vs ref): trans=%.2f mm, rot=%.2f deg",
                                        current_marker_type, ref_res_t, ref_res_r)
                            logger.info("  ref axes: dX=%.2f dY=%.2f dZ=%.2f mm", *ref_xyz)
                        logger.info("  Actual TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *new_tcp)

                if (current_marker_type == "插枪") and (already_return_gun == False) and robot_connected:
                    current_tcp = robot.get_tcp_pose()
                    current_matrix = pose_to_matrix(current_tcp)
                    offset_pose = INSERT_BEFORE_OFFSET_POSE
                    offset_matrix = pose_to_matrix(offset_pose)
                    target_matrix = current_matrix @ offset_matrix
                    target_pose = matrix_to_pose(target_matrix)

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["INSERT_BEFORE_TEMPLATE"] = target_pose

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    flow.update_CFG()

            elif key == ord('d'):
                force_ctrl.run_ForceControl_OpenCover()

            elif key == ord('e'):
                force_ctrl.run_forcecontrol_charge_in()

            elif key == ord('s'):
                arm_controller = SMSSTSController("/dev/ttysWK1")
                arm_controller.connect()
                arm_controller.press_trigger()
                arm_controller.disconnect()

                force_ctrl.run_forcecontrol_charge_out()

                arm_controller.connect()
                arm_controller.reset_position()
                arm_controller.disconnect()
            elif key == ord('x'):
                flow.gripper.set_speed(100)
                
                async def parallel_task():
                    await asyncio.gather(to_thread(flow.gripper.open),
                                   to_thread(robot.move_joint,INIT_JOINT, 5))

                asyncio.run(parallel_task())

            elif key == ord('z'):
                def _servo_press():
                    arm_controller = SMSSTSController("/dev/ttysWK1")
                    arm_controller.connect()
                    arm_controller.press_trigger()
                    arm_controller.disconnect()

                async def parallel_task():
                    await asyncio.gather(to_thread(_servo_press),
                                   to_thread(robot.move_joint,INIT_JOINT, 5))

                asyncio.run(parallel_task())

            elif key == ord('3'):
                # robot.move_joint([112.196, 99.884, -56.131, 128.446, -5.912, -17.849], speed=40)

                flow.gripper.set_speed(100)
                # flow.gripper.set_position(45)
                async def parallel_task():
                    await asyncio.gather(to_thread(flow.gripper.set_position,45),
                                   to_thread(robot.move_joint,INIT_JOINT,40))

                asyncio.run(parallel_task())

                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue

                cover_3D_pose = cover_pose_estimator.pose_estimation()
                if cover_3D_pose is not None:
                    logger.info("Cover 3D pose: %s", cover_3D_pose)
                    cover_3D_matrix = pose_to_matrix(cover_3D_pose)

                    current_tool_pose = get_tcp_pose_in_tool_mm(
                        robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                    if current_tool_pose is None:
                        logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                        continue
                    cur_tcp_matrix = pose_to_matrix(current_tool_pose)

                    # cover_offset_pose = [0, 0, 0, 0, 0, 0]
                    cover_offset_pose = STEREO_DETECT_OFFSET_POSE

                    cover_offset_matrix = pose_to_matrix(cover_offset_pose)
                    target_matrix = cur_tcp_matrix @ (cover_3D_matrix @ cover_offset_matrix)

                    target_pose = matrix_to_pose(target_matrix)
                    print(f" pose: {target_pose}")
                    logger.info("  Target TCP for coarse alignment: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                *target_pose)

                    # 参考关节角（某个停车位置）
                    ref_joint = STEREO_IK_REF_JOINT
                    cur_joint = robot.get_joint_pose()
                    target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=ref_joint)
                    robot.move_by_joint_list(joints=[target_joint], speeds=[40])
                    # robot.move_linear(CartesianPose(*target_pose))

            elif key == ord('a'):
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot read tool TCP")
                    continue
                if takegun_offset is None:
                    logger.warning("No relative offset set; record positions 1 & 2 or load from file first")
                    continue

                current_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_pose is None:
                    logger.warning("Failed to read current pose")
                    continue

                current_matrix = pose_to_matrix(current_pose)
                offset_pose = TAKEGUN_OFFSET_POSE

                offset_matrix = pose_to_matrix(offset_pose)
                target_matrix = current_matrix @ offset_matrix
                target_pose = matrix_to_pose(target_matrix)

                dist_trans, dist_rot = compute_distance(current_pose, target_pose)
                if dist_trans > args.max_trans or dist_rot > args.max_rot:
                    logger.warning("Motion distance too large: trans=%.2f mm (max %.2f), rot=%.2f deg (max %.2f)",
                                   dist_trans, args.max_trans, dist_rot, args.max_rot)
                    response = input("Continue? (y/n): ")
                    if response.lower() != 'y':
                        continue

                logger.info("Applying relative offset:")
                logger.info("  Current:  X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *current_pose)
                logger.info("  Delta:    dX=%.2f dY=%.2f dZ=%.2f mm, dRx=%.2f dRy=%.2f dRz=%.2f deg", *takegun_offset)
                logger.info("  Target:   X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *target_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip relative offset move + 120mm advance")
                else:
                    moving = True
                    motion_success = True
                    robot.set_speed(100)
                    cart = CartesianPose(
                        x=target_pose[0], y=target_pose[1], z=target_pose[2],
                        rx=target_pose[3], ry=target_pose[4], rz=target_pose[5],
                    )
                    ok = robot.move_joint_and_wait(cart, speed=10, timeout=args.move_timeout)
                    if ok:
                        logger.info("Relative offset move complete")
                        current_tool_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if current_tool_pose is not None:
                            advance_pose = offset_pose_along_tool_axis(current_tool_pose, 123.0, axis='z')
                            logger.info("Take gun: advance 120 mm along tool Z-axis")
                            logger.info("  Target: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *advance_pose)
                            robot.set_speed(30)
                            ok = robot.move_and_wait(CartesianPose(
                                x=advance_pose[0], y=advance_pose[1], z=advance_pose[2],
                                rx=advance_pose[3], ry=advance_pose[4], rz=advance_pose[5],
                            ), timeout=args.move_timeout)
                            if ok:
                                logger.info("Take-gun advance 120mm complete")
                            else:
                                logger.warning("Take-gun advance 120mm failed")
                                motion_success = False
                        else:
                            logger.warning("Failed to read TCP for 121mm advance")
                            motion_success = False
                    else:
                        logger.warning("Relative offset move timed out or failed")
                        motion_success = False
                    moving = False

                    final_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if final_pose is not None:
                        logger.info("  Final: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *final_pose)

                    robot.set_speed(DEFAULT_SPEED)
                    if motion_success:
                        while robot.is_moving():
                            logger.info("Waiting for robot to stop before gripper action...")
                            time.sleep(0.1)
                        logger.info("Auto-trigger gripper close + servo press")
                        flow.gripper.set_speed(30)
                        flow.gripper.set_force(100)
                        flow.gripper.set_position(32)
                        timeout_s = 5.0
                        t0 = time.time()
                        while True:
                            status = flow.gripper.get_grip_status()
                            if status in (1, 2):
                                logger.info("Gripper action complete (status=%d)", status)
                                break
                            if time.time() - t0 > timeout_s:
                                logger.warning("Gripper action timeout after %.1fs (status=%s)", timeout_s, status)
                                break
                            time.sleep(0.1)
                        arm_controller = SMSSTSController("/dev/ttysWK1")
                        arm_controller.connect()
                        arm_controller.press_trigger()
                        arm_controller.disconnect()
                    else:
                        logger.warning("Skipping gripper+servo due to motion failure")

            elif key == ord('b'):
                current_pose = robot.get_tcp_pose()
                current_matrix = pose_to_matrix(current_pose)
                offset_pose_cover = INNER_COVER_OFFSET_POSE
                offset_matrix = pose_to_matrix(offset_pose_cover)
                target_matrix = current_matrix @ offset_matrix
                target_pose = matrix_to_pose(target_matrix)
                take_cover_pose = robot.relative_tool_pose(dz=50, init_pose=target_pose).to_list()
                current_joint = robot.get_joint_pose()
                target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=current_joint)
                take_cover_joint = robot.inverse_kinematics(target_pose=take_cover_pose, initial_joints=current_joint)

                robot.move_by_joint_list(joints=[target_joint, take_cover_joint], speeds=[25, 15])

                if already_take_inner_cover == False:
                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["PUSH_POINT1"] = target_pose
                        data["PUSH_POINT2"] = take_cover_pose

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    flow.update_CFG()
                    logger.info('PUSH POINT1 POINT2已写入json.')

            elif key == ord('c'):
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot read tool TCP")
                    continue

                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_tool_pose is None:
                    logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                    continue

                before_retract_joint = robot.get_joint_pose()
                retract_pose = offset_pose_along_tool_axis(current_tool_pose, -100.0, axis='z')
                retract_joint = robot.inverse_kinematics(target_pose=retract_pose, initial_joints=before_retract_joint)

                logger.info("Retract 100 mm along tool Z-axis")
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *retract_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip retract (100mm)")
                else:
                    moving = True
                    motion_success = True
                    robot.set_speed(100)
                    ok = execute_move(robot, retract_pose, timeout=args.move_timeout)
                    if ok:
                        logger.info("Retract 100mm complete")
                    else:
                        logger.warning("Retract 100mm failed")
                        motion_success = False
                    moving = False

                    try:
                        tcp_after = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after = None
                    if tcp_after is not None:
                        logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after)

                    if motion_success:
                        while robot.is_moving():
                            logger.info("Waiting for robot to stop before servo reset...")
                            time.sleep(0.1)
                        logger.info("Auto-trigger servo reset")
                        arm_controller = SMSSTSController("/dev/ttysWK1")
                        arm_controller.connect()
                        arm_controller.reset_position()
                        arm_controller.disconnect()
                    else:
                        logger.warning("Skipping servo reset due to motion failure")
                    robot.set_speed(DEFAULT_SPEED)

                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data["GUN_HOME2"] = before_retract_joint
                    data["GUN_HOME1"] = retract_joint

                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                flow.update_CFG()

            elif key == ord('g'):
                if outer_cover_is_opened == False:
                    flow.run(1)
                else:
                    flow.run(8)

            elif key == ord('h'):
                flow.run(2)
                already_take_inner_cover = True

            elif key == ord('j'):
                flow.run(3)

            elif key == ord('k'):
                flow.run(4)
                already_return_gun = True

            elif key == ord('l'):
                flow.run(5)

            elif key == ord('p'):
                flow.run(7)
                flow.run(6)
            
            elif key == ord('t'):
                # 测试直线运动间隔耗时
                robot.move_linear
        
    finally:
        stop_event.set()
        det_thread.join(timeout=2.0)
        cv2.destroyAllWindows()

        # 停止状态监控线程
        # force_controller.stop_state_monitor()

        if camera is not None:
            camera.close()
        if robot_connected:
            flow.disconnect()


if __name__ == '__main__':
    main()
