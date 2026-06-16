#!/usr/bin/env python3
"""
一次性到位 + 插入/取枪运动 | One-Shot Move with Insert/Extract (v4.5: 自动多次对准)

基于 ArUco 视觉检测 + 手眼标定，一键运动到参考基准位，
并支持在工具坐标系下沿 z 轴执行插入/拔出动作。

v4.5 改进: 按 'm' 自动循环对准直到 ArUco 误差 < 阈值或达到最大次数。

工作流 (Workflow):
    1. 按 'r': 记录当前 ArUco 位姿为参考（自动识别插枪/取枪 marker），同步保存法兰坐标系下的参考 TCP
    2. 按 'm': 自动多次视觉对准直到满足阈值（自动适配插枪/取枪）
    3. 按 'b': 直接回到按 'r' 时保存的机械臂法兰坐标系位姿
    4. 按 'i': 切换到工具坐标系，沿工具 z 轴前进（自动适配插枪/取枪距离）
    5. 按 'q' / ESC: 退出

用法 (Usage):
    python scripts/oneshot_move_crobot_insert_v4.5.py --camera mecheye --insert-cm-insert 7 --insert-cm-take 12
    python scripts/oneshot_move_crobot_insert_v4.5.py --camera mecheye --no-robot --align-max-attempts 4
"""

import argparse
import logging
import os
import sys
import threading
import time
import fcntl
import json


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
from robovision.vision.auto_exposure import AutoExposureController

# --- 第三方驱动 ---
from crobot_driver_interface import CartesianPose
from scripts.cover_main_bak_run import CoverActionFlow
from gripper_controller import GripperController
from relative_move import apply_relative_pose, load_offset_from_file, compute_distance
from Intergration.stereo_camera_calib.yrq.pose_estimation.cover_pose_estimator import CoverPoseEstimator
from Intergration.FTServo_Linux_main.examples.sms_sts_driver import SMSSTSController

from third_party.force_control_crp import RobotController, BridgeCRobotAdapter



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

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICK = 1
PROMPT_TEXT_COLOR = (255, 255, 255)
_TAKEGUN_OFFSET_FILE = "data/relative_offset_takegun.txt"
_POSES_FILE = "data/relative_poses_takegun.txt"
_CHARGING_OFFSET_FILE =  "data/relative_offset_charging.txt"

# AE 硬编码参数
_AE_TARGET_BRIGHTNESS = 128
_AE_DEADBAND = 12
_AE_ADJUST_INTERVAL = 15
_AE_EXPOSURE_LIMIT_MS = 50.0
_AE_GAIN_LIMIT_DB = 12.0

MARKER_DISPLAY_NAMES = {
    "插枪": "Insert",
    "取枪": "Take",
}


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
    parser = argparse.ArgumentParser(description='一次性到位 + 工具坐标系插入')
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
    parser.add_argument('--return-gun-up', type=float, default=0.3,
                        help='归枪抬升距离（cm，内部乘以 10 转换为 mm）')
    parser.add_argument('--return-gun-left', type=float, default=0.2,
                        help='归枪左移距离（cm，内部乘以 10 转换为 mm）')
    parser.add_argument('--move-timeout', type=float, default=MOVE_TIMEOUT,
                        help=f'普通运动等待超时秒数（默认 {MOVE_TIMEOUT}）')
    parser.add_argument('--force-retract-target', type=int, default=70,
                        help='按 o 执行 KEBA mode_3 力控拔枪的目标位移（默认 70）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（固定 IPPE_SQUARE，且不使用时序滤波）')
    parser.add_argument('--no-temporal-filter', action='store_true',
                        help='关闭 Kalman 角点滤波和位姿时序平滑（仅影响标准检测模式）')
    parser.add_argument('--no-ae', action='store_true',
                        help='禁用自动曝光控制（即使配置中已启用）')
    parser.add_argument('--hw-ae', action='store_true',
                        help='启用硬件自动曝光（默认关闭）')
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
    parser.add_argument('--debug', action='store_true')
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

    # 加载曝光参考（记录参考位姿时的 marker ROI 亮度）
    exp_ref_path_insert = args.aruco_ref.replace('aruco_pose_ref', 'aruco_exp_ref')
    exp_ref_path_takegun = args.aruco_ref_takegun.replace('aruco_pose_ref_takegun', 'aruco_exp_ref_takegun')
    ref_bri_insert = None
    if os.path.isfile(exp_ref_path_insert):
        with open(exp_ref_path_insert, 'r') as f:
            try:
                ref_bri_insert = float(f.read().strip().split(',')[0])
                logger.info("Insert ref bri loaded: bri=%.0f", ref_bri_insert)
            except (ValueError, IndexError):
                pass
    ref_bri_takegun = None
    if os.path.isfile(exp_ref_path_takegun):
        with open(exp_ref_path_takegun, 'r') as f:
            try:
                ref_bri_takegun = float(f.read().strip().split(',')[0])
                logger.info("Take ref bri loaded: bri=%.0f", ref_bri_takegun)
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

    # force_ctrl.run_power_on_ppmode()

    # --- 初始化相机和检测器 | Camera & detector init ---
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    # 自动曝光控制器（--no-ae 禁用）
    ae_ctrl = None
    if not args.no_ae:
        try:
            ae_ctrl = AutoExposureController(
                camera,
                target_brightness=_AE_TARGET_BRIGHTNESS,
                deadband=_AE_DEADBAND,
                adjust_interval=_AE_ADJUST_INTERVAL,
                exposure_limit_ms=_AE_EXPOSURE_LIMIT_MS,
                gain_limit_db=_AE_GAIN_LIMIT_DB,
            )
            if args.hw_ae:
                ae_ctrl.setup()
                logger.info("Auto exposure controller initialized (hardware AE)")
            else:
                logger.info("Auto exposure controller initialized (software servo)")
        except Exception:
            logger.warning("Failed to initialize auto exposure controller", exc_info=True)

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

    def _detection_loop():
        nonlocal latest_det, latest_det_frame
        while not stop_event.is_set():
            ok_d, frm = camera.read_frame()
            if not ok_d or frm is None:
                time.sleep(0.005)
                continue
            frm = frm.copy()
            if ae_ctrl is not None:
                ae_ctrl.measure_frame(frm)
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
    logger.info("Ready: r=SetRef m=AutoAlign b=Back2Ref i=Insert f=ForceIn o=ForceOut v=Gripper q=Quit (detection thread started)")

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

    # # 力控控制类
    # force_controller = RobotController()

    # # 加载库
    # if not force_controller.load():
    #     sys.exit(1)
    # # 启动状态监控子线程
    # force_controller.start_state_monitor()

    try:
        first_move = 0 # 随动标志位
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
            if current_marker_type == "插枪" and args.insert_cm_insert <= 0:
                put_text(vis, "Set --insert-cm-insert > 0 to enable insert action", y, PROMPT_TEXT_COLOR)
                y += 28
            elif current_marker_type == "取枪" and args.insert_cm_take <= 0:
                put_text(vis, "Set --insert-cm-take > 0 to enable insert action", y, PROMPT_TEXT_COLOR)
                y += 28
            elif current_marker_type is not None:
                insert_cm = args.insert_cm_insert if current_marker_type == "插枪" else args.insert_cm_take
                put_text(vis, f"Press i to move from current coord sys 2 TCP ({insert_cm} cm)", y, PROMPT_TEXT_COLOR)
                y += 28

            if moving:
                put_text(vis, "MOVING...", y, (0, 100, 255))
                y += 30

            if ae_ctrl is not None:
                bri = ae_ctrl.brightness()
                bri_str = f"AE: brightness={bri:.0f}" if bri is not None else "AE: initializing"
                if ae_ctrl.roi_active:
                    bri_str += " [ROI]"
                put_text(vis, bri_str, y, (200, 200, 100))
                y += 30

            roi_bri = _marker_roi_brightness(frame, target_data) if target_data is not None else None
            roi_str = f"ROI Bri: {roi_bri:.0f}" if roi_bri is not None else "ROI Bri: None"
            roi_color = (0, 255, 128) if roi_bri is not None else (100, 100, 100)
            put_text(vis, roi_str, y, roi_color)
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

            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            ref_str = f"Ref:Insert{'SET' if ref_set_insert else 'NONE'} Take{'SET' if ref_set_takegun else 'NONE'}"
            status = (f"ONESHOT+INSERT | {ref_str} | {robot_str} | "
                      "r=Ref m=Move b=Back i=Insert f=ForceIn o=ForceOut q=Quit")
            put_text(vis, status, h_disp - 15, (140, 140, 140))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if first_move == 1:
                # TODO:机械臂随动至设定好的位置
                robot.move_linear(CartesianPose(355.697, -335.04, 182.446, 103.439, -25.092, 37.714))

                first_move = 0
                with open('/home/nvidia/Downloads/HD/HD_0323/Intergration/stereo_camera_calib/yrq/first_move.json', 'w', encoding='utf-8') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(0, f)
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f, fcntl.LOCK_UN)

            
            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                # 保存当前检测到的 marker 位姿和 TCP 为参考
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

                    # 保存 marker ROI 亮度参考
                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            if bri is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_insert) or '.', exist_ok=True)
                                with open(exp_ref_path_insert, 'w') as f:
                                    f.write(f"{bri:.1f}\n")
                                ref_bri_insert = bri
                                logger.info("Insert ref bri saved: bri=%.0f", bri)
                        except Exception:
                            logger.warning("Failed to save insert bri ref", exc_info=True)
                else:  # 取枪
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

                    # 保存 marker ROI 亮度参考
                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            if bri is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_takegun) or '.', exist_ok=True)
                                with open(exp_ref_path_takegun, 'w') as f:
                                    f.write(f"{bri:.1f}\n")
                                ref_bri_takegun = bri
                                logger.info("Take ref bri saved: bri=%.0f", bri)
                        except Exception:
                            logger.warning("Failed to save take bri ref", exc_info=True)

            elif key == ord('m'):
                # v4.5: 自动多次视觉对准
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

                # 关闭硬件AE，后续 m→b/a 操作保持手动曝光一致
                if args.hw_ae:
                    try:
                        camera.set_exposure_auto(False)
                        logger.debug("Hardware AE disabled at start of align")
                    except Exception:
                        pass

                # 曝光收敛（只做一次）
                ref_bri = ref_bri_insert if current_marker_type == "插枪" else ref_bri_takegun
                if ref_bri is not None and not args.no_robot:
                    try:
                        camera.set_exposure_auto(False)
                        cur_exp = camera.get_exposure_time()
                        cur_gain_db = camera.get_gain()
                        if cur_exp is None:
                            cur_exp = 5000.0
                        if cur_gain_db is None:
                            cur_gain_db = 0.0
                        logger.info("Converging bri to ref: target_bri=%.0f start_exp=%.0fus gain=%.1fdB",
                                    ref_bri, cur_exp, cur_gain_db)
                        for i in range(15):
                            time.sleep(0.15)
                            with det_lock:
                                cur_result, cur_frame = latest_det, latest_det_frame
                            if not cur_result or cur_frame is None:
                                cv2.imshow(win, vis)
                                cv2.waitKey(1)
                                continue
                            cur_bri = None
                            for mid, d in cur_result.items():
                                if mid in (insert_marker_id, takegun_marker_id):
                                    cur_bri = _marker_roi_brightness(cur_frame, d)
                                    break
                            if cur_bri is None:
                                cv2.imshow(win, vis)
                                cv2.waitKey(1)
                                continue
                            err = cur_bri - ref_bri
                            if abs(err) <= 5:
                                logger.info("Bri converged: %.0f->%.0f (err=%.1f iters=%d)",
                                            ref_bri, cur_bri, err, i + 1)
                                break
                            ratio = 1.0 + 0.3 * (-err / max(ref_bri, 1.0))
                            ratio = min(max(ratio, 0.90), 1.10)
                            new_exp = max(100.0, cur_exp * ratio)
                            if new_exp >= _AE_EXPOSURE_LIMIT_MS * 1000 and err < 0:
                                cur_gain_db = min(cur_gain_db + 0.2, _AE_GAIN_LIMIT_DB)
                                camera.set_gain(cur_gain_db)
                            elif new_exp <= 100.0 and err > 0 and cur_gain_db > 0:
                                cur_gain_db = max(cur_gain_db - 0.2, 0.0)
                                camera.set_gain(cur_gain_db)
                            else:
                                cur_exp = new_exp
                                camera.set_exposure_time(cur_exp)
                            cv2.imshow(win, vis)
                            cv2.waitKey(1)
                    except Exception:
                        logger.warning("Exposure convergence failed", exc_info=True)

                # === 自动循环对准 ===
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

                    # 运动完成后 log 当前 TCP
                    new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if new_tcp is not None:
                        logger.info("  对准 #%d 运动后 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    attempt, *new_tcp)

                # 循环结束后最终检查
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

                # 对准后残差日志
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

                # 插枪后处理: 更新 poses_config.json
                if (current_marker_type == "插枪") and (already_return_gun == False) and robot_connected:
                    current_tcp = robot.get_tcp_pose()
                    current_matrix = pose_to_matrix(current_tcp)
                    offset_pose = [-4.203, -204.682, -35.227, -4.467, 4.029, -1.981]
                    offset_matrix = pose_to_matrix(offset_pose)
                    target_matrix = current_matrix @ offset_matrix
                    target_pose = matrix_to_pose(target_matrix)

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["INSERT_BEFORE_TEMPLATE"] = target_pose

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    flow.update_CFG()


            elif key == ord('e'):
                
                # force_ctrl.run_power_on_ppmode()
                force_ctrl.run_forcecontrol_charge_in()
                # force_ctrl.run_ForceControl_return_the_gun()

                #force_ctrl.run_ForceControl_OpenCover()
                # force_controller.stop_state_monitor()

            elif key == ord('d'):
                # force_ctrl.run_power_on_ppmode()
                # force_ctrl.run_forcecontrol_pose_adjust()
                force_ctrl.run_ForceControl_OpenCover()
                # robot.set_speed(DEFAULT_SPEED)

            elif key == ord('s'):
                # 舵机按下
                arm_controller = SMSSTSController("/dev/ttysWK1")
                arm_controller.connect()
                arm_controller.press_trigger()
                arm_controller.disconnect()

                # force_ctrl.run_power_on_ppmode()
                # 力控拔出
                force_ctrl.run_forcecontrol_charge_out()

                # 沿法兰z退3cm
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot read tool TCP")
                    continue
                    
                robot.set_speed(100)
                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_tool_pose is None:
                    logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                    continue

                retract_offset_mm = -60

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(retract_offset_mm), axis='z')
                safe, reason = check_oneshot_safety(abs(retract_offset_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("Take gun: retract %.2f mm (%.1f mm) along tool Z-axis",
                            retract_offset_mm, retract_offset_mm)
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip take-gun retract")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(flange)")
                
                robot.set_speed(DEFAULT_SPEED)

                # 舵机抬起
                arm_controller.connect()
                arm_controller.reset_position()
                arm_controller.disconnect()
         

            elif key == ord('i'):
                # 沿工具坐标系 z 轴前进 (插入) | Move along tool Z-axis (insert)
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot read tool TCP")
                    continue
                if current_marker_type is None:
                    logger.warning("No marker detected, cannot determine insert distance")
                    continue

                # 根据当前检测到的 marker 类型选择插入距离
                if current_marker_type == "插枪":
                    insert_mm = insert_offset_mm
                    insert_cm = args.insert_cm_insert
                    marker_type = "插枪"
                else:
                    insert_mm = take_offset_mm
                    insert_cm = args.insert_cm_take
                    marker_type = "取枪"

                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID,
                    label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_tool_pose is None:
                    logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                    continue

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(insert_mm), axis='z')
                safe, reason = check_oneshot_safety(abs(insert_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("Insert %s: advance %.2f cm (%.2f mm) along tool Z-axis",
                            marker_type, insert_cm, insert_mm)
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip %s insert motion", marker_type)
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After %s insert TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    marker_type, *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(flange)")

            elif key == ord('f'):
                # KEBA 力控插枪 (mode_1)
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot execute KEBA force insert")
                    continue
                if args.no_robot:
                    logger.info("[DRY RUN] Skip KEBA force insert")
                    continue
                moving = True
                execute_keba_force_mode(robot, 1)
                moving = False

            elif key == ord('o'):
                # KEBA 力控拔枪 (mode_3)
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if not robot_connected:
                    logger.warning("Robot not connected, cannot execute KEBA force extract")
                    continue
                if args.no_robot:
                    logger.info("[DRY RUN] Skip KEBA force extract")
                    continue
                moving = True
                execute_keba_force_mode(robot, 3, displace_target=args.force_retract_target)
                moving = False

            elif key == ord('v'):
                # 夹爪控制: 关闭夹爪 → 等待完成 → 自动触发舵机按下
                flow.gripper.set_speed(15)
                flow.gripper.set_position(32)
                # 等待夹爪动作完成（0=运动中, 1=到达位置, 2=夹住物体, 3=物体掉落）
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
                # 自动触发舵机按下
                arm_controller = SMSSTSController("/dev/ttysWK1")
                arm_controller.connect()
                arm_controller.press_trigger()
                arm_controller.disconnect()
               
            elif key == ord('2'):
                # 硬编码: 取枪初始位置 (调试用)
                robot.set_speed(100)
                robot.move_linear(CartesianPose(585.525,300.417,318.319,114.3886,-31.4888,134.5344))

            elif key == ord('4'):
                # 硬编码: 插枪初始位置 (调试用)
                robot.set_speed(100)
                robot.move_linear(CartesianPose(697.509, -202.016, 184.404, 105.993, -26.083, 27.85))
            
            elif key == ord('1'):
                # 双目开盖初始位置
                robot.move_joint([112.196, 99.884, -56.131, 128.446, -5.912, -17.849], speed=40)
            
            elif key == ord('z'):
                robot.move_joint([85.793, 91.365, -66.574, 139.191, -2.617, -18.839], speed=25)

            # elif key == ord('3'):
            #     # key = 1
            #     robot.move_joint([112.196, 99.884, -56.131, 128.446, -5.912, -17.849], speed=40)

            #     flow.gripper.set_position(45)
            #     # 插座盖 3D 位姿估计 → 粗定位运动 (CoverPoseEstimator)
            #     if moving:
            #         logger.warning("Motion in progress, please wait")
            #         continue

            #     cover_3D_pose = cover_pose_estimator.pose_estimation()
            #     if cover_3D_pose is not None:
            #         logger.info("Cover 3D pose: %s", cover_3D_pose)
            #         cover_3D_matrix = pose_to_matrix(cover_3D_pose)

            #         current_tool_pose = get_tcp_pose_in_tool_mm(
            #             robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
            #         if current_tool_pose is None:
            #             logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
            #             continue
            #         cur_tcp_matrix = pose_to_matrix(current_tool_pose)

            #         # cover_offset_pose = [77.095, 18.253, -298.781, 3.657, -2.192, -1.198]
            #         # cover_offset_pose = [60.251, 13.522, -293.08, 0.897, 1.649, -0.852]
            #         # cover_offset_pose = [66.834, 19.923, -299.825, 1.916, -1.491, -1.663]
            #         cover_offset_pose = [0, 0, 0, 0, 0, 0]

            #         cover_offset_matrix = pose_to_matrix(cover_offset_pose)
            #         target_matrix = cur_tcp_matrix @ (cover_3D_matrix @ cover_offset_matrix)

            #         # target_matrix = cur_tcp_matrix @ cover_3D_matrix
            #         target_pose = matrix_to_pose(target_matrix)
            #         print(f" pose: {target_pose}")
            #         logger.info("  Target TCP for coarse alignment: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
            #                     *target_pose)
                    
            #         # cur_joint = robot.get_joint_pose()
            #         # target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=cur_joint)
            #         # robot.move_by_joint_list(joints=[target_joint], speeds=[40])
            #         # robot.move_linear(CartesianPose(*target_pose))
                           
            elif key == ord('5'):

                current_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_pose is None:
                    logger.warning("Failed to read current pose")
                    continue

                # target_pose = apply_relative_pose(current_pose, charging_offset)
                current_matrix = pose_to_matrix(current_pose)
                offset_pose = [85.38998138966815, -113.95230931001532, -5.829060832306595, 4.563053297458938, -11.804965761109615, -4.385023510262975]
                offset_matrix = pose_to_matrix(offset_pose)
                target_matrix = current_matrix @ offset_matrix
                target_pose = matrix_to_pose(target_matrix)

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
                    ok = robot.move_and_wait(cart, timeout=args.move_timeout)
                    if ok:
                        logger.info("Relative offset move complete")
                        current_tool_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if current_tool_pose is not None:
                            advance_pose = offset_pose_along_tool_axis(current_tool_pose, 100.0, axis='z')
                            logger.info("Take gun: advance 120 mm along tool Z-axis")
                            logger.info("  Target: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *advance_pose)
                            robot.set_speed(100)
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
                            logger.warning("Failed to read TCP for 120mm advance")
                            motion_success = False
                    else:
                        logger.warning("Relative offset move timed out or failed")
                        motion_success = False
                    moving = False

                    final_pose = get_robot_tcp_pose_mm(robot, robot_cfg)
                    if final_pose is not None:
                        logger.info("  Final: X=%.2f Y=%.2f Z=%.2f mm, Rx=%.2f Ry=%.2f Rz=%.2f deg", *final_pose)


                # # 复合动作: offset [-110, -130] → 前进 100mm | Insert offset → advance 100mm
                # if moving:
                #     logger.warning("Motion in progress, please wait")
                #     continue
                # if not robot_connected:
                #     logger.warning("Robot not connected, cannot read tool TCP")
                #     continue

                # current_tool_pose = get_tcp_pose_in_tool_mm(
                #     robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                # if current_tool_pose is None:
                #     logger.warning("Failed to read tool TCP")
                #     continue

                # # 第一步: offset [-110, -130]
                # offset_pose = [-30, -200, 0, 0, 0, 0]
                # offset_matrix = pose_to_matrix(offset_pose)
                # current_tool_matrix = pose_to_matrix(current_tool_pose)
                # target_matrix = current_tool_matrix @ offset_matrix
                # target_pose = matrix_to_pose(target_matrix)

                # # 第二步: 沿工具 z 轴前进 100mm
                # advance_pose = offset_pose_along_tool_axis(target_pose, 65.0, axis='z')

                # logger.info("Insert offset [-110, -130] + advance 100mm along Z")
                # logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                #             *current_tool_pose)

                # if args.no_robot:
                #     logger.info("[DRY RUN] Skip insert offset + 100mm advance")
                # else:
                #     moving = True
                #     robot.set_speed(100)
                #     ok = execute_move(robot, target_pose, timeout=args.move_timeout)
                #     if ok:
                #         logger.info("Offset move complete")
                #         robot.set_speed(80)
                #         ok = execute_move(robot, advance_pose, timeout=args.move_timeout)
                #         if ok:
                #             logger.info("Advance 100mm complete")
                #         else:
                #             logger.warning("Advance 100mm failed")
                #     else:
                #         logger.warning("Offset move failed")
                #     moving = False

                #     try:
                #         tcp_after = get_robot_tcp_pose_mm(robot, robot_cfg)
                #     except Exception:
                #         tcp_after = None
                #     if tcp_after is not None:
                #         logger.info("  After offset+advance TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                #                     *tcp_after)
                


            elif key == ord('0'):
                # 调试按键
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

                retract_offset_mm = -120

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(retract_offset_mm), axis='z')
                safe, reason = check_oneshot_safety(abs(retract_offset_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("Take gun: retract %.2f mm (%.1f mm) along tool Z-axis",
                            retract_offset_mm, retract_offset_mm)
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip take-gun retract")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(flange)")

            elif key == ord('w'):
                # 力控拔枪结束后，直线退出
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

                retract_offset_mm = -150

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(retract_offset_mm), axis='z')
                safe, reason = check_oneshot_safety(abs(retract_offset_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("Take gun: retract %.2f mm (%.1f mm) along tool Z-axis",
                            retract_offset_mm, retract_offset_mm)
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip take-gun retract")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(flange)")

            elif key == ord('c'):
                # 拔出 100mm + 自动舵机复位 | Retract 100mm + auto servo reset
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

                    # 自动触发舵机复位（原按键 9），仅在运动成功后执行
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
                
                # 归枪点写入json
                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data["GUN_HOME2"] = before_retract_joint
                    data["GUN_HOME1"] = retract_joint

                with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                flow.update_CFG()

            elif key == ord('6'):
                # 硬编码: 关节运动到预定义位置 1 (调试用)
                robot.move_joint([6.938, -67.507, 83.516, 140.491, 86.041, -90.24])
            elif key == ord('7'):
                # 硬编码: 关节运动到预定义位置 2 (调试用)
                robot.move_joint([6.364, -77.978, 58.77, 126.224, 86.605, -90.345])

            elif key == ord('8'):
                # 舵机: 按下扳机 (SMSSTSController)
                arm_controller = SMSSTSController("/dev/ttysWK1")
                arm_controller.connect()
                arm_controller.press_trigger()
                arm_controller.disconnect()

            elif key == ord('9'):
                # 舵机: 松开扳机 / 复位 (SMSSTSController)
                arm_controller = SMSSTSController("/dev/ttysWK1")
                arm_controller.connect()
                arm_controller.reset_position()
                arm_controller.disconnect()
            
            elif key == ord('a'):            
                # 复合动作: 应用相对位移 → 前进 120mm | Apply relative offset → advance 120mm
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

                # target_pose = apply_relative_pose(current_pose, takegun_offset)
                current_matrix = pose_to_matrix(current_pose)
                offset_pose = [51.889, -230.636, 292.904, -14.379, 1.694, 0.214]
                
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
                    ok = robot.move_joint_and_wait(cart, speed = 10, timeout=args.move_timeout)
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
                    # 自动触发夹爪闭合 + 舵机按下（原按键 v，已合并按键 8）
                    if motion_success:
                        while robot.is_moving():
                            logger.info("Waiting for robot to stop before gripper action...")
                            time.sleep(0.1)
                        logger.info("Auto-trigger gripper close + servo press")
                        flow.gripper.set_speed(30)
                        flow.gripper.set_force(75)
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

                # time.sleep(0.2)

                # # 【按键c】拔出 100mm + 自动舵机复位 | Retract 100mm + auto servo reset
                # if moving:
                #     logger.warning("Motion in progress, please wait")
                #     continue
                # if not robot_connected:
                #     logger.warning("Robot not connected, cannot read tool TCP")
                #     continue

                # current_tool_pose = get_tcp_pose_in_tool_mm(
                #     robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                # if current_tool_pose is None:
                #     logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                #     continue

                # retract_pose = offset_pose_along_tool_axis(current_tool_pose, -100.0, axis='z')

                # logger.info("Retract 100 mm along tool Z-axis")
                # logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                #             *current_tool_pose)
                # logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                #             *retract_pose)

                # if args.no_robot:
                #     logger.info("[DRY RUN] Skip retract (100mm)")
                # else:
                #     moving = True
                #     motion_success = True
                #     robot.set_speed(75)
                #     ok = execute_move(robot, retract_pose, timeout=args.move_timeout)
                #     if ok:
                #         logger.info("Retract 100mm complete")
                #     else:
                #         logger.warning("Retract 100mm failed")
                #         motion_success = False
                #     moving = False

                #     try:
                #         tcp_after = get_robot_tcp_pose_mm(robot, robot_cfg)
                #     except Exception:
                #         tcp_after = None
                #     if tcp_after is not None:
                #         logger.info("  After retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                #                     *tcp_after)

                #     # 自动触发舵机复位（原按键 9），仅在运动成功后执行
                #     if motion_success:
                #         while robot.is_moving():
                #             logger.info("Waiting for robot to stop before servo reset...")
                #             time.sleep(0.1)
                #         logger.info("Auto-trigger servo reset")
                #         arm_controller = SMSSTSController("/dev/ttysWK1")
                #         arm_controller.connect()
                #         arm_controller.reset_position()
                #         arm_controller.disconnect()
                #     else:
                #         logger.warning("Skipping servo reset due to motion failure")

                # 偏移运动完成，恢复硬件AE
                if args.hw_ae:
                    try:
                        camera.set_exposure_auto(True)
                        camera.set_gain_auto(True)
                        logger.debug("Hardware AE restored after offset move")
                    except Exception:
                        pass

            elif key == ord('b'):
                # 取小盖对准后的相对运动
                current_pose = robot.get_tcp_pose()
                current_matrix = pose_to_matrix(current_pose)
                offset_pose_cover = [-2.515, -153.797, 267.396, -2.225, 7.054, -3.599]
                offset_matrix = pose_to_matrix(offset_pose_cover)
                target_matrix = current_matrix @ offset_matrix
                target_pose = matrix_to_pose(target_matrix)
                # robot.move_linear(CartesianPose(*target_pose))
                take_cover_pose = robot.relative_tool_pose(dz = 50, init_pose = target_pose).to_list()
                # robot.move_by_pose_list(poses = [target_pose , take_cover_pose], speeds = [100, 50])
                current_joint = robot.get_joint_pose()
                target_joint = robot.inverse_kinematics(target_pose=target_pose, initial_joints=current_joint)
                take_cover_joint = robot.inverse_kinematics(target_pose=take_cover_pose, initial_joints=current_joint)

                robot.move_by_joint_list(joints = [target_joint , take_cover_joint], speeds = [25, 15])

                if already_take_inner_cover == False:
                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["PUSH_POINT1"] = target_pose
                        data["PUSH_POINT2"] = take_cover_pose

                    with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    flow.update_CFG()
                    logger.info('PUSH POINT1 POINT2已写入json.')

                # 偏移运动完成，恢复硬件AE
                if args.hw_ae:
                    try:
                        camera.set_exposure_auto(True)
                        camera.set_gain_auto(True)
                        logger.debug("Hardware AE restored after offset move")
                    except Exception:
                        pass

            elif key == ord('g'):
                # 步骤 1: 开大盖
                if outer_cover_is_opened == False:
                    flow.run(1)
                else:
                    flow.run(8)

            elif key == ord('h'):
                # 步骤 2: 夹住小盖并放盖
                flow.run(2)
                already_take_inner_cover = True

            elif key == ord('j'):
                # 步骤 3: 移动到插枪点并沿法兰z insert
                flow.run(3)

            elif key == ord('k'):
                # 步骤 4: 归枪
                flow.run(4)
                already_return_gun = True


            elif key == ord('l'):
                # 步骤 5： 移动到取小盖的对准点
                flow.run(5)
            elif key == ord('p'):
                # 步骤 6：夹住小盖并放回、关大盖
                flow.run(7)
                # 回到初始点
                flow.run(6)
        
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
