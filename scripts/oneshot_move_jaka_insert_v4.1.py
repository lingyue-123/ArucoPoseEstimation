#!/usr/bin/env python3
"""
一次性到位 + 插入/取枪运动 | One-Shot Move with Insert/Extract

基于 ArUco 视觉检测 + 手眼标定，一键运动到参考基准位，
并支持在工具坐标系下沿 z 轴执行插入/拔出动作。

工作流 (Workflow):
    1. 按 'r': 记录当前 ArUco 位姿为参考（自动识别插枪/取枪 marker），同步保存法兰坐标系下的参考 TCP
    2. 按 'm': 基于视觉结果一次运动回基准位（自动适配插枪/取枪）
    3. 按 'b': 直接回到按 'r' 时保存的机械臂法兰坐标系位姿
    4. 按 'i': 切换到工具坐标系，沿工具 z 轴前进（自动适配插枪/取枪距离）
    5. 按 'q' / ESC: 退出

用法 (Usage):
    python scripts/oneshot_move_jaka_insert_v4.py --camera mecheye --insert-cm-insert 7 --insert-cm-take 12
    python scripts/oneshot_move_jaka_insert_v4.py --camera mecheye --insert-cm-insert 7 --insert-cm-take 12 --no-robot
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
from relative_move import apply_relative_pose, load_offset_from_file, compute_distance
from Intergration.stereo_camera_calib.yrq.pose_estimation.cover_pose_estimator import CoverPoseEstimator
from Intergration.FTServo_Linux_main.examples.sms_sts_driver import SMSSTSController
# --- 本地辅助 (替代 robovision.robot.tool_coord，适配 CRP 驱动) ---
def check_oneshot_safety(trans_mm, rot_deg, max_trans, max_rot):
    if trans_mm > max_trans:
        return False, (f"平移 {trans_mm:.1f} mm 超过阈值 {max_trans:.1f} mm，请手动移近后重试或增大 --max-trans")
    if rot_deg > max_rot:
        return False, (f"旋转 {rot_deg:.2f} deg 超过阈值 {max_rot:.1f} deg，请手动调整姿态后重试或增大 --max-rot")
    return True, ""


def _marker_roi_brightness(frame, target_data):
    """从 ArUco 检测结果提取 marker ROI 并返回中值亮度。失败返回 None。"""
    corners = target_data.get('filtered_corners') or target_data.get('raw_corners')
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
DEFAULT_MAX_ROT_DEG = 45.0     # 一次性运动允许的最大旋转 (deg)
DEFAULT_SPEED = 50              # 默认运动速度 (%)
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
    parser.add_argument('--hand-eye', type=str, default='data/handeye_eye_in_hand/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考插枪 ArUco 位姿文件')
    parser.add_argument('--aruco-ref-takegun', type=str, default='data/aruco/aruco_pose_ref_takegun.txt',
                        help='参考取枪 ArUco 位姿文件')
    parser.add_argument('--target-marker', type=int, default=0,
                        help='插枪 ArUco Marker ID')
    parser.add_argument('--takegun-marker', type=int, default=2,
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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-ae', action='store_true',
                        help='禁用自动曝光控制（即使配置中已启用）')
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

    # 曝光参考路径（插枪/取枪各一个 txt）
    exp_ref_path_insert = args.aruco_ref.replace('aruco_pose_ref', 'aruco_exp_ref')
    exp_ref_path_takegun = args.aruco_ref_takegun.replace('aruco_pose_ref_takegun', 'aruco_exp_ref_takegun')

    ref_exp_insert = None
    if os.path.isfile(exp_ref_path_insert):
        with open(exp_ref_path_insert, 'r') as f:
            parts = f.read().strip().split(',')
            if len(parts) >= 3:
                ref_exp_insert = (float(parts[0]), float(parts[1]), float(parts[2]))
                logger.info("Insert exp ref loaded: exp=%.0fus gain=%.1fdB bri=%.0f", *ref_exp_insert)

    ref_exp_takegun = None
    if os.path.isfile(exp_ref_path_takegun):
        with open(exp_ref_path_takegun, 'r') as f:
            parts = f.read().strip().split(',')
            if len(parts) >= 3:
                ref_exp_takegun = (float(parts[0]), float(parts[1]), float(parts[2]))
                logger.info("Take exp ref loaded: exp=%.0fus gain=%.1fdB bri=%.0f", *ref_exp_takegun)

    # 加载取枪相对位移偏移量
    loaded_offset = load_offset_from_file(args.takegun_offset_file)
    if loaded_offset is not None:
        takegun_offset = loaded_offset
    
    charging_offset = load_offset_from_file(args.charging_offset_file)


    # --- 初始化机器人 (CoverActionFlow 单例) | Robot init ---
    flow = CoverActionFlow.get_instance(ip=args.robot_ip)
    robot = None
    robot_connected = False
    if not args.no_robot:
        robot_connected = flow.connect()
        if robot_connected:
            robot = flow.arm
            robot.set_speed(args.speed)
            logger.info("Speed set to %d%%: OK", args.speed)
        else:
            logger.warning("Robot connection failed")
    else:
        logger.info("--no-robot mode: no robot connection, calculation only")

    # --- 初始化相机和检测器 | Camera & detector init ---
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    # 自动曝光控制器（仅当相机配置启用 + 相机支持时生效）
    ae_ctrl = None
    cam_cfg = cfg.get_camera(args.camera)
    if cam_cfg.auto_exposure and cam_cfg.auto_exposure.enabled and not args.no_ae:
        ae_cfg = cam_cfg.auto_exposure
        try:
            ae_ctrl = AutoExposureController(
                camera,
                target_brightness=ae_cfg.target_brightness,
                deadband=ae_cfg.deadband,
                adjust_interval=ae_cfg.adjust_interval,
                exposure_limit_ms=ae_cfg.exposure_limit_ms,
                gain_limit_db=ae_cfg.gain_limit_db,
            )
            ae_ctrl.setup()
            logger.info("Auto exposure controller initialized")
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
            if ae_ctrl is not None:
                ae_ctrl.adjust_after_detect(detection_ok=bool(result), detected_markers=result)

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()
    logger.info("Ready: r=SetRef m=Move2Baseline b=Back2Ref i=Insert f=ForceIn o=ForceOut v=Gripper q=Quit (detection thread started)")

    # 双目相机及检测模型初始化 | Stereo cover pose estimator
    cover_pose_estimator = CoverPoseEstimator()

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

                    # 保存曝光参考
                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            exp_us = camera.get_exposure_time()
                            gain_db = camera.get_gain()
                            if bri is not None and exp_us is not None and gain_db is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_insert) or '.', exist_ok=True)
                                with open(exp_ref_path_insert, 'w') as f:
                                    f.write(f"{exp_us:.1f},{gain_db:.2f},{bri:.1f}\n")
                                ref_exp_insert = (exp_us, gain_db, bri)
                                logger.info("Insert exp ref saved: exp=%.0fus gain=%.1fdB bri=%.0f",
                                            exp_us, gain_db, bri)
                        except Exception:
                            logger.warning("Failed to save insert exp ref", exc_info=True)
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

                    # 保存曝光参考
                    if det_frame is not None:
                        try:
                            bri = _marker_roi_brightness(det_frame, target_data)
                            exp_us = camera.get_exposure_time()
                            gain_db = camera.get_gain()
                            if bri is not None and exp_us is not None and gain_db is not None:
                                os.makedirs(os.path.dirname(exp_ref_path_takegun) or '.', exist_ok=True)
                                with open(exp_ref_path_takegun, 'w') as f:
                                    f.write(f"{exp_us:.1f},{gain_db:.2f},{bri:.1f}\n")
                                ref_exp_takegun = (exp_us, gain_db, bri)
                                logger.info("Take exp ref saved: exp=%.0fus gain=%.1fdB bri=%.0f",
                                            exp_us, gain_db, bri)
                        except Exception:
                            logger.warning("Failed to save take exp ref", exc_info=True)

            elif key == ord('m'):
                # 视觉伺服: 一次运动到参考基准位
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                if T_g2b_target is None:
                    logger.warning("Cannot compute target (check ref pose / marker detection / robot connection)")
                    continue

                final_pose = matrix_to_pose(T_g2b_target)
                logger.info("Return to %s baseline (tool %d): trans=%.2f mm, rot=%.2f deg",
                            current_marker_type, BASE_TOOL_ID, trans_err, rot_err)
                logger.info("  Target TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *final_pose)

                safe, reason = check_oneshot_safety(
                    trans_err, rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                # 曝光收敛：对准前让marker亮度对齐到录制参考时的状态
                ref_exp = ref_exp_insert if current_marker_type == "插枪" else ref_exp_takegun
                if ref_exp is not None and not args.no_robot:
                    ref_exp_us, ref_gain_db, ref_bri = ref_exp
                    try:
                        camera.set_exposure_auto(False)
                        camera.set_exposure_time(ref_exp_us)
                        camera.set_gain(ref_gain_db)
                        logger.info("Exp locked to ref: %.0fus %.1fdB target_bri=%.0f",
                                    ref_exp_us, ref_gain_db, ref_bri)
                        for i in range(20):
                            time.sleep(0.03)
                            with det_lock:
                                cur_result, cur_frame = latest_det, latest_det_frame
                            if not cur_result or cur_frame is None:
                                continue
                            for mid, d in cur_result.items():
                                if mid in (insert_marker_id, takegun_marker_id):
                                    cur_bri = _marker_roi_brightness(cur_frame, d)
                                    if cur_bri is not None:
                                        err = cur_bri - ref_bri
                                        if abs(err) <= 5:
                                            logger.info("Bri converged: %.0f->%.0f (err=%.1f iters=%d)",
                                                        ref_bri, cur_bri, err, i + 1)
                                            break
                                        new_exp = ref_exp_us * max(0.5, min(2.0, 1.0 - err / 640.0))
                                        new_exp = max(100, min(50000, new_exp))
                                        camera.set_exposure_time(new_exp)
                                    break
                    except Exception:
                        logger.warning("Exposure convergence failed", exc_info=True)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip %s baseline move", current_marker_type)
                else:
                    moving = True
                    robot.set_speed(80)
                    ok = execute_move(robot, final_pose, timeout=args.move_timeout)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            T_g2b_after = pose_to_matrix(new_tcp)
                            residual_trans, residual_rot = compute_pose_error(
                                T_g2b_after, T_g2b_target)
                            logger.info("%s residual (vs target): trans=%.2f mm, rot=%.2f deg",
                                        current_marker_type, residual_trans, residual_rot)
                            if T_g2b_ref is not None:
                                ref_xyz, ref_res_t, ref_res_r = compute_pose_error_in_frame(
                                    T_g2b_after, T_g2b_ref)
                                logger.info("%s residual (vs ref):    trans=%.2f mm, rot=%.2f deg",
                                            current_marker_type, ref_res_t, ref_res_r)
                                logger.info("  ref axes: dX=%.2f dY=%.2f dZ=%.2f mm", *ref_xyz)
                            logger.info("  Actual TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

                        # 对准完成，恢复硬件AE
                        try:
                            camera.set_exposure_auto(True)
                            camera.set_gain_auto(True)
                            logger.debug("Hardware AE restored after move")
                        except Exception:
                            pass

            elif key == ord('b'):
                # 直接回到按下 'r' 时保存的机械臂 TCP 位置
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                
                if current_marker_type == "插枪":
                    T_g2b_ref = T_g2b_ref_insert
                    marker_type = "插枪"
                elif current_marker_type == "取枪":
                    T_g2b_ref = T_g2b_ref_takegun
                    marker_type = "取枪"
                else:
                    logger.warning("No marker detected, cannot identify reference")
                    continue

                if T_g2b_ref is None:
                    logger.warning("No %s reference TCP saved; press 'r' first", marker_type)
                    continue

                back_pose = matrix_to_pose(T_g2b_ref)
                if T_g2b_cur is not None:
                    back_trans_err, back_rot_err = compute_pose_error(T_g2b_cur, T_g2b_ref)
                else:
                    back_trans_err, back_rot_err = 0.0, 0.0

                logger.info("Return to %s ref point (tool %d): trans=%.2f mm, rot=%.2f deg",
                            marker_type, BASE_TOOL_ID, back_trans_err, back_rot_err)
                logger.info("  Target TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *back_pose)

                safe, reason = check_oneshot_safety(
                    back_trans_err, back_rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info("[DRY RUN] Skip %s ref point move", marker_type)
                else:
                    if not ensure_tool_id(robot, BASE_TOOL_ID,
                                          label=f"tool {BASE_TOOL_ID}(back to {marker_type} ref)"):
                        continue
                    moving = True
                    ok = execute_move(robot, back_pose, timeout=args.move_timeout)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            logger.info("  Back to %s ref: actual TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        marker_type, *new_tcp)

            elif key == ord('e'):
                # 取枪: 缓慢插入 20mm (接续 'w' 两段运动之后)
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

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, 30.0, axis='z')
                safe, reason = check_oneshot_safety(20.0, 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("Take gun: slow advance 20 mm along tool Z-axis")
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  Target TCP:  X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip take-gun slow advance (20mm)")
                else:
                    moving = True
                    robot.set_speed(10)
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After take-gun slow advance TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

            elif key == ord('s'):
                # 取枪: 后退两段 (-40mm 慢速 + -110mm 快速)
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
                robot.set_speed(15)
                insert_pose1 = offset_pose_along_tool_axis(current_tool_pose, -50.0, axis='z')
                robot.set_speed(100)
                insert_pose2 = offset_pose_along_tool_axis(current_tool_pose, -110.0, axis='z')

                logger.info("Take gun: retract along tool Z-axis (step1 -40mm, step2 -110mm)")
                logger.info("  Current TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] Skip take-gun retract")
                else:
                    moving = True
                    robot.set_speed(15)
                    ok = execute_move(robot, insert_pose1, timeout=args.move_timeout)
                    robot.set_speed(100)
                    ok = execute_move(robot, insert_pose2, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  After take-gun retract TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

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
            
            elif key == ord('3'):
                # 插座盖 3D 位姿估计 → 粗定位运动 (CoverPoseEstimator)
                if moving:
                    logger.warning("Motion in progress, please wait")
                    continue
                
                cover_3D_pose = cover_pose_estimator.pose_estimation()
                if cover_3D_pose is not None:
                    logger.info("Cover 3D pose: %s", cover_3D_pose)
                    cover_3D_pose[0] += 65
                    cover_3D_pose[1] -= 15
                    cover_3D_pose[2] -= 450

                    # z_offset = [cover_3D_pose[0][0] + 65, cover_3D_pose[1][0] - 15, cover_3D_pose[2][0] - 450, 0, 0, 0]
                    z_offset_matrix = pose_to_matrix(cover_3D_pose)
                    
                    current_tool_pose = get_tcp_pose_in_tool_mm(
                        robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                    if current_tool_pose is None:
                        logger.warning("Failed to read tool %d TCP", INSERT_TOOL_ID)
                        continue
                    cur_tcp_matrix = pose_to_matrix(current_tool_pose)
                    target_matrix = cur_tcp_matrix @ z_offset_matrix
                    target_pose = matrix_to_pose(target_matrix)
                    # target_pose[3] = 106.123
                    # target_pose[4] = -23.693
                    # target_pose[5] = 30.777

                    logger.info("  Target TCP for coarse alignment: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                *target_pose)
                    # robot.move_linear(CartesianPose(*target_pose))
                    # flow.run_open_cover(target_pose)
                           
            elif key == ord('5'):

                current_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(insert)")
                if current_pose is None:
                    logger.warning("Failed to read current pose")
                    continue

                target_pose = apply_relative_pose(current_pose, charging_offset)

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
                # 取枪: 拔出 (使用命令行 --out-mm 参数)
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

                retract_pose = offset_pose_along_tool_axis(current_tool_pose, -100.0, axis='z')

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
                    robot.set_speed(50)
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

                target_pose = apply_relative_pose(current_pose, takegun_offset)
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
                            advance_pose = offset_pose_along_tool_axis(current_tool_pose, 120.0, axis='z')
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


            elif key == ord('g'):
                # 工艺流程: 步骤 1
                # TODO:直线运动 -> 关节运动 / 开盖后移动到取枪对准点的运动
                flow.run(1)
            elif key == ord('h'):
                # 工艺流程: 步骤 2
                # TODO:直线运动 -> 关节运动 / 取枪对准点 → 插枪对准点的运动
                flow.run(2)
            elif key == ord('j'):
                # 工艺流程: 步骤 3
                # TODO:直线运动 -> 关节运动 / 归枪运动
                flow.run(3)
            elif key == ord('k'):
                # 工艺流程: 步骤 4
                # TODO:直线运动 -> 关节运动 / 归枪后 -> 关盖点运动 & 关盖后 -> 回到机械臂初始姿态的运动
                flow.run(4)
                flow.run(5)
        
    finally:
        stop_event.set()
        det_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()
        if robot_connected:
            flow.disconnect()


if __name__ == '__main__':
    main()
