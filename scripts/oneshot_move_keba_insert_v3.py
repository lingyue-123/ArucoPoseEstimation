#!/usr/bin/env python3
"""
一次性到位 + 插入运动 | One-Shot Move Insert

检测 ArUco → 计算最终目标位姿 → 一次运动到基准位。
可在任意时刻切到指定工具坐标系执行沿工具 z 轴的插入动作。

工作流：
    1. 实时画面 + ArUco 检测
    2. 按 'r'：记录当前 ArUco 位姿为参考（自动识别取枪/插枪ID），同时保存 tool 0（法兰）下的参考 TCP
    3. 在新的位置按 'm'：基于视觉结果一次运动回基准位（自动适配取枪/插枪）
    4. 按 'b'：直接回到按下 r 时保存的机械臂位置（tool 0）
    5. 按 'i'：切到插入工具坐标系，读 TCP，沿工具 z 轴前进（自动适配取枪/插枪插入距离）
    6. 按 'q' 退出

用法：
    python scripts/oneshot_move_insert.py --camera hikvision_normal --insert-cm-insert 3 --insert-cm-take 5
    python scripts/oneshot_move_insert.py --camera hikvision_normal --insert-cm-insert 3 --insert-cm-take 5 --no-robot
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
from robovision.detection.aruco import (
    ArucoDetector, build_raw_aruco_detector, detect_raw_frame,
)
from robovision.geometry.transforms import (
    rotmat_to_euler, rotation_angle_deg,
    pose_to_matrix, matrix_to_pose, compute_new_tool_pose,
    offset_pose_along_tool_axis,
)
from robovision.robot import build_robot
from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file, save_pose_file
from robovision.visualization.aruco_overlay import draw_aruco_result_for_display, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT
from third_party.robot_driver.robot_driver_interface import CartesianPose

try:
    from magnet_interface import magnet_control, magnet_close
except ImportError:
    def magnet_control(enabled):
        logger.warning("magnet_interface 不可用，跳过磁吸%s", "吸合" if enabled else "断开")
        return False

    def magnet_close():
        return None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

DEFAULT_MAX_TRANS_MM = 370.0
DEFAULT_MAX_ROT_DEG = 35.0
DEFAULT_SPEED = 4
MOVE_TIMEOUT = 30.0
INSERT_TOOL_ID = 2
BASE_TOOL_ID = 0

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICK = 1
PROMPT_TEXT_COLOR = (255, 255, 255)

MARKER_DISPLAY_NAMES = {
    "插枪": "Insert",
    "取枪": "Take",
}


def aruco_to_matrix(data: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data['R_m2c']
    T[:3, 3] = data['tvec_m2c'].flatten()
    return T


def compute_pose_error(T_g2b_cur, T_g2b_target):
    T_delta = np.linalg.inv(T_g2b_cur) @ T_g2b_target
    trans_err = float(np.linalg.norm(T_delta[:3, 3]))
    rot_err = rotation_angle_deg(np.eye(3), T_delta[:3, :3])
    return trans_err, rot_err


def compute_pose_error_in_frame(T_g2b_cur, T_g2b_ref):
    R_ref = T_g2b_ref[:3, :3]
    t_diff_base = T_g2b_cur[:3, 3] - T_g2b_ref[:3, 3]
    trans_xyz = R_ref.T @ t_diff_base
    trans_norm = float(np.linalg.norm(trans_xyz))
    R_delta = R_ref.T @ T_g2b_cur[:3, :3]
    rot_err = rotation_angle_deg(np.eye(3), R_delta)
    return trans_xyz, trans_norm, rot_err


def put_text(img, text, y, color=(200, 200, 200)):
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2)
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def marker_display_name(marker_type):
    return MARKER_DISPLAY_NAMES.get(marker_type, "Unknown")


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


def get_tcp_pose_in_tool_mm(robot, robot_cfg, tool_id, label=None):
    try:
        return tcp_pose_to_mm(get_tcp_pose_in_tool(robot, tool_id, label=label), robot_cfg)
    except Exception as exc:
        logger.debug("读取 tool %s TCP 失败: %s", label or tool_id, exc)
        return None


def execute_move(robot, step_pose_6dof, timeout=MOVE_TIMEOUT):
    cart = CartesianPose(
        x=step_pose_6dof[0], y=step_pose_6dof[1], z=step_pose_6dof[2],
        rx=step_pose_6dof[3], ry=step_pose_6dof[4], rz=step_pose_6dof[5],
    )
    logger.info("执行运动...")
    ok = robot.move_and_wait(cart, timeout=timeout)
    if ok:
        logger.info("运动完成")
    else:
        logger.warning("运动超时或失败")
    return ok


def execute_keba_force_mode(robot, mode, displace_target=0):
    """执行 KEBA mode_1/mode_3 力控动作。"""
    move_force = getattr(robot, 'move_force', None)
    if not callable(move_force):
        logger.warning("当前机械臂驱动不支持 KEBA 力控模式")
        return False

    logger.info("执行 KEBA 力控模式%d...", int(mode))
    ok = move_force(int(mode), displace_target=int(displace_target))
    if ok:
        logger.info("KEBA 力控模式%d已下发", int(mode))
    else:
        logger.warning("KEBA 力控模式%d下发失败", int(mode))
    return bool(ok)


from robovision.robot.tool_coord import (
    ensure_tool_id, get_tcp_pose_in_tool, check_oneshot_safety,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='一次性到位 + 工具坐标系插入')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--hand-eye', type=str, default='data/handeye/hand_eye_result.txt',
                        help='手眼标定结果文件（4x4 矩阵）')
    parser.add_argument('--aruco-ref', type=str, default='data/aruco/aruco_pose_ref.txt',
                        help='参考插枪 ArUco 位姿文件')
    parser.add_argument('--aruco-ref-takegun', type=str, default='data/aruco/aruco_pose_ref_takegun.txt',
                        help='参考取枪 ArUco 位姿文件')
    parser.add_argument('--target-marker', type=int, default=2,
                        help='ArUco Marker ID（插枪，默认 2）')
    parser.add_argument('--takegun-marker', type=int, default=0,
                        help='取枪 ArUco Marker ID（默认 0）')
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
    parser.add_argument('--insert-cm-insert', type=float, default=10,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（插枪，cm）')
    parser.add_argument('--insert-cm-take', type=float, default=35.1,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（取枪，cm）')
    parser.add_argument('--return-gun-up', type=float, default=0.3,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（抬枪，cm）')
    parser.add_argument('--return-gun-left', type=float, default=0.2,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（抬枪，cm）')
    parser.add_argument('--move-timeout', type=float, default=MOVE_TIMEOUT,
                        help=f'普通运动等待超时秒数（默认 {MOVE_TIMEOUT}）')
    parser.add_argument('--force-retract-target', type=int, default=70,
                        help='按 o 执行 KEBA mode_3 力控拔枪的目标位移（默认 70）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（固定 IPPE_SQUARE，且不使用时序滤波）')
    parser.add_argument('--no-temporal-filter', action='store_true',
                        help='关闭 Kalman 角点滤波和位姿时序平滑（仅影响标准检测模式）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 初始化两种插入距离（转换为mm）
    insert_mm_insert = args.insert_cm_insert * 10.0
    insert_mm_take = args.insert_cm_take * 10.0
    return_gun_up = args.return_gun_up * 10
    return_gun_left = args.return_gun_left * 10

    # 磁吸状态
    magnet_state = False

    return_gun_pose = None # 初始归枪位置
    test_pose = None # 归枪抬起位置
    return_gun_cur_pose = None # 归枪抬起后插入前位置

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    T_c2g = load_hand_eye_result(args.hand_eye)
    logger.info("手眼标定加载成功: t=[%.2f, %.2f, %.2f]",
                T_c2g[0, 3], T_c2g[1, 3], T_c2g[2, 3])

    # 插枪参考位姿加载
    T_aruco2cam_ref_insert = None
    if os.path.isfile(args.aruco_ref):
        ref_pose = load_pose_file(args.aruco_ref)[-1]
        T_aruco2cam_ref_insert = pose_to_matrix(ref_pose)
        logger.info("插枪参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref, *ref_pose[:3])
    else:
        logger.info("插枪参考位姿文件不存在，移到参考位置后按 r 键保存")

    # 取枪参考位姿加载
    T_aruco2cam_ref_takegun = None
    if os.path.isfile(args.aruco_ref_takegun):
        ref_pose_take = load_pose_file(args.aruco_ref_takegun)[-1]
        T_aruco2cam_ref_takegun = pose_to_matrix(ref_pose_take)
        logger.info("取枪参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref_takegun, *ref_pose_take[:3])
    else:
        logger.info("取枪参考位姿文件不存在，移到参考位置后按 r 键保存")

    # 插枪TCP参考路径
    tcp_ref_path_insert = args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
    T_g2b_ref_insert = None
    if os.path.isfile(tcp_ref_path_insert):
        tcp_ref_pose = load_pose_file(tcp_ref_path_insert)[-1]
        T_g2b_ref_insert = pose_to_matrix(tcp_ref_pose)
        logger.info("插枪参考 TCP(坐标系0) 已加载: t=[%.2f, %.2f, %.2f] mm", *tcp_ref_pose[:3])

    # 取枪TCP参考路径
    tcp_ref_path_takegun = args.aruco_ref_takegun.replace('aruco_pose_ref_takegun', 'tcp_ref_takegun')
    T_g2b_ref_takegun = None
    if os.path.isfile(tcp_ref_path_takegun):
        tcp_ref_pose_take = load_pose_file(tcp_ref_path_takegun)[-1]
        T_g2b_ref_takegun = pose_to_matrix(tcp_ref_pose_take)
        logger.info("取枪参考 TCP(坐标系0) 已加载: t=[%.2f, %.2f, %.2f] mm", *tcp_ref_pose_take[:3])

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
                logger.info("当前机械臂驱动不支持速度设置（KEBA 等）")
            ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(法兰)")
        else:
            logger.warning("机械臂连接失败")
    else:
        logger.info("--no-robot 模式：不连接机械臂，仅计算")

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
        logger.info("检测模式: RAW (IPPE_SQUARE, 无滤波)")
    else:
        detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
        detector.set_temporal_filter(use_temporal_filter)
        raw_detector = None
        logger.info("检测模式: ArucoDetector (%s, 多方法 PnP)",
                    "时序滤波开启" if use_temporal_filter else "无时序滤波")

    # 标记ID配置
    insert_marker_id = args.target_marker
    takegun_marker_id = args.takegun_marker
    all_marker_ids = {insert_marker_id: "插枪", takegun_marker_id: "取枪"}
    
    # 参考位姿状态
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
    logger.info("就绪。r=设参考  m=视觉回基准位  b=回r点位  i=普通插入  f=力控插枪  o=力控拔枪  q=退出  v=磁吸  1=取枪后拔出 (检测线程已启动)")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            with det_lock:
                aruco_result = latest_det
            
            # 检测当前有效的marker（优先插枪，再取枪）
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

            # 先缩放到显示尺寸，再绘制 overlay
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

            # 选择对应的参考位姿
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

            # ArUco 误差：当前检测位姿 vs 参考位姿（不依赖机械臂连接）
            aruco_trans_xyz = None
            aruco_trans_norm = None
            aruco_rot_err = None
            if T_aruco2cam_ref is not None and target_data is not None:
                T_aruco2cam_cur = aruco_to_matrix(target_data)
                aruco_trans_xyz, aruco_trans_norm, aruco_rot_err = compute_pose_error_in_frame(
                    T_aruco2cam_cur, T_aruco2cam_ref)

            insert_pose_coord2 = None

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

            # 显示插入距离信息
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

            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            ref_str = f"Ref:Insert{'SET' if ref_set_insert else 'NONE'} Take{'SET' if ref_set_takegun else 'NONE'}"
            status = f"ONESHOT+INSERT | {ref_str} | {robot_str} | r=Ref m=Move b=Back i=Insert f=ForceIn o=ForceOut q=Quit"
            put_text(vis, status, h_disp - 15, (140, 140, 140))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                if current_marker_id is None:
                    logger.warning("未检测到插枪/取枪Marker，无法保存参考")
                    continue
                
                # 根据当前检测到的marker类型保存对应的参考位姿
                if current_marker_type == "插枪":
                    T_aruco2cam_ref_insert = aruco_to_matrix(target_data)
                    ref_pose_vec = matrix_to_pose(T_aruco2cam_ref_insert)
                    os.makedirs(os.path.dirname(args.aruco_ref) or '.', exist_ok=True)
                    save_pose_file(args.aruco_ref, np.array([ref_pose_vec]))
                    ref_set_insert = True
                    logger.info("插枪参考位姿已保存: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

                    if T_g2b_cur is not None:
                        T_g2b_ref_insert = T_g2b_cur
                        save_pose_file(tcp_ref_path_insert, np.array([matrix_to_pose(T_g2b_cur)]))
                        logger.info("插枪参考 TCP(坐标系0) 已保存: t=[%.2f, %.2f, %.2f] mm",
                                    T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])
                else:  # 取枪
                    T_aruco2cam_ref_takegun = aruco_to_matrix(target_data)
                    ref_pose_vec = matrix_to_pose(T_aruco2cam_ref_takegun)
                    os.makedirs(os.path.dirname(args.aruco_ref_takegun) or '.', exist_ok=True)
                    save_pose_file(args.aruco_ref_takegun, np.array([ref_pose_vec]))
                    ref_set_takegun = True
                    logger.info("取枪参考位姿已保存: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

                    if T_g2b_cur is not None:
                        T_g2b_ref_takegun = T_g2b_cur
                        save_pose_file(tcp_ref_path_takegun, np.array([matrix_to_pose(T_g2b_cur)]))
                        logger.info("取枪参考 TCP(坐标系0) 已保存: t=[%.2f, %.2f, %.2f] mm",
                                    T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])

            elif key == ord('m'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if T_g2b_target is None:
                    logger.warning("无法计算目标（检查参考位姿/ArUco检测/机械臂连接）")
                    continue

                final_pose = matrix_to_pose(T_g2b_target)
                logger.info(f"回{current_marker_type}基准位(坐标系0): 平移=%.2f mm, 旋转=%.2f deg", trans_err, rot_err)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *final_pose)

                safe, reason = check_oneshot_safety(
                    trans_err, rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info(f"[DRY RUN] 不执行回{current_marker_type}基准位运动")
                else:
                    if not ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(法兰)"):
                        continue
                    moving = True
                    ok = execute_move(robot, final_pose, timeout=args.move_timeout)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            T_g2b_after = pose_to_matrix(new_tcp)
                            residual_trans, residual_rot = compute_pose_error(
                                T_g2b_after, T_g2b_target)
                            logger.info(f"{current_marker_type}运动后残差(vs target): 平移=%.2f mm, 旋转=%.2f deg",
                                        residual_trans, residual_rot)
                            if T_g2b_ref is not None:
                                ref_xyz, ref_res_t, ref_res_r = compute_pose_error_in_frame(
                                    T_g2b_after, T_g2b_ref)
                                logger.info(f"{current_marker_type}运动后残差(vs ref):    平移=%.2f mm, 旋转=%.2f deg",
                                            ref_res_t, ref_res_r)
                                logger.info("  ref 坐标系分轴: dX=%.2f dY=%.2f dZ=%.2f mm",
                                            *ref_xyz)
                            logger.info("  实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

            elif key == ord('b'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                
                # 根据当前marker类型选择对应的参考位姿
                if current_marker_type == "插枪":
                    T_g2b_ref = T_g2b_ref_insert
                    marker_type = "插枪"
                elif current_marker_type == "取枪":
                    T_g2b_ref = T_g2b_ref_takegun
                    marker_type = "取枪"
                else:
                    logger.warning("未检测到插枪/取枪Marker，无法回参考点")
                    continue

                if T_g2b_ref is None:
                    logger.warning(f"尚未保存{marker_type} r 点机械臂位置，请先按 r")
                    continue

                back_pose = matrix_to_pose(T_g2b_ref)
                if T_g2b_cur is not None:
                    back_trans_err, back_rot_err = compute_pose_error(T_g2b_cur, T_g2b_ref)
                else:
                    back_trans_err, back_rot_err = 0.0, 0.0

                logger.info(f"回到{marker_type} r 点位(坐标系0): 平移=%.2f mm, 旋转=%.2f deg",
                            back_trans_err, back_rot_err)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *back_pose)

                safe, reason = check_oneshot_safety(
                    back_trans_err, back_rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info(f"[DRY RUN] 不执行回{marker_type} r 点位运动")
                else:
                    if not ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(回{marker_type}参考点)"):
                        continue
                    moving = True
                    ok = execute_move(robot, back_pose, timeout=args.move_timeout)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            logger.info(f"  回{marker_type}r点位后实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

            elif key == ord('i'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法读取当前 2 号工具坐标系 TCP")
                    continue
                if current_marker_type is None:
                    logger.warning("未检测到插枪/取枪Marker，无法确定插入距离")
                    continue

                # 根据当前marker类型选择插入距离
                if current_marker_type == "插枪":
                    insert_mm = insert_mm_insert
                    insert_cm = args.insert_cm_insert
                    marker_type = "插枪"
                else:
                    insert_mm = insert_mm_take
                    insert_cm = args.insert_cm_take
                    marker_type = "取枪"

                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)")
                if current_tool_pose is None:
                    logger.warning("无法读取当前 2 号工具坐标系 TCP")
                    continue

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(insert_mm), axis='z')
                safe, reason = check_oneshot_safety(abs(insert_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info(f"{marker_type}插入动作(坐标系2): 基于当前实时 TCP，沿工具 z 轴前进 %.2f cm (%.2f mm)",
                            insert_cm, insert_mm)
                logger.info("  坐标系2当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  坐标系2目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info(f"[DRY RUN] 不执行{marker_type}插入运动")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info(f"  {marker_type}插入后坐标系2 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(法兰)")

            elif key == ord('f'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法执行 KEBA 力控插枪")
                    continue
                if args.no_robot:
                    logger.info("[DRY RUN] 不执行 KEBA 力控插枪")
                    continue

                moving = True
                execute_keba_force_mode(robot, 1)
                moving = False

            elif key == ord('o'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法执行 KEBA 力控拔枪")
                    continue
                if args.no_robot:
                    logger.info("[DRY RUN] 不执行 KEBA 力控拔枪")
                    continue

                moving = True
                execute_keba_force_mode(
                    robot,
                    3,
                    displace_target=args.force_retract_target,
                )
                moving = False
            elif key == ord('v'):
                # 切换磁吸状态（吸合 ↔ 断开）
                magnet_state = not magnet_state
                logger.info("尝试%s磁吸装置...", "吸合" if magnet_state else "断开")
                # 调用磁吸控制函数
                success = magnet_control(magnet_state)
                if success:
                    logger.info("磁吸装置%s成功", "吸合" if magnet_state else "断开")
                else:
                    logger.error("磁吸装置%s失败", "吸合" if magnet_state else "断开")
                    # 失败时恢复状态标识（避免状态不一致）
                    magnet_state = not magnet_state

            elif key == ord('1'):
                # 取枪后拔出
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法读取当前 2 号工具坐标系 TCP")
                    continue

                current_tool_pose = get_tcp_pose_in_tool_mm(
                    robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)")
                if current_tool_pose is None:
                    logger.warning("无法读取当前 2 号工具坐标系 TCP")
                    continue

                insert_pose = offset_pose_along_tool_axis(current_tool_pose, float(-insert_mm_take), axis='z') # 注意符号
                safe, reason = check_oneshot_safety(abs(insert_mm_take), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info(f"取枪拔出动作(坐标系2): 基于当前实时 TCP，沿工具 z 轴前进 %.2f cm (%.2f mm)",
                            insert_mm_take, insert_mm_take)
                logger.info("  坐标系2当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  坐标系2目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info(f"[DRY RUN] 不执行取枪拔出运动")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info(f"拔出后坐标系2 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(法兰)")

                    # 记录归枪点
                    return_gun_pose = get_tcp_pose_in_tool_mm(
                        robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)")
                    if return_gun_pose is None:
                        logger.warning("无法读取当前 归枪点 TCP")
                        continue

            elif key == ord('2'):
                # 取枪初始位置
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                
                first_pose = [779.163,-26.174,87.842,175.4801,-71.6582,8.1004]

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行回 取枪 点位运动")
                else:
                    moving = True
                    ok = execute_move(robot, first_pose)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            logger.info("  回取枪点位后实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)
            
            elif key == ord('3'):
                # 插枪初始位置
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                
                first_pose = [305.510,-467.418,214.217,162.5176,-66.2000,-80.3214]
                # first_pose = [-403.7447430954047, -16.869922040947117, 31.830182937853067, -102.25921115206256, 49.67026756967543, 52.61107046443207]

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行回 插枪 点位运动")
                else:
                    moving = True
                    ok = execute_move(robot, first_pose)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            logger.info("  回插枪点位后实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

            elif key == ord('4'):
                # 归枪初始位置
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                
                first_pose = return_gun_pose

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行回 插枪 点位运动")
                else:
                    moving = True
                    ok = execute_move(robot, first_pose)
                    moving = False
                    if ok:
                        new_tcp = get_robot_tcp_pose_mm(robot, robot_cfg)
                        if new_tcp is not None:
                            logger.info("  回插枪点位后实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)
                            
            elif key == ord('5'):
                # 归枪点后插入
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法读取当前 2 号工具坐标系 TCP")
                    continue

                return_gun_cur_pose = get_tcp_pose_in_tool_mm(
                        robot, robot_cfg, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)")

                insert_pose = offset_pose_along_tool_axis(return_gun_cur_pose, float(insert_mm_take - 50), axis='z')
                safe, reason = check_oneshot_safety(abs(insert_mm_take), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info(f"取枪拔出动作(坐标系2): 基于当前实时 TCP，沿工具 z 轴前进 %.2f cm (%.2f mm)",
                            insert_mm_take, insert_mm_take)
                logger.info("  坐标系2当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *return_gun_cur_pose)
                logger.info("  坐标系2目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info(f"[DRY RUN] 不执行取枪拔出运动")
                else:
                    if not ensure_tool_id(robot, INSERT_TOOL_ID, label=f"tool {INSERT_TOOL_ID}(插入工具)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose, timeout=args.move_timeout)
                    moving = False

                    try:
                        tcp_after_insert = get_robot_tcp_pose_mm(robot, robot_cfg)
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info(f"拔出后坐标系2 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_tool_id(robot, BASE_TOOL_ID, label=f"tool {BASE_TOOL_ID}(法兰)")
            elif key == ord('6'):
                robot.arm_move_joint([5.266, -72.876, 70.155, 131.741, 87.884, -84.064])
            elif key == ord('7'):
                robot.arm_move_joint([5.167, -76.472, 61.907, 127.093, 87.977, -84.083])

    finally:
        stop_event.set()
        det_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.disconnect()


if __name__ == '__main__':
    main()
