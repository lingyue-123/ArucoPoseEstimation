#!/usr/bin/env python3
"""
一次性到位运动 | One-Shot Move

检测 ArUco → 计算最终目标位姿 → 一次运动到位。
与 visual_servo.py 的区别：无 gain 缩放、无 SLERP 插值、安全阈值更大。
支持按 i 键进行插入操作：切换到 2 号工具坐标系，沿 z 轴前进指定距离。

工作流：
    1. 实时画面 + ArUco 检测
    2. 按 'r'：记录当前 ArUco 位姿为参考
    3. 按 'm'：计算最终目标（gain=1.0），安全检查后一次到位
    4. 按 'i'：实时读取当前 2 号工具坐标系 TCP，沿局部 z 轴前进 --insert-cm
    5. 按 'o'：实时读取当前 2 号工具坐标系 TCP，沿局部 z 轴后退 --insert-cm（反插入）
    6. 运动完成后继续显示画面，可观察残差或再按 'm' 微调
    7. 按 'q' 退出

用法：
    # 仅计算，不连机械臂
    python scripts/oneshot_move.py --camera hikvision_normal --no-robot

    # 实际运动
    python scripts/oneshot_move.py --camera hikvision_normal

    # 自定义安全阈值
    python scripts/oneshot_move.py --camera hikvision_normal --max-trans 300 --max-rot 45

    # 插入操作
    python scripts/oneshot_move.py --camera hikvision_normal --insert-cm 5
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
import math

JAKA_SDK_PATH = "/home/nvidia/Downloads/jaka-python-sdk"
if sys.platform.startswith("linux"):
    os.environ["LD_LIBRARY_PATH"] = f"{JAKA_SDK_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"
sys.path.insert(0, JAKA_SDK_PATH)

import jkrc

from robovision.cameras import build_camera
from robovision.cameras.base import CameraIntrinsics
from robovision.config.loader import get_config
from robovision.detection.aruco import ArucoDetector
from robovision.geometry.transforms import (
    rotmat_to_euler, rotation_angle_deg,
    pose_to_matrix, matrix_to_pose, compute_new_tool_pose,
    offset_pose_along_tool_axis,
)
from robovision.robot import build_robot
from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file, save_pose_file
from robovision.visualization.aruco_overlay import draw_aruco_result
from third_party.robot_driver.robot_driver_interface import CartesianPose

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

DEFAULT_MAX_TRANS_MM = 350.0
DEFAULT_MAX_ROT_DEG = 30.0
DEFAULT_SPEED = 10
MOVE_TIMEOUT = 30.0
INSERT_COORD_SYS = 2
BASE_COORD_SYS = 0
COORD_SWITCH_TIMEOUT = 1.0
COORD_SWITCH_SETTLE_SEC = 0.2

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.65
_FONT_THICK = 2


def aruco_to_matrix(data: dict) -> np.ndarray:
    """ArUco 检测结果 dict → 4x4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = data['R_m2c']
    T[:3, 3] = data['tvec_m2c'].flatten()
    return T

def compute_pose_error(T_g2b_cur, T_g2b_target):
    """计算当前位姿与目标位姿之间的误差 (trans_mm, rot_deg)。"""
    T_delta = np.linalg.inv(T_g2b_cur) @ T_g2b_target
    trans_err = float(np.linalg.norm(T_delta[:3, 3]))
    rot_err = rotation_angle_deg(np.eye(3), T_delta[:3, :3])
    return trans_err, rot_err

def compute_pose_error_in_frame(T_g2b_cur, T_g2b_ref):
    """计算当前 TCP 相对于参考 TCP 的误差，分解到参考坐标系的 x/y/z 轴。"""
    R_ref = T_g2b_ref[:3, :3]
    t_diff_base = T_g2b_cur[:3, 3] - T_g2b_ref[:3, 3]
    trans_xyz = R_ref.T @ t_diff_base
    trans_norm = float(np.linalg.norm(trans_xyz))
    R_delta = R_ref.T @ T_g2b_cur[:3, :3]
    rot_err = rotation_angle_deg(np.eye(3), R_delta)
    return trans_xyz, trans_norm, rot_err

def put_text(img, text, y, color=(200, 200, 200)):
    """基础文字绘制。"""
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def get_coord_sys(robot):
    if robot is None or not hasattr(robot, 'get_coord_sys'):
        return None
    try:
        return robot.get_coord_sys()
    except Exception as exc:
        logger.warning("读取坐标系失败: %s", exc)
        return None


def set_coord_sys(robot, coord, label=None):
    if robot is None or not hasattr(robot, 'set_coord_sys'):
        return True
    try:
        ret = robot.set_coord_sys(coord)
    except Exception as exc:
        logger.warning("坐标系切换到 %s 失败: %s", coord, exc)
        return False

    ok = (ret == 0 or ret is True)
    desc = label or str(coord)
    if ok:
        logger.info("坐标系设置 → %s", desc)
    else:
        logger.warning("坐标系设置失败 → %s (%s)", desc, ret)
    return ok


def ensure_coord_sys(robot, coord, label=None,
                     timeout=COORD_SWITCH_TIMEOUT,
                     settle_sec=COORD_SWITCH_SETTLE_SEC):
    """切换并等待坐标系真正生效，避免读位姿或发运动时仍处于旧坐标系。"""
    if robot is None:
        return False
    if not set_coord_sys(robot, coord, label=label):
        return False

    deadline = time.time() + float(timeout)
    last_coord = None
    while time.time() < deadline:
        last_coord = get_coord_sys(robot)
        if last_coord == coord:
            if settle_sec > 0:
                time.sleep(float(settle_sec))
            return True
        time.sleep(0.05)

    logger.warning("坐标系切换超时，期望=%s 实际=%s", coord, last_coord)
    return False

def execute_move(robot, step_pose_6dof, timeout=MOVE_TIMEOUT):
    """封装 CartesianPose 创建 + move_and_wait。"""
    cart = CartesianPose(
        x=step_pose_6dof[0], y=step_pose_6dof[1], z=step_pose_6dof[2],
        rx=step_pose_6dof[3], ry=step_pose_6dof[4], rz=step_pose_6dof[5],
    )
    
    x = cart.x
    y = cart.y
    z = cart.z
    rx_rad = math.radians(cart.rx)  # 角度转弧度
    ry_rad = math.radians(cart.ry)
    rz_rad = math.radians(cart.rz)
    tcp_pos = [x, y, z, rx_rad, ry_rad, rz_rad]
    
    move_speed = 30
    move_block = True
    move_mode = 0 # 0为绝对运动/1为相对运动
    
    logger.info("执行运动...")
    ok = robot.move_and_wait(tcp_pos, move_mode, move_block, move_speed) 
    if ok:
        logger.info("运动完成")
    else:
        logger.warning("运动超时或失败")
    return ok

def detect_raw(gray, aruco_detector, valid_ids, marker_sizes, K, dist):
    """检测：detectMarkers → cornerSubPix → solvePnP(IPPE_SQUARE) + 重投影误差。"""
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return {}

    result = {}
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, mid in enumerate(ids.flatten()):
        mid = int(mid)
        if mid not in valid_ids:
            continue

        c = corners[i].reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray_blur, c, (5, 5), (-1, -1), criteria)
        img_pts = c.reshape(4, 2).astype(np.float64)

        marker_len = marker_sizes.get(mid, 100.0)
        half = marker_len / 2
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue

        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        reproj_err = float(np.mean(np.linalg.norm(
            proj_pts.reshape(-1, 2) - img_pts, axis=1)))

        R, _ = cv2.Rodrigues(rvec)
        euler_zyx = rotmat_to_euler(R, order='ZYX')
        result[mid] = {
            'raw_corners': corners[i].reshape(4, 2),
            'filtered_corners': img_pts,
            'rvec_m2c': rvec, 'tvec_m2c': tvec, 'R_m2c': R,
            'euler_m2c_zyx': euler_zyx,
            'reproj_err': reproj_err,
            'method': 'IPPE_SQ', 'status': 'OK',
            'marker_length': marker_len,
        }
    return result

def check_oneshot_safety(trans_mm, rot_deg, max_trans, max_rot):
    """检查一次性运动是否超出安全阈值。"""
    if trans_mm > max_trans:
        return False, (f"平移 {trans_mm:.1f} mm 超过阈值 {max_trans:.1f} mm，"
                       "请手动移近后重试或增大 --max-trans")
    if rot_deg > max_rot:
        return False, (f"旋转 {rot_deg:.2f} deg 超过阈值 {max_rot:.1f} deg，"
                       "请手动调整姿态后重试或增大 --max-rot")
    return True, ""

def parse_args():
    parser = argparse.ArgumentParser(description='一次性到位运动（One-Shot Move）')
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
    parser.add_argument('--max-trans', type=float, default=DEFAULT_MAX_TRANS_MM,
                        help=f'最大允许平移 mm（默认 {DEFAULT_MAX_TRANS_MM}）')
    parser.add_argument('--max-rot', type=float, default=DEFAULT_MAX_ROT_DEG,
                        help=f'最大允许旋转 deg（默认 {DEFAULT_MAX_ROT_DEG}）')
    parser.add_argument('--speed', type=int, default=DEFAULT_SPEED,
                        help=f'运动速度 %%（默认 {DEFAULT_SPEED}）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（IPPE_SQUARE，无 Kalman/SLERP 滤波）')
    parser.add_argument('--insert-cm', type=float, default=15.0,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（cm）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    insert_mm = args.insert_cm * 10.0

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    # 手眼标定
    T_c2g = load_hand_eye_result(args.hand_eye)
    logger.info("手眼标定加载成功: t=[%.2f, %.2f, %.2f]",
                T_c2g[0, 3], T_c2g[1, 3], T_c2g[2, 3])

    # 参考 ArUco 位姿
    T_aruco2cam_ref = None
    if os.path.isfile(args.aruco_ref):
        ref_pose = load_pose_file(args.aruco_ref)[-1]
        T_aruco2cam_ref = pose_to_matrix(ref_pose)
        logger.info("参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref, *ref_pose[:3])
    else:
        logger.info("参考位姿文件不存在，移到参考位置后按 r 键保存")

    # 参考 TCP
    tcp_ref_path = args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
    T_g2b_ref = None
    if os.path.isfile(tcp_ref_path):
        tcp_ref_pose = load_pose_file(tcp_ref_path)[-1]
        T_g2b_ref = pose_to_matrix(tcp_ref_pose)
        logger.info("参考 TCP 已加载: t=[%.2f, %.2f, %.2f] mm", *tcp_ref_pose[:3])

    # 机械臂
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
            if hasattr(robot, 'set_coord_sys'):
                ret = robot.set_coord_sys(0)
                logger.info("坐标系设置 → 0(法兰): %s",
                            "OK" if ret == 0 else f"失败({ret})")
        else:
            logger.warning("机械臂连接失败")
    else:
        logger.info("--no-robot 模式：不连接机械臂，仅计算")

    # 相机与检测器
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    # 检测器初始化
    use_raw = args.raw
    if use_raw:
        dict_name = marker_cfg.dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        raw_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        valid_ids = set(marker_cfg.valid_ids)
        marker_sizes = marker_cfg.marker_sizes
        detector = None
        logger.info("检测模式: RAW (IPPE_SQUARE, 无滤波)")
    else:
        detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
        raw_detector = None
        logger.info("检测模式: ArucoDetector (Kalman + 多方法 PnP)")

    target_id = args.target_marker
    ref_set = T_aruco2cam_ref is not None
    moving = False

    win = f"OneShot [{args.camera}] ID={target_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # ——— 后台检测线程 ———
    det_lock = threading.Lock()       # 保护 latest_det / latest_det_frame
    latest_det = {}                   # 最新检测结果
    latest_det_frame = None           # 检测时使用的帧（用于叠加绘制）
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
                result = detect_raw(gray, raw_detector, valid_ids, marker_sizes, K, dist)
            else:
                result = detector.detect(frm)
            with det_lock:
                latest_det = result
                latest_det_frame = frm

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()
    logger.info("就绪。r=设参考  m=一次到位  b=回r点  i=插入  o=反插入  q=退出  (检测线程已启动)")

    try:
        while True:
            # 读实时帧用于显示
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            # 非阻塞获取最新检测结果
            with det_lock:
                aruco_result = latest_det
            target_data = aruco_result.get(target_id)

            # 获取机械臂 TCP
            tcp = None
            T_g2b_cur = None
            if robot_connected:
                try:
                    tcp = robot.get_tcp_pose()
                except Exception:
                    pass
                if tcp is not None:
                    T_g2b_cur = pose_to_matrix(tcp)

            # 在大图上绘制完整 ArUco 叠加（坐标轴、角点、位姿信息）
            vis = draw_aruco_result(
                frame, aruco_result, intrinsics,
                robot_pose=tcp,
                use_kalman=False,
                robot_connected=robot_connected,
            )
            h_orig, w_orig = vis.shape[:2]
            y = 35 + len(aruco_result) * 100 + 80

            if not target_data:
                put_text(vis, f"ID{target_id} NOT DETECTED", y, (0, 0, 255))
                y += 40

            # 计算目标和误差
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

            # 优先用参考 TCP 误差显示
            display_trans = trans_err
            display_rot = rot_err
            ref_trans_xyz = None
            if T_g2b_ref is not None and T_g2b_cur is not None:
                ref_trans_xyz, display_trans, display_rot = compute_pose_error_in_frame(T_g2b_cur, T_g2b_ref)

            # OSD: 误差和目标
            if display_trans is not None:
                if display_trans < 2.0:
                    err_color = (0, 255, 0)
                elif display_trans < 10.0:
                    err_color = (0, 200, 255)
                else:
                    err_color = (0, 0, 255)
                put_text(vis, f"Error: {display_trans:.2f} mm  {display_rot:.2f} deg", y, err_color)
                y += 30
                if ref_trans_xyz is not None:
                    put_text(vis, f"  dX={ref_trans_xyz[0]:+.2f}  dY={ref_trans_xyz[1]:+.2f}  dZ={ref_trans_xyz[2]:+.2f} mm",
                             y, err_color)
                    y += 28

                if target_pose is not None:
                    put_text(vis, f"Target: X={target_pose[0]:.1f} Y={target_pose[1]:.1f} Z={target_pose[2]:.1f}",
                             y, (255, 200, 0))
                    y += 28
                    put_text(vis, f"        Rx={target_pose[3]:.2f} Ry={target_pose[4]:.2f} Rz={target_pose[5]:.2f}",
                             y, (255, 200, 0))
                    y += 30
            elif not ref_set:
                put_text(vis, "Press r at ref position to set reference", y, (0, 165, 255))
                y += 30

            if moving:
                put_text(vis, "MOVING...", y, (0, 100, 255))
                y += 30

            # 状态栏
            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            ref_str = "Ref:SET" if ref_set else "Ref:NONE"
            status = f"ONESHOT [{args.camera}] | {ref_str} | {robot_str} | r=Ref m=Move b=Back i=Insert o=Reverse q=Quit"
            put_text(vis, status, h_orig - 20, (140, 140, 140))

            # resize 后显示
            scale = 1
            disp = cv2.resize(vis, (int(w_orig * scale), int(h_orig * scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                # 保存当前 ArUco 检测为参考
                if target_data is None:
                    logger.warning("未检测到 ID%d，无法保存参考", target_id)
                    continue
                T_aruco2cam_ref = aruco_to_matrix(target_data)
                ref_pose_vec = matrix_to_pose(T_aruco2cam_ref)
                os.makedirs(os.path.dirname(args.aruco_ref) or '.', exist_ok=True)
                save_pose_file(args.aruco_ref, np.array([ref_pose_vec]))
                ref_set = True
                logger.info("参考位姿已保存: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

                if T_g2b_cur is not None:
                    T_g2b_ref = T_g2b_cur
                    save_pose_file(tcp_ref_path, np.array([matrix_to_pose(T_g2b_cur)]))
                    logger.info("参考 TCP 已保存: t=[%.2f, %.2f, %.2f] mm",
                                T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])

            elif key == ord('m'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if T_g2b_target is None:
                    logger.warning("无法计算目标（检查参考位姿/ArUco检测/机械臂连接）")
                    continue

                # 直接使用最终目标（gain=1.0，无 SLERP）
                final_pose = matrix_to_pose(T_g2b_target)

                logger.info("一次到位: 平移=%.2f mm, 旋转=%.2f deg", trans_err, rot_err)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *final_pose)

                safe, reason = check_oneshot_safety(
                    trans_err, rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行运动")
                else:
                    moving = True
                    ok = execute_move(robot, final_pose)
                    moving = False
                    if ok:
                        # 运动后读取新 TCP 并计算残差
                        new_tcp = robot.get_tcp_pose()
                        if new_tcp is not None:
                            T_g2b_after = pose_to_matrix(new_tcp)
                            # 对比目标 TCP 的残差
                            residual_trans, residual_rot = compute_pose_error(
                                T_g2b_after, T_g2b_target)
                            logger.info("运动后残差(vs target): 平移=%.2f mm, 旋转=%.2f deg",
                                        residual_trans, residual_rot)
                            # 对比参考 TCP 的残差（分轴）
                            if T_g2b_ref is not None:
                                ref_xyz, ref_res_t, ref_res_r = compute_pose_error_in_frame(
                                    T_g2b_after, T_g2b_ref)
                                logger.info("运动后残差(vs ref):    平移=%.2f mm, 旋转=%.2f deg",
                                            ref_res_t, ref_res_r)
                                logger.info("  ref 坐标系分轴: dX=%.2f dY=%.2f dZ=%.2f mm",
                                            *ref_xyz)
                            logger.info("  实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

            elif key == ord('b'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if T_g2b_ref is None:
                    logger.warning("尚未保存 r 点机械臂位置，请先按 r")
                    continue

                back_pose = matrix_to_pose(T_g2b_ref)
                if T_g2b_cur is not None:
                    back_trans_err, back_rot_err = compute_pose_error(T_g2b_cur, T_g2b_ref)
                else:
                    back_trans_err, back_rot_err = 0.0, 0.0

                logger.info("回到 r 点位(坐标系0): 平移=%.2f mm, 旋转=%.2f deg",
                            back_trans_err, back_rot_err)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *back_pose)

                safe, reason = check_oneshot_safety(
                    back_trans_err, back_rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行回 r 点位运动")
                else:
                    if not ensure_coord_sys(robot, BASE_COORD_SYS, label=f"{BASE_COORD_SYS}(回r点位坐标系)"):
                        continue
                    moving = True
                    ok = execute_move(robot, back_pose)
                    moving = False
                    if ok:
                        new_tcp = robot.get_tcp_pose()
                        if new_tcp is not None:
                            logger.info("  回r点位后实际 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                        *new_tcp)

            elif key == ord('i'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if args.insert_cm <= 0:
                    logger.warning("请通过 --insert-cm 指定插入距离（cm）")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法读取当前 TCP位姿，无法执行插入")
                    continue

                # 设置工具0
                if hasattr(robot, 'set_tool_id'):
                    try:
                        rt = robot.set_tool_id(0)
                        logger.info("设置 tool_id=0: %s", rt)
                    except Exception as exc:
                        logger.warning("设置 tool_id=0 失败: %s", exc)
                        continue
                else:
                    logger.warning("不支持 set_tool_id 接口，无法进行插入")
                    continue

                current_tool_pose = robot.get_tcp_pose()
                if current_tool_pose is None:
                    logger.warning("无法读取当前 TCP位姿，无法执行插入")
                    continue

                # 当前位姿 * Z 轴平移矩阵
                current_matrix = pose_to_matrix(current_tool_pose)
                last_z_trans = np.eye(4)
                last_z_trans[2, 3] = insert_mm
                insert_matrix = current_matrix @ last_z_trans
                insert_pose = matrix_to_pose(insert_matrix)

                safe, reason = check_oneshot_safety(abs(insert_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("插入动作: 基于当前实时 TCP，沿工具 z 轴前进 %.2f cm (%.2f mm)",
                            args.insert_cm, insert_mm)
                logger.info("  当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行插入运动")
                else:
                    # if not ensure_coord_sys(robot, INSERT_COORD_SYS, label=f"{INSERT_COORD_SYS}(插入坐标系)"):
                    #     continue
                    moving = True
                    ok = execute_move(robot, insert_pose)
                    moving = False

                    try:
                        tcp_after_insert = robot.get_tcp_pose()
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  插入后 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

                    ensure_coord_sys(robot, BASE_COORD_SYS, label=f"{BASE_COORD_SYS}(恢复基准坐标系)")

            elif key == ord('o'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if args.insert_cm <= 0:
                    logger.warning("请通过 --insert-cm 指定反插入距离（cm）")
                    continue
                if not robot_connected:
                    logger.warning("机械臂未连接，无法执行反插入操作")
                    continue

                # 设置工具0
                if hasattr(robot, 'set_tool_id'):
                    try:
                        rt = robot.set_tool_id(0)
                        logger.info("设置 tool_id=0: %s", rt)
                    except Exception as exc:
                        logger.warning("设置 tool_id=0 失败: %s", exc)
                        continue
                else:
                    logger.warning("不支持 set_tool_id 接口，无法进行反插入")
                    continue

                current_tool_pose = robot.get_tcp_pose()
                if current_tool_pose is None:
                    logger.warning("无法读取当前 TCP位姿，无法执行反插入")
                    continue

                # 当前位姿 * Z 轴反向平移矩阵
                current_matrix = pose_to_matrix(current_tool_pose)
                last_z_trans = np.eye(4)
                last_z_trans[2, 3] = -insert_mm  # 反向移动
                reverse_insert_matrix = current_matrix @ last_z_trans
                reverse_insert_pose = matrix_to_pose(reverse_insert_matrix)

                safe, reason = check_oneshot_safety(abs(insert_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("反插入动作: 基于当前实时 TCP，沿工具 z 轴后退 %.2f cm (%.2f mm)",
                            args.insert_cm, insert_mm)
                logger.info("  当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *reverse_insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行反插入运动")
                else:
                    # if not ensure_coord_sys(robot, INSERT_COORD_SYS, label=f"{INSERT_COORD_SYS}(反插入坐标系)"):
                    #     continue
                    moving = True
                    ok = execute_move(robot, reverse_insert_pose)
                    moving = False

                    try:
                        tcp_after_reverse_insert = robot.get_tcp_pose()
                    except Exception:
                        tcp_after_reverse_insert = None

                    if tcp_after_reverse_insert is not None:
                        logger.info("  反插入后 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_reverse_insert)

                    ensure_coord_sys(robot, BASE_COORD_SYS, label=f"{BASE_COORD_SYS}(恢复基准坐标系)")
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

