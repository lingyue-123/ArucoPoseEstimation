#!/usr/bin/env python3
"""
一次性到位 + 插入运动 | One-Shot Move Insert

检测 ArUco → 计算最终目标位姿 → 一次运动到基准位。
可在任意时刻切到 2 号工具坐标系执行沿局部 z 轴的插入动作。

工作流：
    1. 实时画面 + ArUco 检测
    2. 按 'r'：记录当前 ArUco 位姿为参考，同时保存 0 坐标系下的参考 TCP
    3. 在新的位置按 'm'：基于视觉结果一次运动回基准位（0 坐标系）
    4. 按 'b'：直接回到按下 r 时保存的机械臂位置（0 坐标系）
    5. 按 'i'：实时读取当前 2 号工具坐标系 TCP，沿局部 z 轴前进 --insert-cm
    6. 按 'q' 退出

用法：
    python scripts/oneshot_move_insert.py --camera hikvision_normal --insert-cm 3
    python scripts/oneshot_move_insert.py --camera hikvision_normal --insert-cm 3 --no-robot
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
DEFAULT_SPEED = 4
MOVE_TIMEOUT = 30.0
INSERT_COORD_SYS = 2
BASE_COORD_SYS = 0
COORD_SWITCH_TIMEOUT = 1.0
COORD_SWITCH_SETTLE_SEC = 0.2

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.65
_FONT_THICK = 2


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
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


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


def get_tcp_pose_in_coord(robot, coord, label=None):
    prev_coord = get_coord_sys(robot)
    changed = prev_coord is not None and prev_coord != coord

    if changed and not ensure_coord_sys(robot, coord, label=label):
        return None

    try:
        pose = robot.get_tcp_pose()
    except Exception as exc:
        logger.warning("读取 %s TCP 失败: %s", label or coord, exc)
        pose = None

    if changed and prev_coord is not None:
        ensure_coord_sys(robot, prev_coord, label=f"恢复到 {prev_coord}")
    return pose


def parse_args():
    parser = argparse.ArgumentParser(description='一次性到位 + 工具坐标系插入')
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
    parser.add_argument('--insert-cm', type=float, default=15.0,
                        help='按 i 时沿 2 号工具坐标系 z 轴前进距离（cm）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（IPPE_SQUARE，无 Kalman/SLERP 滤波）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def detect_raw(gray, aruco_detector, valid_ids, marker_sizes, K, dist):
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
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
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
    if trans_mm > max_trans:
        return False, (f"平移 {trans_mm:.1f} mm 超过阈值 {max_trans:.1f} mm，"
                       "请手动移近后重试或增大 --max-trans")
    if rot_deg > max_rot:
        return False, (f"旋转 {rot_deg:.2f} deg 超过阈值 {max_rot:.1f} deg，"
                       "请手动调整姿态后重试或增大 --max-rot")
    return True, ""


def make_insert_pose(tool_ref_pose, insert_mm):
    return offset_pose_along_tool_axis(tool_ref_pose, float(insert_mm), axis='z')


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    insert_mm = args.insert_cm * 10.0

    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()
    robot_cfg = cfg.get_robot(driver=args.robot_driver)

    T_c2g = load_hand_eye_result(args.hand_eye)
    logger.info("手眼标定加载成功: t=[%.2f, %.2f, %.2f]",
                T_c2g[0, 3], T_c2g[1, 3], T_c2g[2, 3])

    T_aruco2cam_ref = None
    if os.path.isfile(args.aruco_ref):
        ref_pose = load_pose_file(args.aruco_ref)[-1]
        T_aruco2cam_ref = pose_to_matrix(ref_pose)
        logger.info("参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                    args.aruco_ref, *ref_pose[:3])
    else:
        logger.info("参考位姿文件不存在，移到参考位置后按 r 键保存")

    tcp_ref_path = args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
    T_g2b_ref = None
    if os.path.isfile(tcp_ref_path):
        tcp_ref_pose = load_pose_file(tcp_ref_path)[-1]
        T_g2b_ref = pose_to_matrix(tcp_ref_pose)
        logger.info("参考 TCP(坐标系0) 已加载: t=[%.2f, %.2f, %.2f] mm", *tcp_ref_pose[:3])


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
            ensure_coord_sys(robot, BASE_COORD_SYS, label=f"{BASE_COORD_SYS}(初始坐标系)")
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

    win = f"OneShotInsert [{args.camera}] ID={target_id}"
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
                result = detect_raw(gray, raw_detector, valid_ids, marker_sizes, K, dist)
            else:
                result = detector.detect(frm)
            with det_lock:
                latest_det = result
                latest_det_frame = frm

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()
    logger.info("就绪。r=设参考  m=视觉回基准位  b=回r点位  i=插入  q=退出  (检测线程已启动)")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            with det_lock:
                aruco_result = latest_det
            target_data = aruco_result.get(target_id)

            tcp = None
            T_g2b_cur = None
            if robot_connected:
                try:
                    tcp = robot.get_tcp_pose()
                except Exception:
                    pass
                if tcp is not None:
                    T_g2b_cur = pose_to_matrix(tcp)

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

            display_trans = trans_err
            display_rot = rot_err
            ref_trans_xyz = None
            if T_g2b_ref is not None and T_g2b_cur is not None:
                ref_trans_xyz, display_trans, display_rot = compute_pose_error_in_frame(T_g2b_cur, T_g2b_ref)

            insert_pose_coord2 = None

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
                    put_text(vis, f"Base Target(0): X={target_pose[0]:.1f} Y={target_pose[1]:.1f} Z={target_pose[2]:.1f}",
                             y, (255, 200, 0))
                    y += 28
                    put_text(vis, f"                Rx={target_pose[3]:.2f} Ry={target_pose[4]:.2f} Rz={target_pose[5]:.2f}",
                             y, (255, 200, 0))
                    y += 30
            elif not ref_set:
                put_text(vis, "Press r at ref position to set reference", y, (0, 165, 255))
                y += 30

            if args.insert_cm <= 0:
                put_text(vis, "Set --insert-cm > 0 to enable insert action", y, (0, 165, 255))
                y += 28
            else:
                put_text(vis, "Press i to move from current coord sys 2 TCP", y, (150, 255, 255))
                y += 28

            if moving:
                put_text(vis, "MOVING...", y, (0, 100, 255))
                y += 30

            robot_str = "Robot:ON" if robot_connected else ("Robot:OFF(dry)" if args.no_robot else "Robot:OFF")
            ref_str = "Ref:SET" if ref_set else "Ref:NONE"
            status = f"ONESHOT+INSERT | {ref_str} | {robot_str} | r=Ref m=Move b=Back i=Insert q=Quit"
            put_text(vis, status, h_orig - 20, (140, 140, 140))

            scale = 0.4
            disp = cv2.resize(vis, (int(w_orig * scale), int(h_orig * scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
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
                    logger.info("参考 TCP(坐标系0) 已保存: t=[%.2f, %.2f, %.2f] mm",
                                T_g2b_cur[0, 3], T_g2b_cur[1, 3], T_g2b_cur[2, 3])


            elif key == ord('m'):
                if moving:
                    logger.warning("运动中，请等待完成")
                    continue
                if T_g2b_target is None:
                    logger.warning("无法计算目标（检查参考位姿/ArUco检测/机械臂连接）")
                    continue

                final_pose = matrix_to_pose(T_g2b_target)
                logger.info("回基准位(坐标系0): 平移=%.2f mm, 旋转=%.2f deg", trans_err, rot_err)
                logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *final_pose)

                safe, reason = check_oneshot_safety(
                    trans_err, rot_err, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行回基准位运动")
                else:
                    if not ensure_coord_sys(robot, BASE_COORD_SYS, label=f"{BASE_COORD_SYS}(基准运动坐标系)"):
                        continue
                    moving = True
                    ok = execute_move(robot, final_pose)
                    moving = False
                    if ok:
                        new_tcp = robot.get_tcp_pose()
                        if new_tcp is not None:
                            T_g2b_after = pose_to_matrix(new_tcp)
                            residual_trans, residual_rot = compute_pose_error(
                                T_g2b_after, T_g2b_target)
                            logger.info("运动后残差(vs target): 平移=%.2f mm, 旋转=%.2f deg",
                                        residual_trans, residual_rot)
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
                    logger.warning("机械臂未连接，无法读取当前 2 号工具坐标系 TCP")
                    continue

                current_tool_pose = get_tcp_pose_in_coord(
                    robot, INSERT_COORD_SYS, label=f"{INSERT_COORD_SYS}(插入坐标系实时TCP)")
                if current_tool_pose is None:
                    logger.warning("无法读取当前 2 号工具坐标系 TCP")
                    continue

                insert_pose = make_insert_pose(current_tool_pose, insert_mm)
                safe, reason = check_oneshot_safety(abs(insert_mm), 0.0, args.max_trans, args.max_rot)
                if not safe:
                    logger.warning(reason)
                    continue

                logger.info("插入动作(坐标系2): 基于当前实时 TCP，沿工具 z 轴前进 %.2f cm (%.2f mm)",
                            args.insert_cm, insert_mm)
                logger.info("  坐标系2当前 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *current_tool_pose)
                logger.info("  坐标系2目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                            *insert_pose)

                if args.no_robot:
                    logger.info("[DRY RUN] 不执行插入运动")
                else:
                    if not ensure_coord_sys(robot, INSERT_COORD_SYS, label=f"{INSERT_COORD_SYS}(插入坐标系)"):
                        continue
                    moving = True
                    ok = execute_move(robot, insert_pose)
                    moving = False

                    try:
                        tcp_after_insert = robot.get_tcp_pose()
                    except Exception:
                        tcp_after_insert = None

                    if tcp_after_insert is not None:
                        logger.info("  插入后坐标系2 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                                    *tcp_after_insert)

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
