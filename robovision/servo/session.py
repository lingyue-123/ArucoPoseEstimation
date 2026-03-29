"""
视觉伺服会话 | Visual Servo Session

封装相机、机械臂、检测器、手眼标定的初始化和清理。
各脚本只需关注自己的按键/控制逻辑。
"""

import logging
import os

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.aruco import ArucoDetector
from robovision.robot import build_robot
from robovision.calibration.hand_eye import load_hand_eye_result
from robovision.io.pose_file import load_pose_file, save_pose_file
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose, compute_new_tool_pose,
)
from robovision.visualization.aruco_overlay import draw_aruco_result
from robovision.servo.core import aruco_to_matrix

logger = logging.getLogger(__name__)


class ServoSession:
    """封装相机、机械臂、检测器、手眼标定的初始化和清理。"""

    def __init__(self, args):
        self.args = args
        self.camera = None
        self.robot = None
        self.robot_connected = False
        self.detector = None
        self.intrinsics = None
        self.T_c2g = None
        self.T_aruco2cam_ref = None
        self.T_g2b_ref = None
        self.target_id = getattr(args, 'target_marker', 1)
        self.no_robot = getattr(args, 'no_robot', False)

    def setup(self):
        """初始化所有资源。"""
        args = self.args
        cfg = get_config()
        marker_cfg = cfg.get_marker()
        detection_cfg = cfg.get_detection()
        robot_cfg = cfg.get_robot(driver=args.robot_driver)

        # 手眼标定
        self.T_c2g = load_hand_eye_result(args.hand_eye)
        logger.info("手眼标定加载成功: t=[%.2f, %.2f, %.2f]",
                    self.T_c2g[0, 3], self.T_c2g[1, 3], self.T_c2g[2, 3])

        # 参考 ArUco 位姿
        if os.path.isfile(args.aruco_ref):
            ref_pose = load_pose_file(args.aruco_ref)[-1]
            self.T_aruco2cam_ref = pose_to_matrix(ref_pose)
            logger.info("参考位姿已加载: %s  t=[%.2f, %.2f, %.2f] mm",
                        args.aruco_ref, *ref_pose[:3])
        else:
            logger.info("参考位姿文件不存在，移到参考位置后按 r 键保存")

        # 参考 TCP
        tcp_ref_path = args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
        if os.path.isfile(tcp_ref_path):
            tcp_ref_pose = load_pose_file(tcp_ref_path)[-1]
            self.T_g2b_ref = pose_to_matrix(tcp_ref_pose)
            logger.info("参考 TCP 已加载: t=[%.2f, %.2f, %.2f] mm", *tcp_ref_pose[:3])

        # 机械臂
        if not self.no_robot:
            if args.robot_ip:
                robot_cfg.ip = args.robot_ip
            self.robot = build_robot(robot_cfg)
            self.robot_connected = self.robot.connect()
            if self.robot_connected:
                # 设置速度 2% 和法兰坐标系
                if hasattr(self.robot, 'set_speed'):
                    ret = self.robot.set_speed(2)
                    logger.info("速度设置 → 2%%: %s", "OK" if ret == 0 else f"失败({ret})")
                if hasattr(self.robot, 'set_coord_sys'):
                    ret = self.robot.set_coord_sys(0)
                    logger.info("坐标系设置 → 0(法兰): %s", "OK" if ret == 0 else f"失败({ret})")
            else:
                logger.warning("机械臂连接失败")
        else:
            logger.info("--no-robot 模式：不连接机械臂，仅计算")

        # 相机与检测器
        self.camera = build_camera(args.camera, cfg)
        self.camera.open()
        self.intrinsics = self.camera.get_intrinsics()
        self.detector = ArucoDetector.from_config(self.intrinsics, marker_cfg, detection_cfg)

    def read_and_detect(self):
        """
        读取一帧并检测 ArUco。

        Returns:
            (frame, aruco_result, target_data, tcp, T_g2b_cur)
            frame 为 None 时表示读帧失败
        """
        ok, frame = self.camera.read_frame()
        if not ok or frame is None:
            return None, {}, None, None, None

        aruco_result = self.detector.detect(frame)
        target_data = aruco_result.get(self.target_id)

        tcp = None
        T_g2b_cur = None
        if self.robot_connected:
            tcp = self.robot.get_tcp_pose()
            if tcp is not None:
                T_g2b_cur = pose_to_matrix(tcp)

        return frame, aruco_result, target_data, tcp, T_g2b_cur

    def draw_aruco(self, frame, aruco_result, tcp):
        """绘制 ArUco 可视化叠加层。"""
        return draw_aruco_result(
            frame, aruco_result, self.intrinsics,
            robot_pose=tcp,
            use_kalman=False,
            robot_connected=self.robot_connected,
        )

    def set_reference(self, target_data, T_g2b_cur=None):
        """记录当前 ArUco 位姿为参考，同时保存 TCP 参考。"""
        if target_data is None:
            logger.warning("未检测到 ID%d，无法保存参考", self.target_id)
            return False
        self.T_aruco2cam_ref = aruco_to_matrix(target_data)
        ref_pose_vec = matrix_to_pose(self.T_aruco2cam_ref)
        os.makedirs(os.path.dirname(self.args.aruco_ref) or '.', exist_ok=True)
        save_pose_file(self.args.aruco_ref, np.array([ref_pose_vec]))
        logger.info("参考位姿已保存: t=[%.2f, %.2f, %.2f] mm", *ref_pose_vec[:3])

        if T_g2b_cur is not None:
            self.T_g2b_ref = T_g2b_cur
            tcp_ref_path = self.args.aruco_ref.replace('aruco_pose_ref', 'tcp_ref')
            tcp_pose_vec = matrix_to_pose(T_g2b_cur)
            save_pose_file(tcp_ref_path, np.array([tcp_pose_vec]))
            logger.info("参考 TCP 已保存: t=[%.2f, %.2f, %.2f] mm", *tcp_pose_vec[:3])

        return True

    def compute_ref_error(self, T_g2b_cur):
        """
        计算当前 TCP 与参考 TCP 的误差，分解到参考坐标系。

        Returns:
            (trans_xyz, trans_norm, rot_err) 或 (None, None, None)
            trans_xyz: ndarray (3,) — 参考坐标系下 [dx, dy, dz] (mm)
        """
        if self.T_g2b_ref is None or T_g2b_cur is None:
            return None, None, None
        from robovision.servo.core import compute_pose_error_in_frame
        return compute_pose_error_in_frame(T_g2b_cur, self.T_g2b_ref)

    def compute_target(self, target_data, tcp, T_g2b_cur=None):
        """
        由当前 ArUco 检测和 TCP 计算目标位姿。

        Returns:
            (T_g2b_target, trans_err, rot_err) 或 (None, None, None)
        """
        if self.T_aruco2cam_ref is None or target_data is None:
            return None, None, None
        if tcp is None or T_g2b_cur is None:
            return None, None, None

        from robovision.servo.core import compute_pose_error
        T_aruco2cam_cur = aruco_to_matrix(target_data)
        T_g2b_target = compute_new_tool_pose(
            base_from_tool_old=T_g2b_cur,
            cam_from_target_old=T_aruco2cam_cur,
            tool_from_cam=self.T_c2g,
            cam_from_target_new=self.T_aruco2cam_ref,
        )
        trans_err, rot_err = compute_pose_error(T_g2b_cur, T_g2b_target)
        return T_g2b_target, trans_err, rot_err

    def close(self):
        """清理资源。"""
        cv2.destroyAllWindows()
        if self.camera is not None:
            self.camera.close()
        if self.robot is not None:
            self.robot.disconnect()

    @property
    def ref_set(self):
        return self.T_aruco2cam_ref is not None

    @property
    def robot_status_str(self):
        if self.robot_connected:
            return "Robot:ON"
        elif self.no_robot:
            return "Robot:OFF(dry)"
        return "Robot:OFF"
