import logging
import os
import time
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.crobot_driver_interface import (
    CartesianPose,
    pose_to_homogeneous_matrix,
    homogeneous_matrix_to_pose,
)
from third_party.crp_robot_sdk.crp_robot import Robot
from third_party.crp_robot_sdk.crp_robot._types import JointPosition, RobotPosition, MotionParam, DHParam
from third_party.crp_robot_sdk.crp_robot._enums import MotionType, MoveStrategy, ProgramStatus, RobotMode

from contextlib import contextmanager
logger = logging.getLogger("BridgeCRobotAdapter")

_POLL_INTERVAL_S = 0.05
_INSTRUCTION_READY_TIMEOUT_S = 15.0
_MOTION_COMPLETE_TIMEOUT_S = 60.0


class BridgeCRobotAdapter:
    def __init__(self, ip: str, so_path: str):
        self.ip = ip
        self.so_path = so_path
        self.bridge_robot = Robot(so_path)
        self._connected = False

        self.a_list = [0, 621.620, 559.133, 0, 0, 0]
        self.d_list = [0, 0, 0, -164.261, 119.327, 115.0]
        self.alpha_deg_list = [90, 0, 0, 90, 90, 0]
        self.offset_deg_list = [0, 0, -90, 90, -90, 0]
        
    @contextmanager
    def instruction_session(self):
        self._ensure_ready_for_motion()
        self._ensure_instruction_program()
        try:
            yield
        finally:
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            while(self.bridge_robot.is_moving()):
                logger.info("Waiting for robot to stop moving...")
                time.sleep(_POLL_INTERVAL_S)
    
    @contextmanager
    def path_session(self):
        self._ensure_ready_for_motion()
        self._ensure_path()
        try:
            yield
        finally:
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            while(self.bridge_robot.is_moving()):
                logger.info("Waiting for robot to stop moving...")
                time.sleep(_POLL_INTERVAL_S)
        # return self._InstructionSession(self)

    def connect(self) -> bool:
        ok = self.bridge_robot.connect(self.ip, disable_hardware=True)
        self._connected = bool(ok)
        return self._connected

    def disconnect(self) -> None:
        if self._connected:
            try:
                self.bridge_robot.disconnect()
            finally:
                self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self.bridge_robot.is_connected()

    def _ensure_ready_for_motion(self):
        self.bridge_robot.set_work_mode(RobotMode.Playing)
        if self.bridge_robot.has_error():
            if self.bridge_robot.has_emergency_error():
                errors = []
                try:
                    errors = self.bridge_robot.get_errors()
                except Exception as exc:
                    logger.error("get_errors failed while handling emergency error: %s", exc)
                if errors:
                    raise RuntimeError(f"robot has emergency error: {errors}")
                raise RuntimeError("robot has emergency error")
            self.bridge_robot.clear_error()
        if not self.bridge_robot.is_servo_on():
            if not self.bridge_robot.servo_on():
                raise RuntimeError("servo on failed")

    def _ensure_instruction_program(self):
        status = self.bridge_robot.get_program_status()
        if status == ProgramStatus.Stop:
            if not self.bridge_robot.start_program("guidanceInst.pro", 0):
                raise RuntimeError("start guidanceInst.pro failed")
        elif status == ProgramStatus.Pause:
            if not self.bridge_robot.resume_program("guidanceInst.pro"):
                raise RuntimeError("resume guidanceInst.pro failed")
        start = time.time()
        while not self.bridge_robot.motion.is_ready(MotionType.Instruction):
            if time.time() - start > _INSTRUCTION_READY_TIMEOUT_S:
                raise TimeoutError("waiting for instruction motion ready timed out")
            time.sleep(_POLL_INTERVAL_S)

    def _wait_for_motion_complete(self, timeout: float = _MOTION_COMPLETE_TIMEOUT_S) -> None:
        start = time.time()
        while self.is_moving():
            if time.time() - start > timeout:
                raise TimeoutError("waiting for robot motion complete timed out")
            time.sleep(_POLL_INTERVAL_S)

    def set_speed(self, speed_pct: int) -> bool:
        return self.bridge_robot.set_speed_ratio(speed_pct)

    def get_tcp_pose(self) -> Optional[List[float]]:
        try:
            pos = self.bridge_robot.get_position()
            return [pos.x, pos.y, pos.z, pos.Rx, pos.Ry, pos.Rz]
        except Exception as exc:
            logger.error("get_tcp_pose failed: %s", exc)
            return None

    def get_joint_pose(self) -> Optional[List[float]]:
        try:
            joints = self.bridge_robot.get_joint()
            return [round(v, 3) for v in joints.body]
        except Exception as exc:
            logger.warning("get_joint_pose first attempt failed: %s", exc)

        try:
            self._ensure_ready_for_motion()
            time.sleep(_POLL_INTERVAL_S)
            joints = self.bridge_robot.get_joint()
            return [round(v, 3) for v in joints.body]
        except Exception as exc:
            logger.error("get_joint_pose failed after retry: %s", exc)
            return None

    def is_moving(self) -> bool:
        try:
            return self.bridge_robot.is_moving()
        except Exception as exc:
            logger.error("is_moving failed: %s", exc)
            return False

    def move_linear(self, target: CartesianPose, speed: int = 100, start: bool = True, end: bool = True) -> int:
        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_instruction_program()
            rp = RobotPosition(x=target.x, y=target.y, z=target.z, Rx=target.rx, Ry=target.ry, Rz=target.rz)
            ok = self.bridge_robot.motion.move_l(
                0,
                rp,
                MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1),
                MoveStrategy.DistanceFirst,
            )
            if not ok:
                return 0
            if end:
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_linear failed: %s", exc)
            return 0

    def move_joint(self, joint: List[float], speed: int = 100, start: bool = True, end: bool = True) -> int:
        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_instruction_program()
            jp = JointPosition(body=list(joint))
            ok = self.bridge_robot.motion.move_abs_j(
                0,
                jp,
                MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1),
            )
            if not ok:
                return 0
            if end:
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_joint failed: %s", exc)
            return 0

    def move_by_pose_list(self, poses: List[List[float]], speeds: List[int], end: bool = True) -> int:
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            for idx, pose in enumerate(poses):
                rp = RobotPosition(x=pose[0], y=pose[1], z=pose[2], Rx=pose[3], Ry=pose[4], Rz=pose[5])
                ok = self.bridge_robot.motion.move_l(
                    idx,
                    rp,
                    MotionParam(speed=float(speeds[idx]), pl=0.0, smooth=0, acc=1, dec=1),
                    MoveStrategy.DistanceFirst,
                )
                if not ok:
                    return 0
            if end:
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_by_pose_list failed: %s", exc)
            return 0

    def move_by_joint_list(self, joints: List[List[float]], speeds: List[int]) -> int:
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            for idx, joint in enumerate(joints):
                ok = self.bridge_robot.motion.move_abs_j(
                    idx,
                    JointPosition(body=list(joint)),
                    MotionParam(speed=float(speeds[idx]), pl=0.0, smooth=0, acc=1, dec=1),
                )
                if not ok:
                    return 0
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_by_joint_list failed: %s", exc)
            return 0

    def move_and_wait(self, target, timeout: float = 30) -> bool:
        if self.move_linear(target, start=True, end=True) != 1:
            return False
        start = time.time()
        while self.is_moving():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def move_joint_and_wait(self, target, speed, timeout: float = 30) -> bool:
        ik_ref_joint = self.get_joint_pose()
        joint_pose = self.inverse_kinematics(target.to_list(), ik_ref_joint)
        if self.move_by_joint_list(joints=[joint_pose], speeds=[speed]) != 1:
            return False
        start = time.time()
        while self.is_moving():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def relative_tool_pose(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                           drx: float = 0.0, dry: float = 0.0, drz: float = 0.0, init_pose: list = None,
                           mode: str = 'linear', speed: int = 100, wait: bool = True):
        if init_pose is None:
            return -1
        current_carpose = CartesianPose(*init_pose[:6])
        T_base_tcp = pose_to_homogeneous_matrix(current_carpose, degrees=True)
        delta_pose = CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
        T_tool_delta = pose_to_homogeneous_matrix(delta_pose, degrees=True)
        T_target_base = np.dot(T_base_tcp, T_tool_delta)
        return homogeneous_matrix_to_pose(T_target_base, degrees=True)

    def dh_transform(self, a: float, alpha_rad: float, d: float, theta_rad: float) -> np.ndarray:
        sa = np.sin(alpha_rad)
        ca = np.cos(alpha_rad)
        st = np.sin(theta_rad)
        ct = np.cos(theta_rad)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1],
        ])

    def forward_kinematics(self, joints: List[float], representation: str = 'euler'):
        T = np.eye(4)
        for i in range(len(joints)):
            T = T @ self.dh_transform(
                self.a_list[i],
                np.deg2rad(self.alpha_deg_list[i]),
                self.d_list[i],
                np.deg2rad(joints[i]) + np.deg2rad(self.offset_deg_list[i]),
            )
        if representation == 'matrix':
            return T
        pos = T[:3, 3]
        rot = T[:3, :3]
        rpy = R.from_matrix(rot).as_euler('ZYX', degrees=True)
        return [pos[0], pos[1], pos[2], rpy[2], rpy[1], rpy[0]]

    def inverse_kinematics(self, target_pose: List[float], initial_joints: Optional[List[float]] = None,
                           representation: str = 'euler', max_iter: int = 200, tol: float = 1e-6) -> List[float]:
        joint_count = len(self.a_list)
        if initial_joints is None:
            joints = np.zeros(joint_count)
        else:
            joints = np.array(initial_joints, dtype=float)

        target_pos = np.array(target_pose[:3], dtype=float)
        if representation == 'euler':
            rx, ry, rz = target_pose[3], target_pose[4], target_pose[5]
            target_rot = R.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
        elif representation == 'rotvec':
            target_rot = R.from_rotvec(target_pose[3:6], degrees=True).as_matrix()
        else:
            raise ValueError(f'Unsupported representation: {representation}')

        for _ in range(max_iter):
            transform = np.eye(4)
            transform_list = []
            for i in range(joint_count):
                transform = transform @ self.dh_transform(
                    self.a_list[i],
                    np.deg2rad(self.alpha_deg_list[i]),
                    self.d_list[i],
                    np.deg2rad(joints[i]) + np.deg2rad(self.offset_deg_list[i]),
                )
                transform_list.append(transform.copy())

            current_pos = transform[:3, 3]
            current_rot = transform[:3, :3]

            err_pos = target_pos - current_pos
            rot_diff = target_rot @ current_rot.T
            err_rot = R.from_matrix(rot_diff).as_rotvec()
            error = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(error) < tol:
                break

            jacobian = np.zeros((6, joint_count))
            p_end = current_pos
            for i in range(joint_count):
                if i == 0:
                    z_axis = np.array([0.0, 0.0, 1.0])
                    p_axis = np.array([0.0, 0.0, 0.0])
                else:
                    z_axis = transform_list[i - 1][:3, 2]
                    p_axis = transform_list[i - 1][:3, 3]
                jacobian[:3, i] = np.cross(z_axis, p_end - p_axis)
                jacobian[3:, i] = z_axis

            lamda = 0.1
            identity = np.eye(6)
            delta_theta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lamda * identity, error)
            joints = joints + np.rad2deg(delta_theta)

        return joints.tolist()
