#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import json
import fcntl

# 从你提供的接口导入
from keba_control_interface import RobotInterface, CartesianPose, JointPose

from actuator import MotorController 

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoverAction")


class CoverActionFlow:
    def __init__(self):
        # 配置（KEBA 机器人）
        self.ROBOT_IP = "192.168.1.133"
        self.KIN_LIB_PATH = "/home/nvidia/Downloads/HD/HD_0323/KEBA/RobotControlkeba.so"

        # 初始化设备：使用你的 RobotInterface
        self.arm = RobotInterface(kin_lib_path=self.KIN_LIB_PATH)
        self.motor = MotorController(port="/dev/ttysWK3", baudrate=115200, device_id=1)

        self.connect_all()

    def connect_all(self):
        logger.info("=== 开始连接设备 ===")
        self.motor.connect()
        self.arm.init_robot(self.ROBOT_IP, port=502)
        self.arm.arm_PowerON()
        logger.info("=== 设备已就绪 ===")

    # ------------------------------------------------------------------------------------------
    # 简洁版通用相对轨迹方法（已升级：连续运动 + move_joint_extend）
    # ------------------------------------------------------------------------------------------
    def run_relative_trajectory(
        self,
        base_pose: CartesianPose,
        target_poses: list,
        ref_pose: CartesianPose = None,
        motion_type: str = "linear",
        # speed_joint: float = 0.3  # rad/s
    ):
        if ref_pose is None:
            pose_tuple = self.arm.get_tcp_pose()
            if not pose_tuple:
                raise RuntimeError("获取当前位姿失败")
            ref_pose = CartesianPose(*pose_tuple)
            # print('+++++++当前参考位姿：',ref_pose)

        # ref_pose = CartesianPose(x=0.0726, y=-0.7884, z=0.2059, roll=175.7162, pitch=-70.0000, yaw=-80.9862)

        rel_transforms = []
        for pose in target_poses:
            t_rel = self.arm.compute_relative_transform(base_pose, pose)
            rel_transforms.append(t_rel)

        ref_mat = self.arm.pose_to_homogeneous_matrix(ref_pose)

        # 关节轨迹先全部算出来
        joint_path = []
        for idx, t_rel in enumerate(rel_transforms):
            target_mat = ref_mat @ t_rel
            target_pose = self.arm.homogeneous_matrix_to_pose(target_mat)
            print("相对运动笛卡尔位姿： ", target_pose)

            if motion_type == "linear":
                self.arm.move_linear(target_pose)
                logger.info(f"相对轨迹直线运动 → 点 {idx+1}")

            elif motion_type == "joint":
                current_joints = self.arm.arm_get_current_joint()
                current_joints.Q2 -= 90
                current_joints.Q3 = -current_joints.Q3
                current_joints.Q4 -= 90
                joint_target = self.arm.inverse_kinematics(target_pose, current_joints)
                joint_target.Q2 += 90
                joint_target.Q3 = -joint_target.Q3
                joint_target.Q4 += 90
                if joint_target is None:
                    raise RuntimeError(f"逆解失败 → 点 {idx+1}")
                joint_path.append(joint_target)

        # 连续运动发送9个点
        if motion_type == "joint" and len(joint_path) > 0:
            logger.info("执行连续旋盖轨迹")
            for jpos in joint_path:
                self.arm.arm_move_joint(jpos)
                print("+++++++++++++++解算关节角：", jpos)
                pass

            # logger.info("连续轨迹完成")

    # ----------------------------------------------------------------------------------------------
    # 动作 1：开盖
    # ----------------------------------------------------------------------------------------------
    def run_open_cover(self):
        logger.info("\n[动作 1] 开始开盖")

        # with open('/data/yrq/cover_pose.json', 'r', encoding='utf-8') as f:
        #     fcntl.flock(f, fcntl.LOCK_SH)
        #     pose_start = json.load(f)
        #     fcntl.flock(f, fcntl.LOCK_UN)

        joint_init = JointPose(Q1=-42.768742, Q2=-24.339911, Q3=127.591118, Q4=118.949028, Q5=58.601036, Q6=-66.259186)
        self.arm.arm_move_joint(joint_init)
        # self.arm.move_linear(pose_init)
        time.sleep(4)
        self.motor.set_target_position(14000)
        self.arm.move_relative_tool(dz= 88.89, wait=False)
        time.sleep(2)
        self.arm.move_relative_tool(dz= -28.391, wait=True)


        logger.info("[动作 1] 开盖完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 2：旋盖
    # ----------------------------------------------------------------------------------------------
    def run_screw_cover(self):
        logger.info("\n[动作 2] 开始旋盖")

        # 这里是你示教的关节角（角度值）
        # abs_joint_poses = [
        #     # [-70.6520, -37.9961, 118.9954, 136.4139, 76.0198, -86.374],
        #     [-70.6476, -36.9239, 120.6086, 136.9546, 76.0157, -86.3733],
        #     # [-70.4655, -35.3986, 122.8659, 137.6706, 75.8452, -86.3073],
        #     # [-70.9171, -34.0715, 124.7535, 138.2710, 76.2679, -86.4708],
        #     [-71.2957, -32.6480, 126.7302, 138.8560, 76.6225, -86.6074],
        #     # [-72.0798, -30.9200, 129.0291, 139.4900, 77.3571, -86.8891],
        #     [-73.1335, -29.4826, 130.8358, 139.9388, 78.3449, -87.2652],
        #     # [-74.6858, -27.6238, 133.0416, 140.3895, 79.8002, -87.8143],
        #     [-77.4520, -25.7562, 134.9743, 140.6042, 82.3964, -88.7809],
        #     [-79.3113, -24.6272, 136.0286, 140.6048, 84.1426, -89.4239],
        #     [-82.0861, -23.7438, 136.5920, 140.3607, 86.7499, -90.3763],
        #     # [-83.9327, -23.9139, 136.0814, 140.0463, 88.4855, -91.0072],
        #     # [-87.5003, -23.0431, 136.3562, 139.4387, 91.8385, -92.2225],
        #     [-87.5003, -18.0431, 141.3562, 139.4387, 91.8385, -92.2225],


        #     # [-85.9327, -23.7139, 136.2814, 140.0463, 90.0855, -91.7072],
        #     # [-87.642278, -23.363242, 136.015051, 139.424277, 91.972593, -92.272818]
        # ]

        abs_joint_poses = [
            [-71.2957, -32.6480, 126.7302, 138.8560, 76.6225, -86.6074],
            # [-77.4520, -25.7562, 134.9743, 140.6042, 82.3964, -88.7809],
            [-77.134163, -24.053692, 136.978073, 140.885559, 82.097763, -88.669205],
            [-83.932700, -23.913900, 136.081400, 140.046300, 88.485500, -91.007200],
            # [-82.0861, -23.7438, 136.5920, 140.3607, 86.7499, -90.3763],
            # [-87.5003, -18.0431, 141.3562, 139.4387, 91.8385, -92.2225],
            [-86.139267, -17.744228, 141.405731, 138.803818, 90.884682, -91.466713]
        ]

        cartesian_poses = []
        for joint_deg in abs_joint_poses:

            current_joint = JointPose(
                Q1=joint_deg[0],
                Q2=joint_deg[1],
                Q3=joint_deg[2],
                Q4=joint_deg[3],
                Q5=joint_deg[4],
                Q6=joint_deg[5]
            )

            current_joint.Q2 -= 90
            current_joint.Q3 = -current_joint.Q3
            current_joint.Q4 -= 90

            cartesian = self.arm.forward_kinematics(current_joint)

            cartesian_poses.append(cartesian)

            print("逆解结果： ",cartesian_poses)

        # 接近点位
        self.arm.move_relative_tool(dy=-37.848, wait=True)
        self.arm.move_relative_tool(dz=16.531, wait=True)
        self.arm.move_relative_tool(dy= 18.492, wait=True)
        # self.arm.move_relative_tool(dz=-12.19, wait=True)

        base_pose = CartesianPose(x=0.0726, y=-0.7884, z=0.2059, roll=175.7162, pitch=-70.0000, yaw=-80.9862)
        self.run_relative_trajectory(
            base_pose=base_pose,
            target_poses=cartesian_poses,
            motion_type="joint",
        )
        # time.sleep(2)
        # self.arm.move_relative_tool(dz= -30, wait=True)
        # self.arm.move_relative_tool(dy= 30, wait=True)


        # self.arm.move_relative_tool(dz=-50, wait=True)
        # self.motor.homing_motor()
        time.sleep(1)

        logger.info("[动作 2] 旋盖完成 ")


    # ----------------------------------------------------------------------------------------------
    # 动作 3：退盖 + 回收
    # ----------------------------------------------------------------------------------------------
    def run_back_and_reset(self):
        logger.info("\n[动作 3] 退盖 + 回收推杆")

        # 开始时，获取当前位置作为参考位置
        current_pose_tuple = self.arm.get_tcp_pose()
        if current_pose_tuple is None:
            raise RuntimeError("无法获取当前位姿")
        ref_pose = CartesianPose(*current_pose_tuple)

        self.arm.move_relative_tool(dz=-408.928, wait=True)
        self.motor.set_target_position(14000)
        self.arm.move_relative_tool(dy=-346.848, wait=True)
        self.arm.move_relative_tool(dz=209.37, wait=True)

        pose_p1 = CartesianPose(x=189.504, y=563.076, z=246.710, rx=-176.139, ry=67.990, rz=-88.553)
        pose_p2 = CartesianPose(x=81.080, y=528.523, z=259.566, rx=-176.139, ry=67.990, rz=-88.553)
        pose_p3 = CartesianPose(x=-62.406, y=533.868, z=255.941, rx=-176.139, ry=67.990, rz=-88.553)
        pull_poses = [pose_p1, pose_p2, pose_p3]

        base_pose = CartesianPose(x=-148.221, y=760.854, z=163.341, rx=-176.139, ry=67.990, rz=-88.553)
        self.run_relative_trajectory(
            base_pose=base_pose,
            target_poses=pull_poses,
            ref_pose=ref_pose,
            motion_type="linear"
        )

        push_pose = CartesianPose(x=-107.718, y=687.552, z=221.397, rx=166.012, ry=65.132, rz=-103.779)
        self.run_relative_trajectory(
            base_pose=base_pose,
            target_poses=[push_pose],
            ref_pose=ref_pose,
            motion_type="joint",
        )

        self.arm.move_relative_tool(dz=35, wait=True)
        self.arm.move_relative_tool(dz=-200, wait=True)
        self.motor.homing_motor()
        time.sleep(1.5)

        logger.info("[动作 3] 退盖 & 回收完成 ")

    # ----------------------------------------------------------------------------------------------
    # 总入口
    # ----------------------------------------------------------------------------------------------
    def run(self, mode: int):
        try:
            if mode == 1:
                self.run_open_cover()
            elif mode == 2:
                self.run_screw_cover()
            elif mode == 3:
                self.run_back_and_reset()
            elif mode == 4:
                self.run_open_cover()
                self.run_screw_cover()
            else:
                logger.error("模式错误：请输入 1/2/3/4")
        except KeyboardInterrupt:
            logger.warning("手动中断")
        finally:
            self.motor.disconnect()
            self.arm.close()
            logger.info("已断开所有设备")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("   python3 cover_action_main.py 1    # 开盖")
        print("   python3 cover_action_main.py 2    # 旋盖")
        print("   python3 cover_action_main.py 3    # 退盖+回收")
        sys.exit(1)

    mode = int(sys.argv[1])
    flow = CoverActionFlow()
    flow.run(mode)
