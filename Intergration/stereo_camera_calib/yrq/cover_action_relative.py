#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import json
import fcntl


from jaka_driver_interface import JAKARobot ,CartesianPose     # 机械臂接口文件
from actuator import MotorController    # 推杆接口文件

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoverAction")


class CoverActionFlow:
    def __init__(self):
        # 配置
        self.ROBOT_IP = "192.168.1.106"
        self.JAKA_SDK_PATH = "/home/nvidia/Downloads/jaka-python-sdk"

        # 初始化设备（调用外部接口）
        self.arm = JAKARobot(self.ROBOT_IP, self.JAKA_SDK_PATH)
        self.motor = MotorController(port="/dev/ttysWK3", baudrate=115200, device_id=1)

        # 统一连接
        self.connect_all()

    def connect_all(self):
        logger.info("=== 开始连接设备 ===")
        self.motor.connect()
        self.arm.connect()
        self.arm.robot_powered()
        self.arm.robot_enable()
        self.arm.set_cartesian_speed(400)
        self.arm.set_joint_speed(2)
        logger.info("=== 设备已就绪 ===")

    # ----------------------------------------------------------------------------------------------
    # 动作 1：开盖 推杆长度17cm
    # 推杆前推 → 机械臂到初始位 → 法兰 Z 轴向下 70mm
    # ----------------------------------------------------------------------------------------------
    def run_open_cover(self):
        logger.info("\n[动作 1] 开始开盖")

        # 1. 推杆前推
        self.arm.set_cartesian_speed(1000.0)
        # 2. 机械臂回到初始位（关节/直线均可）
        # camera_pose = [88.382, 65.533, -117.357, 207.708, 92.914, -264.911]
        # self.arm.move_joint(camera_pose, 2.5)
        


        ## 视觉检测
        with open('/data/yrq/cover_pose.json', 'r', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_SH) 
                pose_start = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
                
        pose_init = CartesianPose(pose_start[0], pose_start[1], pose_start[2], pose_start[3], pose_start[4], pose_start[5] )
        self.arm.move_linear_block(pose_init)
        # 推杆伸长
        self.motor.set_target_position(14000)
        # time.sleep(1.5)



        # # 3. 法兰 Z 轴向前 
        # self.arm.move_relative_tool(dz= 171.25, mode='linear', wait=True)
        self.arm.move_relative_tool(dz= 105, mode='linear', wait=True) # 103

        # # 4. 法兰 Z 轴向后 
        self.arm.move_relative_tool(dz= -35, mode='linear', wait=True)


        logger.info("[动作 1] 开盖完成 ✅")

    # ----------------------------------------------------------------------------------------------
    # 动作 2：旋盖
    # 法兰 X +30 → Z +2 → X +3 → 圆弧运动
    # ----------------------------------------------------------------------------------------------
    def run_screw_cover(self):
            # ========== 1. 预定义绝对关节角度轨迹（度） ==========
        abs_joint_poses = [
            [91.191, 51.477, -100.181, 204.620, 90.350, -263.764],  # c_pose1
            [89.312, 55.484, -105.868, 206.284, 92.065, -264.531],  # c_pose2
            [87.509, 58.679, -110.191, 207.374, 93.711, -265.269],  # c_pose3
            [85.517, 61.374, -113.737, 208.158, 95.528, -266.087],  # c_pose4
            [83.439, 64.053, -117.065, 208.710, 97.423, -266.946],  # c_pose5
            [80.969, 66.431, -119.958, 209.072, 99.672, -267.977],  # c_pose6
            [78.807, 68.372, -122.198, 209.202, 101.639, -268.891],  # c_pose7
            [74.756, 70.984, -125.159, 209.143, 105.316, -270.640],  # c_pose8
            [71.520, 72.173, -126.561, 208.940, 108.243, -272.081],  # c_pose9
        ]

        # ========== 2. 将绝对关节角度转换为笛卡尔位姿（正运动学） ========== 
        cartesian_poses = []
        for joint_deg in abs_joint_poses:
            ret, pose_tuple = self.arm.kine_forward(joint_deg)  # 返回 (0, (x,y,z,rx,ry,rz)) 角度制
            if ret != 0:
                raise RuntimeError(f"正运动学失败: {pose_tuple}")
            pose = CartesianPose(*pose_tuple)
            cartesian_poses.append(pose)
        print("9组关节角正解位姿: ",cartesian_poses)



        logger.info("\n[动作 2] 开始旋盖")
        self.arm.set_cartesian_speed(800.0)

        # 法兰 y 轴向前 
        self.arm.move_relative_tool(dy= 47.6, mode='linear', wait=True)

        # 法兰 z 轴向前 
        self.arm.move_relative_tool(dz= 25.056, mode='linear', wait=True)

        # 法兰 y 轴向后
        self.arm.move_relative_tool(dy= -18.176, mode='linear', wait=True)

        # 法兰z轴往后 参考点
        self.arm.move_relative_tool(dz= -12.19, mode='linear', wait=True)

        # 相对运动
        # 获取当前实际位姿（作为参考点）
        current_pose_tuple = self.arm.get_tcp_pose()
        if current_pose_tuple is None:
            raise RuntimeError("无法获取当前位姿")
        ref_pose = CartesianPose(*current_pose_tuple)

        # ==========  选择基准位姿（例如第一个点），计算相对变换矩阵列表 ==========
        base_pose = ref_pose
        rel_transforms = []
        for pose in cartesian_poses:
            T_rel = self.arm.compute_relative_transform(base_pose, pose)
            rel_transforms.append(T_rel)

        ref_matrix = self.arm.pose_to_homogeneous_matrix(ref_pose)
        # ========== 5. 执行相对轨迹（每个点用齐次变换计算目标位姿并运动） ==========
        for i, T_rel in enumerate(rel_transforms):
            # 计算目标位姿矩阵：P_target = P_ref @ T_rel
            target_matrix = ref_matrix @ T_rel
            target_pose = self.arm.homogeneous_matrix_to_pose(target_matrix)       
            # 关节运动，先逆解再 move_joint
            current_joint = self.arm.get_joint_pose()
            ret, joint_target = self.arm.kine_inverse(current_joint, target_pose)
            if ret == 0:
                self.arm.move_joint(joint_target, speed_joint=2)
                # print("kine_pose: ",joint_target)

        # # 轨迹点  1
        # c_pose1 = [91.191, 51.477, -100.181, 204.620, 90.350, -263.764]
        # self.arm.move_joint(c_pose1, 1)
        # # 轨迹点 2
        # c_pose2 = [89.312, 55.484, -105.868, 206.284, 92.065, -264.531]
        # self.arm.move_joint(c_pose2, 1)
        # # 轨迹点 3
        # c_pose3 = [87.509, 58.679, -110.191, 207.374, 93.711, -265.269]
        # self.arm.move_joint(c_pose3, 1)
        # # 轨迹点 4
        # c_pose4 = [85.517, 61.374, -113.737, 208.158, 95.528, -266.087]
        # self.arm.move_joint(c_pose4, 1)
        # # 轨迹点 5
        # c_pose5 = [83.439, 64.053, -117.065, 208.710, 97.423, -266.946]
        # self.arm.move_joint(c_pose5, 1)
        # # 轨迹点 6
        # c_pose6 = [80.969, 66.431, -119.958, 209.072, 99.672, -267.977]
        # self.arm.move_joint(c_pose6, 1)
        # # 轨迹点 7
        # c_pose7 =  [78.807, 68.372, -122.198, 209.202, 101.639, -268.891]
        # self.arm.move_joint(c_pose7, 1)
        # # 轨迹点 8
        # c_pose8 =  [74.756, 70.984, -125.159, 209.143, 105.316, -270.640]
        # self.arm.move_joint(c_pose8, 1)
        # # 轨迹点 9
        # c_pose9 =  [71.520, 72.173, -126.561, 208.940, 108.243, -272.081]
        # self.arm.move_joint(c_pose9, 1)

        # 法兰 z 轴向后
        self.arm.move_relative_tool(dz= -50, mode='linear', wait=True)
        # 推杆收回
        self.motor.homing_motor()
        time.sleep(1.5)




        # logger.info("执行圆弧旋盖动作")
        # current = self.arm.get_tcp_pose()
        # if current:
        #     from collections import namedtuple
        #     Pose = namedtuple('Pose', ['x', 'y', 'z', 'rx', 'ry', 'rz'])
        #     x, y, z, rx, ry, rz = current
        #     mid_pose = Pose(x + 10, y, z, rx, ry, rz)
        #     end_pose = Pose(x + 15, y - 8, z, rx, ry, rz)
        #     self.arm.move_circular(mid_pose, end_pose)
        #     self.arm.wait_move_done()

        logger.info("[动作 2] 旋盖完成 ✅")

    # ----------------------------------------------------------------------------------------------
    # 动作 3：退盖 + 回收推杆
    # 法兰 Z 轴后退 30mm → 推杆回零
    # ----------------------------------------------------------------------------------------------
    def run_back_and_reset(self):
        logger.info("\n[动作 3] 退盖 + 回收推杆")
        self.arm.set_cartesian_speed(1000.0)

        # 初始位姿 拔枪点
        pose_takegun = [93.428, 34.308, -79.144, 202.764, 84.303, -269.254]
        self.arm.move_joint(pose_takegun, 2)

        # Z 轴后退 30mm
        self.arm.move_relative_tool(dz= -408.928, mode='linear', wait=True)

        # 推杆伸长
        self.motor.set_target_position(14000)
        time.sleep(1.5)

        # 往 -Y轴方向直线运动
        pose_y =  CartesianPose(x=180.300, y=369.106, z=324.999, rx=-176.139, ry=67.990, rz=-88.553)
        self.arm.move_linear_block(pose_y)

        # # 按压
        self.arm.move_relative_tool(dz= 209.37, mode='linear', wait=True)

        # 拨盖姿态1
        pose_p1 = CartesianPose(x=189.504, y=563.076, z=246.710, rx=-176.139, ry=67.990, rz=-88.553)
        self.arm.move_linear_block(pose_p1)

        # 拨盖姿态2
        pose_p2 = CartesianPose(x=81.080, y=528.523, z=259.566, rx=-176.139, ry=67.990, rz=-88.553)
        self.arm.move_linear_block(pose_p2)

        # 拨盖姿态3
        pose_p3 = CartesianPose(x=-62.406, y=533.868, z=255.941, rx=-176.139, ry=67.990, rz=-88.553)
        self.arm.move_linear_block(pose_p3)

        # 按压对准
        pose_push = [89.069, 48.305, -96.556, 204.148, 92.287, -264.630]
        self.arm.move_joint(pose_push, 2)

        # 按压
        self.arm.move_relative_tool(dz= 35, mode='linear', wait=True)

        # 按压
        self.arm.move_relative_tool(dz= -200, mode='linear', wait=True)


        # 推杆回零回收
        self.motor.homing_motor()
        time.sleep(1.5)

        logger.info("[动作 3] 退盖 & 回收完成 ✅")

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
            self.arm.disconnect()
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
