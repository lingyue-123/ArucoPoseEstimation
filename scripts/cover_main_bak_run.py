#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import json
import fcntl
import numpy as np

from crobot_driver_interface import CRobot, CartesianPose, pose_to_homogeneous_matrix, homogeneous_matrix_to_pose, get_flange_relative_move
from gripper_controller import GripperController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoverAction")

# with open("/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json", "r", encoding="utf-8") as f:
#     CFG = json.load(f)

class CoverActionFlow:
    _instance = None

    @classmethod
    def get_instance(cls, ip=None):
        if cls._instance is None:
            cls._instance = cls(ip=ip)
        return cls._instance

    def __init__(self, ip=None):
        self.arm = CRobot(ip=ip or '192.168.1.12')
        # self.arm = bridge_robot
        self._connected = False
        self.gripper = GripperController(port='/dev/ttysWK3', baudrate=115200, slave_id=4)
        with open("/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json", "r", encoding="utf-8") as f:
            self.CFG = json.load(f)
    def connect(self):
        if self._connected:
            logger.info("Robot already connected, skipping")
            return True
        logger.info("=== 开始连接设备 ===")
        if not self.arm.connect():
            logger.error("机器人连接失败！")
            return False
        self.gripper.connect()
        self._connected = True
        logger.info("=== 设备已就绪 ===")
        self.arm.set_speed(30)
        logger.info("=== 速度设置30% ===")
        return True

    def disconnect(self):
        if self._connected:
            self.arm.disconnect()
            self.gripper.disconnect()
            self._connected = False
            logger.info("Robot disconnected")
    def update_CFG(self):
        with open("/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json", "r", encoding="utf-8") as f:
            self.CFG = json.load(f)

    @staticmethod
    def _require_motion_ok(ret, action_name):
        if ret != 1:
            raise RuntimeError(f"{action_name} failed")

    def run_relative_trajectory(
        self,
        base_pose: CartesianPose,
        target_poses: list,
        ref_pose: CartesianPose = None,
        motion_type: str = "linear"
    ):
        if ref_pose is None:
            ref_pose_list = self.arm.get_tcp_pose()
            ref_pose = CartesianPose(*ref_pose_list)
            if ref_pose is None:
                raise RuntimeError("获取当前位姿失败")

        ref_mat = pose_to_homogeneous_matrix(ref_pose, degrees=True)
        joint_path = []

        for idx, target_pose in enumerate(target_poses):
            base_mat = pose_to_homogeneous_matrix(base_pose, degrees=True)
            target_mat = pose_to_homogeneous_matrix(target_pose, degrees=True)
            t_rel = np.linalg.inv(base_mat) @ target_mat
            new_target_mat = ref_mat @ t_rel
            new_target_pose = homogeneous_matrix_to_pose(new_target_mat, degrees=True)
            
            if motion_type == "linear":
                self.arm.move_linear(new_target_pose)
                print("线性运动:",new_target_pose)
                logger.info(f"相对轨迹直线运动 → 点 {idx+1}")

            elif motion_type == "joint":
                joint_ik = self.CFG["JOINT_IK_DEFAULT"]
                target_pose_l = [new_target_pose.x, new_target_pose.y, new_target_pose.z, new_target_pose.rx, new_target_pose.ry, new_target_pose.rz] 
                joint_target = self.arm.inverse_kinematics(target_pose_l, joint_ik, representation='euler')
                if joint_target is None:
                    raise RuntimeError(f"逆解失败 → 点 {idx+1}")
                joint_path.append(joint_target)

        if motion_type == "joint" and len(joint_path) > 0:
            logger.info("执行连续旋盖轨迹")
            for jpos in joint_path:
                self.arm.move_joint(jpos, speed=20)
                logger.info(f"相对轨迹关节运动 → 点 {idx}")

        logger.info("相对轨迹执行完成")

    def relative_pose(
        self,
        base_pose: CartesianPose,
        target_poses: list,
        ref_pose: CartesianPose = None,
        motion_type: str = "linear"
    ):
        if ref_pose is None:
            ref_pose_list = self.arm.get_tcp_pose()
            ref_pose = CartesianPose(*ref_pose_list)
            if ref_pose is None:
                raise RuntimeError("获取当前位姿失败")

        ref_mat = pose_to_homogeneous_matrix(ref_pose, degrees=True)
        joint_path = []
        pose_path = []

        for idx, target_pose in enumerate(target_poses):
            base_mat = pose_to_homogeneous_matrix(base_pose, degrees=True)
            target_mat = pose_to_homogeneous_matrix(target_pose, degrees=True)
            t_rel = np.linalg.inv(base_mat) @ target_mat
            new_target_mat = ref_mat @ t_rel
            new_target_pose = homogeneous_matrix_to_pose(new_target_mat, degrees=True)
            
            if motion_type == "linear":
                # print("线性运动:",new_target_pose)
                new_target_pose = [new_target_pose.x, new_target_pose.y, new_target_pose.z, new_target_pose.rx, new_target_pose.ry, new_target_pose.rz]
                pose_path.append(new_target_pose)

            elif motion_type == "joint":
                joint_ik = self.CFG["JOINT_IK_DEFAULT"]
                target_pose_l = [new_target_pose.x, new_target_pose.y, new_target_pose.z, new_target_pose.rx, new_target_pose.ry, new_target_pose.rz] 
                joint_target = self.arm.inverse_kinematics(target_pose_l, joint_ik, representation='euler')
                if joint_target is None:
                    raise RuntimeError(f"逆解失败 → 点 {idx+1}")
                joint_path.append(joint_target)
        if motion_type == 'linear':
            return pose_path
        elif motion_type == 'joint':
            return joint_path
        logger.info("相对轨迹执行完成")

    # ----------------------------------------------------------------------------------------------
    # 动作 1：开盖
    # ----------------------------------------------------------------------------------------------
    def run_open_cover(self):
        logger.info("\n[动作 1] 开始开盖")
        self.gripper.set_speed(100) 
        self.gripper.set_position(45)

        pose0 = CartesianPose(*self.CFG["POINT_NEW_REF"]).to_list()
        # pose0 = self.arm.get_tcp_pose()
        pose1 = self.arm.relative_tool_pose(dz=25,init_pose=pose0).to_list()
        pose2 = self.arm.relative_tool_pose(dz=29, init_pose=pose0).to_list()
        pose3 = self.arm.relative_tool_pose(dz=0, init_pose=pose0).to_list()
        poses = [pose0, pose1, pose2, pose3]
        joint_poses = []
        init_joint = self.arm.get_joint_pose()
        if init_joint is None:
            init_joint = self.CFG["JOINT_IK_DEFAULT"]
            logger.warning("获取当前关节失败，开盖动作回退到 JOINT_IK_DEFAULT 种子")
        for pose in poses:
            pose = self.arm.inverse_kinematics(pose, init_joint)
            joint_poses.append(pose)
        speeds = [30,15,5,30]
        ret = self.arm.move_by_joint_list(joints=joint_poses, speeds=speeds)
        self._require_motion_ok(ret, "run_open_cover move_by_joint_list")

        logger.info("[动作 1] 开盖完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 2：旋盖
    # ----------------------------------------------------------------------------------------------
    def run_screw_cover(self):
        logger.info("\n[动作 2] 开始旋盖")
        # current_pose = self.arm.get_tcp_pose()
        # pose_dz = [self.arm.relative_tool_pose(dz = -10, init_pose=current_pose).to_list()]
        # speeds = [100]
        # self.arm.move_by_pose_list(poses=pose_dz, speeds=speeds)

        # target_pose = self.arm.get_tcp_pose()
        # with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
        #                 data = json.load(f)
        #                 data["POINT_NEW_REF"] = target_pose

        # with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        # self.update_CFG()
        
        ref_pose = CartesianPose(*self.CFG["POINT_NEW_REF"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_REF"])

        # ref_pose = CartesianPose(*self.CFG["POINT_NEW_REF"]) # 拨盖起点，获取当前位姿为参考点
        pose1 = CartesianPose(*self.CFG["SCREW_P1"])
        pose2 = CartesianPose(*self.CFG["SCREW_P2"])
        pose3 = CartesianPose(*self.CFG["SCREW_P3"])
        joint1 = CartesianPose(*self.CFG["SCREW_J1"])
        joint2 = CartesianPose(*self.CFG["SCREW_J2"])
        joint3 = CartesianPose(*self.CFG["SCREW_J3"])

        cartesian_poses = [pose1, pose2, pose3,joint1,joint2,joint3]
        joint_pose = self.relative_pose(
            base_pose=base_pose,
            target_poses=cartesian_poses,
            ref_pose = ref_pose,
            motion_type="joint"
        )
        speeds = [25,25,25,25,25,25]
        ret = self.arm.move_by_joint_list(joint_pose,speeds)
        self._require_motion_ok(ret, "run_screw_cover screw trajectory")

        # 取小盖前的调整位姿
        insert_adjust_base_pose = CartesianPose(*self.CFG["INSERT_ADJUST_POINT_TEMPLATE_REF"])
        insert_adjust_poses = [CartesianPose(*self.CFG["INSERT_ADJUST_POINT"])]
        adjust_joint_pose = self.relative_pose(
            base_pose=insert_adjust_base_pose,
            target_poses= insert_adjust_poses,
            ref_pose = ref_pose,
            motion_type="joint"
        )
        speeds = [30]
        ret = self.arm.move_by_joint_list(adjust_joint_pose,speeds)
        self._require_motion_ok(ret, "run_screw_cover insert adjust")

        self.gripper.set_position(25)

        logger.info("[动作 2] 旋盖完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 3：夹盖
    # ----------------------------------------------------------------------------------------------
    def gripper_action(self):
        self.gripper.set_speed(50) 

        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_COVER_REF"])


        # self.gripper.set_position(22)

        # init_poses = self.relative_pose(
        #     base_pose=base_pose,
        #     target_poses=[CartesianPose(*self.CFG["PUSH_POINT1"]),
        #                    CartesianPose(*self.CFG["PUSH_POINT2"])],
        #     ref_pose=ref_pose,
        #     motion_type="joint"
        # )
        # init_speed = [25,5]
        # self.arm.move_by_joint_list(init_poses, init_speed)

        self.gripper.set_force(50)
        self.gripper.set_position(45)
        timeout_s = 5.0
        t0 = time.time()
        while True:
            status = self.gripper.get_grip_status()
            if status in (1, 2):
                logger.info("Gripper open complete (status=%d)", status)
                break
            if time.time() - t0 > timeout_s:
                logger.warning("Gripper open timeout after %.1fs (status=%s)", timeout_s, status)
                break
            time.sleep(0.1)

        all_targets = [
            # 夹住后撤点1
            CartesianPose(*self.CFG["PUSH_POINT3"]), # 移动到放盖点前
            CartesianPose(*self.CFG["PUSH_POINT4"]), # 放盖的点
            CartesianPose(*self.CFG["PUSH_POINT4_dz"])
        ]

        poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="joint"
        )
        speeds = [25,25,25]
        self.arm.move_by_joint_list(poses, speeds)
        # self.arm.move_relative_tool(dz=3) # 放盖经常走不到底，这里找补一下，继续前进3mm

        self.gripper.set_position(22)
        t1 = time.time()
        while True:
            status = self.gripper.get_grip_status()
            if status in (1, 2):
                logger.info("Gripper open complete (status=%d)", status)
                break
            if time.time() - t1 > timeout_s:
                logger.warning("Gripper open timeout after %.1fs (status=%s)", timeout_s, status)
                break
            time.sleep(0.1)

        current_tcp_pose = self.arm.get_tcp_pose()
        pose_dz = self.arm.relative_tool_pose(dz = -30, init_pose=current_tcp_pose).to_list()
        take_gun_poses = self.arm.inverse_kinematics(target_pose=pose_dz, initial_joints=self.arm.get_joint_pose())
        
        # innercover_base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_INNERCOVER_REF"])
        # take_gun_poses = self.relative_pose(
        #     base_pose=innercover_base_pose,
        #     target_poses=[CartesianPose(*self.CFG["PUSH_POINT5"])],
        #     ref_pose=ref_pose,
        #     motion_type="joint"
        # )
        speeds = [25]
        self.arm.move_by_joint_list([take_gun_poses], speeds)

        self.gripper.set_position(0)


        # 移动到取枪对准点
        # TODO:直线运动 -> 关节运动
        target_pose  = self.CFG["PUSH_POINT6_TAKEGUN"]
        self.arm.move_by_joint_list([target_pose],speeds=[25])      

    # ----------------------------------------------------------------------------------------------
    # 动作 4：移动到插枪对准点
    # ----------------------------------------------------------------------------------------------
    def gun_insert_before(self):
        logger.info("\n[动作 4] 移动到插枪对准点")

        self.update_CFG()
        target_poses = self.CFG["INSERT_BEFORE_TEMPLATE"]
        ik_joint = self.CFG["JOINT_IK_DEFAULT"]
        joint_pose = [self.arm.inverse_kinematics(target_pose=target_poses, initial_joints=ik_joint)]
        speeds = [25]
        self.arm.move_by_joint_list(joint_pose, speeds)
        print("插枪对准点关节运动完成")
        # self.arm.move_by_pose_list(target_poses, speeds)


        pose0 = self.arm.get_tcp_pose()
        current_joint = self.arm.get_joint_pose()
        pose1 = self.arm.relative_tool_pose(dz=100,init_pose=pose0).to_list()
        target_joint = [self.arm.inverse_kinematics(target_pose=pose1, initial_joints=current_joint)]
        speeds = [10]
        self.arm.move_by_joint_list(target_joint, speeds)

        logger.info("[动作 4] 移动到插枪对准点完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 5：归枪
    # ----------------------------------------------------------------------------------------------
    def gun_home(self):
        logger.info("\n[动作 5] 归枪")

        # cur_pose = self.arm.get_tcp_pose()
        # pose = self.arm.relative_tool_pose(dz = -150, init_pose=cur_pose)
        # self.arm.move_linear(pose)

        ref_pose = CartesianPose(*self.CFG["POINT_NEW_REF"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_REF"])

        pose0 = CartesianPose(*self.CFG["GUN_HOME1"]).to_list()
        pose1 = CartesianPose(*self.CFG["GUN_HOME2"]).to_list()
        gun_init_poses = [pose0, pose1]
        gun_speeds = [25, 5]
        self.arm.move_by_joint_list(joints=gun_init_poses, speeds=gun_speeds)

        # 微调
        
       
        gripper = self.gripper
        gripper.set_speed(100)
        gripper.set_position(0)
        # 等待夹爪张开到位（0=运动中, 1=到达位置, 2=夹住物体, 3=物体掉落）
        timeout_s = 5.0
        t0 = time.time()
        while True:
            status = gripper.get_grip_status()
            if status in (1, 2):
                logger.info("Gripper open complete (status=%d)", status)
                break
            if time.time() - t0 > timeout_s:
                logger.warning("Gripper open timeout after %.1fs (status=%s)", timeout_s, status)
                break
            time.sleep(0.1)

        pose_j = self.arm.get_tcp_pose()
        pose3 = self.arm.relative_tool_pose(dz = -150, init_pose=pose_j).to_list()
        poses = [pose3]
        speeds = [100]
        self.arm.move_by_pose_list(poses=poses, speeds=speeds)

        gripper.set_speed(100) 
        gripper.set_position(22)

        logger.info("[动作 5] 归枪")

    # ----------------------------------------------------------------------------------------------
    # 动作 6：关内盖
    # ----------------------------------------------------------------------------------------------
    def inner_cover_close_step1(self):
        gripper = self.gripper
        gripper.set_speed(50) 
        gripper.set_position(22)
        
        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_INNERCOVER_REF"])

        all_targets = [
            CartesianPose(*self.CFG["PUSH_POINT5"])
        ]

        init_poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="joint"
        )
        init_speeds = [30]
        self.arm.move_by_joint_list(joints=init_poses, speeds=init_speeds)
    
    def inner_cover_close_step2(self):
        gripper = self.gripper
        gripper.set_force(50)
        gripper.set_position(45)
        timeout_s = 5.0
        t0 = time.time()
        while True:
            status = gripper.get_grip_status()
            if status in (1, 2):
                logger.info("Gripper open complete (status=%d)", status)
                break
            if time.time() - t0 > timeout_s:
                logger.warning("Gripper open timeout after %.1fs (status=%s)", timeout_s, status)
                break
            time.sleep(0.1)

        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_COVER_REF"])

        current_pose = self.arm.get_tcp_pose()
        init_pose = self.arm.relative_tool_pose(dz = -30, init_pose = current_pose).to_list()
        all_targets = [
            CartesianPose(*self.CFG["PUSH_POINT3"]),
        ]
        poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="linear"
        )
        target_poses = [init_pose, poses[0], self.CFG["PUSH_POINT1"], self.CFG["PUSH_POINT2"]]
        poses = []
        for pose in target_poses:
            pose = self.arm.inverse_kinematics(target_pose=pose, initial_joints=self.CFG["JOINT_IK_DEFAULT"])
            poses.append(pose)
        speeds = [30, 30, 30, 25]
        self.arm.move_by_joint_list(joints=poses, speeds=speeds)

        gripper.set_position(22)
        t1 = time.time()
        while True:
            status = gripper.get_grip_status()
            if status in (1, 2):
                logger.info("Gripper open complete (status=%d)", status)
                break
            if time.time() - t1 > timeout_s:
                logger.warning("Gripper open timeout after %.1fs (status=%s)", timeout_s, status)
                break
            time.sleep(0.1)
        
        target_l = [CartesianPose(*self.CFG["PUSH_POINT6"])]
        pose_l = self.relative_pose(
            base_pose=base_pose,
            target_poses=target_l,
            ref_pose=ref_pose,
            motion_type="joint"
        )
        speeds = [30]
        self.arm.move_by_joint_list(joints=pose_l, speeds=speeds)

        gripper.set_position(45)

    # ----------------------------------------------------------------------------------------------
    # 动作 7：关外盖
    # ----------------------------------------------------------------------------------------------
    def outer_cover_close(self):

        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_COVER_REF"])

        all_targets = [
            CartesianPose(*self.CFG["OUTER_CLOSE_1"]),
            CartesianPose(*self.CFG["OUTER_CLOSE"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_0"])
        ]

        init_poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="joint"
        )
        speeds = [30, 30, 10]
        self.arm.move_by_joint_list(joints=init_poses, speeds=speeds)

        current_pose = self.arm.get_tcp_pose()
        pose1 = self.arm.relative_tool_pose(dz=57, init_pose=current_pose).to_list() # 30
        pose2 = self.arm.relative_tool_pose(dz=-170, init_pose=current_pose).to_list()
        poses1 = [pose1, pose2]
        joint_poses = []
        ref_ik_pose = self.CFG["JOINT_IK_DEFAULT"]
        for pose in poses1:
            pose = self.arm.inverse_kinematics(pose, ref_ik_pose)
            joint_poses.append(pose)
        speeds = [5, 25]
        self.arm.move_by_joint_list(joints=joint_poses, speeds=speeds) 
        

    def move(self):
        joint_poses =  [self.CFG["FINALL_POINT"]]
        speeds = [25]
        self.arm.move_by_joint_list(joints=joint_poses,speeds=speeds)

    # ----------------------------------------------------------------------------------------------
    # 动作 8：移动取小盖前对准位姿
    # ----------------------------------------------------------------------------------------------
    def move_inner_cover_pose(self):
        ref_pose = CartesianPose(*self.CFG["POINT_NEW_REF"])
        insert_adjust_base_pose = CartesianPose(*self.CFG["INSERT_ADJUST_POINT_TEMPLATE_REF"])
        insert_adjust_poses = [CartesianPose(*self.CFG["INSERT_ADJUST_POINT"])]
        adjust_joint_pose = self.relative_pose(
            base_pose=insert_adjust_base_pose,
            target_poses= insert_adjust_poses,
            ref_pose = ref_pose,
            motion_type="joint"
        )
        speeds = [10]
        self.arm.move_by_joint_list(adjust_joint_pose, speeds)



    # ----------------------------------------------------------------------------------------------
    # 总入口
    # ----------------------------------------------------------------------------------------------
    def run(self, mode: int, pose=None):
        try:
            if mode == 1:
                self.run_open_cover()
                self.run_screw_cover()
            elif mode == 2:
                self.gripper_action()
            elif mode == 3:
                self.gun_insert_before()
            elif mode == 4:
                self.gun_home()
            elif mode == 5:
                self.inner_cover_close_step1()
            elif mode == 7:
                self.inner_cover_close_step2()
                self.outer_cover_close()
            elif mode == 6:
                self.move()
            elif mode == 8:
                self.move_inner_cover_pose()
            else:
                logger.error("模式错误：请输入 1/2/3/4/5")
        except KeyboardInterrupt:
            logger.warning("手动中断")
            raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("   python3 cover_action_main.py 1    # 开大小盖，移动到取枪对准点")
        print("   python3 cover_action_main.py 2    # 拿枪后移动到插枪对准点")
        print("   python3 cover_action_main.py 3    # 归枪")
        print("   python3 cover_action_main.py 4    # 关大小盖")
        sys.exit(1)

    mode = int(sys.argv[1])
    flow = CoverActionFlow.get_instance()
    if not flow.connect():
        logger.error("连接失败，退出")
        sys.exit(1)
    flow.run(mode)
    flow.disconnect()

    
    

    
