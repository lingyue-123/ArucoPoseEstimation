#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import json
import fcntl
import numpy as np
import asyncio
from crobot_driver_interface import CRobot, CartesianPose, pose_to_homogeneous_matrix, homogeneous_matrix_to_pose, get_flange_relative_move,visualize_trajectory
from gripper_controller import GripperController
from Intergration.FTServo_Linux_main.examples.sms_sts_driver import SMSSTSController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoverAction")

async def to_thread(func,/,*args,**kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None,func,*args,**kwargs)

# with open("/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json", "r", encoding="utf-8") as f:
#     CFG = json.load(f)

class _NullContext:
    # no-op context manager used when timer is None
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _timed(timer, label):
    """Return a context manager that times a motion segment, or nullcontext if timer is None."""
    if timer is None:
        return _NullContext()
    return timer.segment(label)


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
        self.arm.set_speed(20)
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
        self.gripper.set_position(52)

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
    def run_screw_cover(self, timer=None):
        logger.info("\n[动作 2] 开始旋盖")
        with _timed(timer, "模式切换(PP→CSP)"):
            self.arm.switch_motion_model()
        # current_pose = self.arm.get_tcp_pose()
        # pose_dz = [self.arm.relative_tool_pose(dz = -10, init_pose=current_pose).to_list()]
        # speeds = [100]
        # self.arm.move_by_pose_list(poses=pose_dz, speeds=speeds)

        target_pose = self.arm.get_tcp_pose()
        with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data["POINT_NEW_REF"] = target_pose

        with open('/home/nvidia/Downloads/HD/HD_0323/scripts/poses_config.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        self.update_CFG()
        
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
            target_poses=[cartesian_poses[0], cartesian_poses[1],  cartesian_poses[2]],
            ref_pose = ref_pose,
            motion_type="linear"
        )
        speeds = [25,25,25]
        # ret = self.arm.move_by_joint_list(joint_pose,speeds)
        circle_pose = self.relative_pose(
            base_pose=base_pose,
            target_poses=[cartesian_poses[3],cartesian_poses[4],cartesian_poses[5]],
            ref_pose = ref_pose,
            motion_type="linear"
        )
        # self.arm.move_circular(CartesianPose(*circle_pose[0]),CartesianPose(*circle_pose[2]),speed=50)
        # self._require_motion_ok(ret, "run_screw_cover screw trajectory")

        # 伺服运动
        waypoints = [target_pose, joint_pose[0],joint_pose[1],joint_pose[2],circle_pose[0],circle_pose[1],circle_pose[2]]
        # print(waypoints)
        with _timed(timer, "旋盖-伺服轨迹运动"):
            self.arm.plan_and_move_position(waypoints=waypoints, total_time=2.0, dt=0.008, tool_no=10, user_no=0,profile='trapezoid',accel_frac=0.1)
        with _timed(timer, "模式切换(CSP→PP)"):
            self.arm.switch_motion_model()

        self.arm.set_speed(100)
    
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
        with _timed(timer, "旋盖-夹爪张开"):
            self.gripper.set_position(25)
        with _timed(timer, "旋盖-调整位姿"):
            ret = self.arm.move_by_joint_list(adjust_joint_pose,speeds)
        self._require_motion_ok(ret, "run_screw_cover insert adjust")

        # 异步运动
        # async def parallel_task():
        #     await asyncio.gather(to_thread(flow.gripper.set_position, 25),
        #                     to_thread(self.arm.move_by_joint_list, adjust_joint_pose, speeds))
        # asyncio.run(parallel_task())

        logger.info("[动作 2] 旋盖完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 3：夹盖
    # ----------------------------------------------------------------------------------------------
    def gripper_action(self, timer=None):
        with _timed(timer, "模式切换(PP→CSP)"):
            self.arm.switch_motion_model()

        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_COVER_REF"])

        current_pose = self.arm.get_tcp_pose()

        time0 = time.time()

        with _timed(timer, "夹盖-夹爪张开(52)"):
            self.gripper.set_position(52)

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
            CartesianPose(*self.CFG["PUSH_POINT3"]),
            CartesianPose(*self.CFG["PUSH_POINT4"]),
            CartesianPose(*self.CFG["PUSH_POINT4_dz"])
        ]

        poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="linear"
        )
        speeds = [25,25,25]
        plan_points = [current_pose,poses[0],poses[1],poses[2]]
        with _timed(timer, "夹盖-轨迹下沉"):
            self.arm.plan_and_move_position(waypoints=plan_points, total_time=2.0, dt=0.008, tool_no=10, user_no=0,profile='trapezoid',accel_frac=0.2)
        with _timed(timer, "模式切换(CSP→PP)"):
            self.arm.switch_motion_model()
        self.arm.set_speed(100)
        with _timed(timer, "夹盖-夹爪张开(25)"):
            self.gripper.set_position(25)
        t1 = time.time()
        # 确保夹爪张开到25，再退3cm
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
        
        speeds = [25]
        with _timed(timer, "夹盖-相对后退(-30mm)"):
            self.arm.move_by_joint_list([take_gun_poses], speeds)

        target_pose  = self.CFG["PUSH_POINT6_TAKEGUN"]

        # 异步运动
        async def parallel_task():
            await asyncio.gather(to_thread(self.gripper.set_position, 0),
                            to_thread(self.arm.move_by_joint_list, [target_pose], [60]))
        with _timed(timer, "夹盖-夹爪闭合+移动取枪点(异步)"):
            asyncio.run(parallel_task())

    # ----------------------------------------------------------------------------------------------
    # 动作 4：移动到插枪对准点
    # ----------------------------------------------------------------------------------------------
    def gun_insert_before(self, timer=None):
        logger.info("\n[动作 4] 移动到插枪对准点")

        self.update_CFG()
        target_poses = self.CFG["INSERT_BEFORE_TEMPLATE"]
        ik_joint = self.CFG["JOINT_IK_DEFAULT"]
        joint_pose = [self.arm.inverse_kinematics(target_pose=target_poses, initial_joints=ik_joint)]
        speeds = [25]
        # self.arm.move_by_joint_list(joint_pose, speeds)

        # 同时触发舵机抬起和机械臂运动
        def _servo_reset():
            arm_controller = SMSSTSController("/dev/ttysWK1")
            arm_controller.connect()
            arm_controller.reset_position()
            arm_controller.disconnect()

        async def parallel_task():
            await asyncio.gather(to_thread(_servo_reset),
                            to_thread(self.arm.move_by_joint_list, joint_pose, speeds))

        with _timed(timer, "插枪前-舵机复位+关节移动(异步)"):
            asyncio.run(parallel_task())

        print("插枪对准点关节运动完成")

        pose0 = self.arm.get_tcp_pose()
        current_joint = self.arm.get_joint_pose()
        pose1 = self.arm.relative_tool_pose(dz=100,init_pose=pose0).to_list()
        target_joint = [self.arm.inverse_kinematics(target_pose=pose1, initial_joints=current_joint)]
        speeds = [10]
        with _timed(timer, "插枪前-前进100mm"):
            self.arm.move_by_joint_list(target_joint, speeds)

        logger.info("[动作 4] 移动到插枪对准点完成 ")

    # ----------------------------------------------------------------------------------------------
    # 动作 5：归枪
    # ----------------------------------------------------------------------------------------------
    def gun_home(self, timer=None):
        logger.info("\n[动作 5] 归枪")

        ref_pose = CartesianPose(*self.CFG["POINT_NEW_REF"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_REF"])

        pose0 = CartesianPose(*self.CFG["GUN_HOME1"]).to_list()
        pose1 = CartesianPose(*self.CFG["GUN_HOME2"]).to_list()
        gun_init_poses = [pose0, pose1]
        gun_speeds = [30, 5]
        # self.arm.move_by_joint_list(joints=gun_init_poses, speeds=gun_speeds)

        # 同时触发舵机抬起和归枪运动
        def _servo_reset():
            arm_controller = SMSSTSController("/dev/ttysWK1")
            arm_controller.connect()
            arm_controller.reset_position()
            arm_controller.disconnect()

        async def parallel_task():
            await asyncio.gather(to_thread(_servo_reset),
                            to_thread(self.arm.move_by_joint_list, gun_init_poses, gun_speeds))

        with _timed(timer, "归枪-舵机复位+关节移动(异步)"):
            asyncio.run(parallel_task())
       
        gripper = self.gripper
        gripper.set_speed(100)
        with _timed(timer, "归枪-夹爪张开"):
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
        with _timed(timer, "归枪-pose后退150mm"):
            self.arm.move_by_pose_list(poses=poses, speeds=speeds)

        logger.info("[动作 5] 归枪")

    # ----------------------------------------------------------------------------------------------
    # 动作 6：关内盖
    # ----------------------------------------------------------------------------------------------
    def inner_cover_close_step1(self, timer=None):
        gripper = self.gripper
        # gripper.set_position(25)
        
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
        init_speeds = [60]
        # self.arm.move_by_joint_list(joints=init_poses, speeds=init_speeds)

        # 异步运动
        async def parallel_task():
            await asyncio.gather(to_thread(gripper.set_position, 25),
                            to_thread(self.arm.move_by_joint_list, init_poses, init_speeds))
        with _timed(timer, "关内盖1-夹爪+关节移动(异步)"):
            asyncio.run(parallel_task()) 
    
    def inner_cover_close_step2(self, timer=None):
        with _timed(timer, "模式切换(PP→CSP)"):
            self.arm.switch_motion_model() # 模式切换pp->csp
        gripper = self.gripper
        with _timed(timer, "关内盖2-夹爪闭合(52)"):
            gripper.set_position(52)# 夹紧
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
        target_poses = [current_pose, init_pose, poses[0], self.CFG["PUSH_POINT1"], self.CFG["PUSH_POINT2"]]

        with _timed(timer, "关内盖2-伺服轨迹推盖"):
            self.arm.plan_and_move_position(waypoints=target_poses, total_time = 2.5, dt=0.008, tool_no=10, user_no=0)

        with _timed(timer, "关内盖2-夹爪张开(25)"):
            gripper.set_position(25)
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
        
        # target_l = [CartesianPose(*self.CFG["PUSH_POINT6"])]
        # pose_l = self.relative_pose(
        #     base_pose=base_pose,
        #     target_poses=target_l,
        #     ref_pose=ref_pose,
        #     motion_type="joint"
        # )
        # speeds = [30]

        # # 异步运动
        # async def parallel_task():
        #     await asyncio.gather(to_thread(gripper.set_position, 52),
        #                     to_thread(self.arm.move_by_joint_list, pose_l, speeds))
        # asyncio.run(parallel_task()) 


    # ----------------------------------------------------------------------------------------------
    # 动作 7：关外盖
    # ----------------------------------------------------------------------------------------------
    def outer_cover_close(self, timer=None):
        gripper = self.gripper
        ref_pose = CartesianPose(*self.CFG["PUSH_POINT2"])
        base_pose = CartesianPose(*self.CFG["POINT_TEMPLATE_COVEROFF_REF"])
        current_pose = self.arm.get_tcp_pose()

        all_targets = [
            CartesianPose(*self.CFG["OUTER_CLOSE_0"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_1"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_2"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_3"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_4"]),
            CartesianPose(*self.CFG["OUTER_CLOSE_5"]),
            CartesianPose(*self.CFG["OUTER_CLOSE"]),
        ]

        init_poses = self.relative_pose(
            base_pose=base_pose,
            target_poses=all_targets,
            ref_pose=ref_pose,
            motion_type="linear"
        )
        init_poses.insert(0, current_pose)

        # 异步运动
        def _gripper_close_async():
            time.sleep(1.5)
            gripper.set_position(52)
        async def parallel_task():
            await asyncio.gather(to_thread(_gripper_close_async),
                            to_thread(self.arm.plan_and_move_position, init_poses, 4.0, 0.008, 10, 0, True, True, 'trapezoid', 0.1))
        with _timed(timer, "关外盖-延迟夹爪+伺服轨迹(异步)"):
            asyncio.run(parallel_task())

        with _timed(timer, "模式切换(CSP→PP)"):
            self.arm.switch_motion_model()
        current_pose = self.arm.get_tcp_pose()
        pose1 = self.arm.relative_tool_pose(dz=14, init_pose=current_pose).to_list() # 30
        pose2 = self.arm.relative_tool_pose(dz=-170, init_pose=current_pose).to_list()
        poses1 = [pose1, pose2]

        # PP模式
        joint_poses = []
        ref_ik_pose = self.CFG["JOINT_IK_DEFAULT"]
        for pose in poses1:
            pose = self.arm.inverse_kinematics(pose, ref_ik_pose)
            joint_poses.append(pose)
        speeds = [10, 30]
        with _timed(timer, "关外盖-上升14mm+退出170mm"):
            self.arm.move_by_joint_list(joints=joint_poses, speeds=speeds)

        ## CSP模式
        # poses1.insert(0,current_pose)
        # self.arm.plan_and_move_position(waypoints=poses1, total_time=5)
        # self.arm.switch_motion_model() # 模式切换csp->pp


        

    def move(self, timer=None):
        joint_poses =  [self.CFG["FINALL_POINT"]]
        speeds = [60]
        with _timed(timer, "回终点位"):
            self.arm.move_by_joint_list(joints=joint_poses,speeds=speeds)

    # ----------------------------------------------------------------------------------------------
    # 动作 8：移动取小盖前对准位姿
    # ----------------------------------------------------------------------------------------------
    def move_inner_cover_pose(self, timer=None):
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
        with _timed(timer, "取小盖前对准移动"):
            self.arm.move_by_joint_list(adjust_joint_pose, speeds)



    # ----------------------------------------------------------------------------------------------
    # 总入口
    # ----------------------------------------------------------------------------------------------
    def run(self, mode: int, pose=None, timer=None):
        try:
            if mode == 1:
                # self.run_open_cover()
                self.run_screw_cover(timer=timer)
            elif mode == 2:
                self.gripper_action(timer=timer)
            elif mode == 3:
                self.gun_insert_before(timer=timer)
            elif mode == 4:
                self.gun_home(timer=timer)
            elif mode == 5:
                self.inner_cover_close_step1(timer=timer)
            elif mode == 7:
                self.inner_cover_close_step2(timer=timer)
                self.outer_cover_close(timer=timer)
            elif mode == 6:
                self.move(timer=timer)
                pass
            elif mode == 8:
                self.move_inner_cover_pose(timer=timer)
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

    
    

    
