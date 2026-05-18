import logging
import os
import sys
from typing import Optional
import math

import numpy as np
import time

from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass   

# 假设你已有的基类（无需修改）
from base import RobotBase

logger = logging.getLogger(__name__)

class CartesianPose:
    """笛卡尔位姿数据类（定义在类外部，全局可用）"""
    def __init__(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz
    def __repr__(self):
        return f"CartesianPose(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, rx={self.rx:.3f}, ry={self.ry:.3f}, rz={self.rz:.3f})"


def _ensure_sdk_path(sdk_path: str) -> None:
    """确保 JAKA SDK 路径在 sys.path 和 LD_LIBRARY_PATH 中。"""
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if sdk_path not in ld_path:
        os.environ['LD_LIBRARY_PATH'] = sdk_path + ':' + ld_path
        os.execv(sys.executable, [sys.executable] + sys.argv)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


class JAKARobot(RobotBase):
    """
    JAKA 协作机器人完整接口（对齐官方 jkrc SDK，补全关节运动、圆弧运动、状态获取）
    """
    ABS = 0  # 绝对运动
    INCR = 1 # 相对运动

    def __init__(self, ip: str, sdk_path: str):
        self._ip = ip
        if not os.path.isabs(sdk_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sdk_path = os.path.join(project_root, sdk_path)
        self._sdk_path = sdk_path
        self._robot = None
        self._connected = False
        self._powered = False
        self._enabled = False

        # 运动速度
        self._speed_mmps = 1000       # 笛卡尔速度 mm/s
        self._speed_joint_deg = 10  # 关节速度 deg/s

        _ensure_sdk_path(sdk_path)

    # ========================= 连接 / 上电 / 使能 =========================
    def connect(self) -> bool:
        try:
            import jkrc
            self._robot = jkrc.RC(self._ip)
            ret = self._robot.login()
            if ret[0] == 0:
                self._connected = True
                logger.info("机械臂连接成功: %s", self._ip)
                return True
            else:
                logger.warning("登录失败，错误码: %s", ret)
                return False
        except Exception as e:
            logger.error("连接异常: %s", e)
            return False

    def robot_powered(self):
        if not self._connected:
            return False
        ret = self._robot.power_on()
        if ret[0] == 0:
            self._powered = True
            logger.info("机械臂上电成功")
        return ret[0] == 0

    def robot_enable(self):
        if not self._powered:
            return False
        ret = self._robot.enable_robot()
        if ret[0] == 0:
            self._enabled = True
            logger.info("机械臂使能成功")
        return ret[0] == 0

    def disconnect(self) -> None:
        if self._robot:
            try:
                self._robot.logout()
            except:
                pass
        self._connected = False
        self._powered = False
        self._enabled = False
        logger.info("机械臂已断开")

    # ========================= 速度设置 =========================
    def set_cartesian_speed(self, speed_mmps: float):
        self._speed_mmps = speed_mmps

    def set_joint_speed(self, speed_deg: float):
        self._speed_joint_deg = speed_deg

    # ========================= 位姿转换 =========================
    def _target_to_jaka_pose(self, target) -> list:
        return [
            float(target.x), float(target.y), float(target.z),
            float(np.radians(target.rx)),
            float(np.radians(target.ry)),
            float(np.radians(target.rz)),
        ]

    def _joint_to_jaka_angle(self, j6_list: list) -> list:
        return [np.radians(a) for a in j6_list]

    # ========================= 获取位姿 =========================
    def get_tcp_pose(self) -> Optional[tuple]:
        if not self._connected:
            return None
        try:
            ret = self._robot.get_tcp_position()
            if ret[0] == 0:
                x, y, z, rx, ry, rz = ret[1]
                return (x, y, z,
                        np.degrees(rx),
                        np.degrees(ry),
                        np.degrees(rz))
        except:
            return None

    def get_joint_pose(self) -> Optional[list]:
        """获取关节角度 [j1-j6] 单位：度"""
        if not self._connected:
            return None
        try:
            ret = self._robot.get_joint_position()
            if ret[0] == 0:
                return [np.degrees(a) for a in ret[1]]
        except:
            return None

    # ========================= 坐标系 / 工具 =========================
    def get_coord_sys(self) -> Optional[int]:
        try:
            ret = self._robot.get_tool_id()
            return ret[1] if ret[0] == 0 else None
        except:
            return None

    def set_tool_id(self, tool_id: int) -> bool:
        try:
            ret = self._robot.set_tool_id(tool_id)
            return ret[0] == 0
        except:
            return False

    # ========================= 直线运动 =========================
    def move_linear(self, target) -> int:
        """
        绝对直线运动（官方格式）
        linear_move(目标位姿, 模式, 是否阻塞, 速度)
        """
        if not self._enabled:
            return -1
        pos = self._target_to_jaka_pose(target)
        ret = self._robot.linear_move(pos, self.ABS, True, self._speed_mmps)
        return ret[0]

    def move_linear_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0) -> int:
        """
        相对直线运动（官方格式：Z轴-30mm 例子）
        linear_move(目标, INCR, True, 速度)
        """
        if not self._enabled:
            return -1
        delta = [
            dx, dy, dz,
            np.radians(drx),
            np.radians(dry),
            np.radians(drz)
    ]
        # ✅ 完全和你官方代码一样：linear_move(pos, INCR, True, 速度)
        ret = self._robot.linear_move(delta, self.INCR, False, self._speed_mmps)
        return ret[0]

    def move_linear_block(self, target, speed=None) -> int:
        """阻塞式直线运动（官方示例用的 True 阻塞）"""
        if not self._enabled:
            return -1
        pos = self._target_to_jaka_pose(target)
        speed = speed or self._speed_mmps
        ret = self._robot.linear_move(pos, self.ABS, True, speed)
        return ret[0]

    def move_linear_relative_block(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, speed=10) -> int:
        if not self._enabled:
            return -1
        delta = [
            dx, dy, dz,
            np.radians(drx),
            np.radians(dry),
            np.radians(drz)
        ]
        # 完全匹配：linear_move(tcp_pos, INCR, True, 10)
        ret = self._robot.linear_move(delta, self.INCR, True, speed)
        return ret[0]
    
    # ========================= 关节运动正逆解 =========================
        # ========================= 运动学接口（正解/逆解） =========================
    def kine_inverse(self, ref_pos, cartesian_pose):
        """
        逆运动学求解（角度单位：度）
        :param ref_pos: 参考关节位置，list/tuple (6个元素)，单位度
        :param cartesian_pose: 目标笛卡尔位姿，可接受 CartesianPose 对象 或 (x,y,z,rx,ry,rz) 元组，角度单位为度
        :return: (ret_code, joint_pos_deg)  
                 ret_code=0 成功，joint_pos_deg 为 [j1,...,j6] 度  
                 失败返回 (非0, 错误信息)
        """
        if self._robot is None:
            return (-1, "robot not initialized")

        # 转换参考关节角：度 -> 弧度
        if isinstance(ref_pos, (list, tuple)) and len(ref_pos) == 6:
            ref_rad = [math.radians(v) for v in ref_pos]
        else:
            return (-2, "ref_pos must be list/tuple of 6 joint angles in degrees")

        # 转换目标位姿：统一转为 (x,y,z,rx,ry,rz) 弧度
        if isinstance(cartesian_pose, CartesianPose):
            pose_rad = [
                cartesian_pose.x, cartesian_pose.y, cartesian_pose.z,
                math.radians(cartesian_pose.rx),
                math.radians(cartesian_pose.ry),
                math.radians(cartesian_pose.rz)
            ]
        elif isinstance(cartesian_pose, (list, tuple)) and len(cartesian_pose) == 6:
            pose_rad = [
                cartesian_pose[0], cartesian_pose[1], cartesian_pose[2],
                math.radians(cartesian_pose[3]),
                math.radians(cartesian_pose[4]),
                math.radians(cartesian_pose[5])
            ]
        else:
            return (-3, "cartesian_pose must be CartesianPose or tuple/list of 6 values (x,y,z,rx,ry,rz) in degrees")

        try:
            ret = self._robot.kine_inverse(ref_rad, pose_rad)
            if ret[0] == 0:
                # 成功：返回的关节角度是弧度 -> 转为度
                joint_deg = [math.degrees(rad) for rad in ret[1]]
                return (0, joint_deg)
            else:
                return (ret[0], f"kine_inverse failed, error code: {ret[0]}")
        except Exception as e:
            return (-4, f"exception in kine_inverse: {e}")

    def kine_forward(self, joint_pos):
        """
        正运动学求解（角度单位：度）
        :param joint_pos: 关节空间位置，list/tuple (6个元素)，单位度
        :return: (ret_code, cartesian_pose)  
                 ret_code=0 成功，cartesian_pose 为 (x,y,z,rx,ry,rz) 元组，角度单位为度  
                 失败返回 (非0, 错误信息)
        """
        if self._robot is None:
            return (-1, "robot not initialized")

        if not (isinstance(joint_pos, (list, tuple)) and len(joint_pos) == 6):
            return (-2, "joint_pos must be list/tuple of 6 joint angles in degrees")

        # 度 -> 弧度
        joint_rad = [math.radians(v) for v in joint_pos]

        try:
            ret = self._robot.kine_forward(joint_rad)
            if ret[0] == 0:
                x, y, z, rx_rad, ry_rad, rz_rad = ret[1]
                # 弧度 -> 度
                pose_deg = (x, y, z, math.degrees(rx_rad), math.degrees(ry_rad), math.degrees(rz_rad))
                return (0, pose_deg)
            else:
                return (ret[0], f"kine_forward failed, error code: {ret[0]}")
        except Exception as e:
            return (-3, f"exception in kine_forward: {e}")

    # ========================= 扩展直线运动（你给的扩展接口） =========================
    def move_linear_extend(self, target, speed=20, acc=5, tol=0.1) -> int:
        if not self._enabled:
            return -1
        pos = self._target_to_jaka_pose(target)
        ret = self._robot.linear_move_extend(pos, self.ABS, False, speed, acc, tol)
        return ret[0]



    # ========================= 关节运动（新增！） =========================
    def move_joint(self, joint_target_deg: list, speed_joint) -> int:
        """
        关节运动（对齐官方SDK）
        :param joint_target_deg: [j1,j2,j3,j4,j5,j6] 单位：度
        """
        if not self._enabled:
            return -1
        # 转弧度
        j_rad = self._joint_to_jaka_angle(joint_target_deg)
        
        # ✅ 完全对齐官方调用格式！！！
        ret = self._robot.joint_move(j_rad, self.ABS, True, speed_joint)
        return ret[0]


    # ========================= 圆弧运动（新增！） =========================
    def move_circular(self, mid_pose, end_pose) -> int:
        """
        圆弧运动：当前 -> 中间点 -> 终点
        """
        if not self._enabled:
            return -1
        mid = self._target_to_jaka_pose(mid_pose)
        end = self._target_to_jaka_pose(end_pose)
        ret = self._robot.circular_move(mid, end, False, self._speed_mmps)
        return ret[0]

    # ========================= 运动等待（补全！） =========================
    def wait_move_done(self, timeout=30) -> bool:
        """轮询等待运动完成"""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_moving() is False:
                return True
            time.sleep(0.1)
        logger.warning("运动超时")
        return False

    def is_moving(self) -> bool:
        """机械臂是否正在运动"""
        try:
            ret = self._robot.get_robot_state()
            return ret[1] == 1  # 1=运动中 0=停止
        except:
            return False

    # ========================= 基类要求的必选方法：move_and_wait =========================
    def move_and_wait(self, target) -> bool:
        """
        基类 RobotBase 要求的抽象方法：运动并等待完成
        """
        ret = self.move_linear(target)
        if ret != 0:
            logger.error("move_and_wait 运动启动失败，错误码：%s", ret)
            return False
        return self.wait_move_done()

    # ========================= 安全状态 =========================
    def is_in_error(self) -> bool:
        try:
            ret = self._robot.get_error_code()
            return ret[1] != 0
        except:
            return True

    # ========================= with 语句 =========================
    def __enter__(self) -> 'JAKARobot':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @property
    def is_connected(self):
        return self._connected
    


    def pose_to_homogeneous_matrix(self, pose: CartesianPose, degrees=True) -> np.ndarray:
        """位姿 → 齐次矩阵"""
        euler_angles = [pose.rz, pose.ry, pose.rx]
        r = R.from_euler('ZYX', euler_angles, degrees=degrees)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = [pose.x, pose.y, pose.z]
        return T

    def homogeneous_matrix_to_pose(self, matrix: np.ndarray, degrees=True) -> CartesianPose:
        """齐次矩阵 → 位姿"""
        if matrix.shape != (4, 4):
            raise ValueError(f"矩阵必须是4x4，当前：{matrix.shape}")
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        rot_matrix = matrix[:3, :3]
        r = R.from_matrix(rot_matrix)
        euler_angles = r.as_euler('ZYX', degrees=degrees)
        rz, ry, rx = euler_angles
        return CartesianPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
    
    def compute_relative_transform(self, base_pose: CartesianPose, target_pose: CartesianPose) -> np.ndarray:
        """
        计算相对变换矩阵 T，使得 target_matrix = base_matrix @ T
        :param base_pose: 基准位姿
        :param target_pose: 目标位姿
        :return: 4x4 齐次变换矩阵 T
        """
        T_base = self.pose_to_homogeneous_matrix(base_pose)
        T_target = self.pose_to_homogeneous_matrix(target_pose)
        T_rel = np.linalg.inv(T_base) @ T_target
        return T_rel

    def move_relative_tool(self, dx=0.0, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0, mode='linear', wait=True):
        """
        工具坐标系下的增量运动（你要的完整功能）
        :param dx, dy, dz: 工具系位移 mm
        :param drx, dry, drz: 工具系旋转 度
        :param mode: 'linear' 直线 / 'joint' 关节
        :param wait: 是否等待完成
        :return: 0成功, -1失败, -2超时
        """
        if not self._enabled:
            logger.error("未使能，无法运动")
            return -1

        # 1. 获取当前位姿
        current = self.get_tcp_pose()
        if current is None:
            logger.error("获取当前位姿失败")
            return -1

        x, y, z, rx, ry, rz = current
        current_pose = CartesianPose(x, y, z, rx, ry, rz)

        # 2. 当前位姿 → 齐次矩阵
        T_base_tcp = self.pose_to_homogeneous_matrix(current_pose, degrees=True)

        # 3. 工具系增量（角度是度）
        delta_pose = CartesianPose(dx, dy, dz, drx, dry, drz)
        T_tcp_delta = self.pose_to_homogeneous_matrix(delta_pose, degrees=True)

        # 4. 计算目标位姿
        T_target = T_base_tcp @ T_tcp_delta
        target_pose = self.homogeneous_matrix_to_pose(T_target, degrees=True)

        # 5. 运动
        logger.info(f"工具系相对运动 dz={dz:.1f}mm")

        if mode == 'linear':
            ret = self.move_linear(target_pose)
        else:
            # 关节运动需要解算IK，这里直接用MoveL替代更安全
            ret = self.move_linear(target_pose)

        if ret != 0:
            logger.error(f"运动启动失败，错误码：{ret}")
            return -1

        # 6. 等待
        if wait:
            ok = self.wait_move_done(timeout=30)
            return 0 if ok else -2

        return 0
    

    def calculate_linear_distance(self, pose1: CartesianPose, pose2: CartesianPose) -> float:
        """
        计算两组法兰笛卡尔位姿之间的直线距离（单位：mm）
        核心逻辑：基于3D空间两点间距离公式，仅计算XYZ坐标的直线距离（旋转角不影响直线距离）
        
        注意事项：
        1.  传入参数必须是CartesianPose对象，与jaka_driver_interface.py中直线运动（move_linear等）的target参数格式完全一致；
        2.  距离计算仅基于X、Y、Z三个平移坐标，RX、RY、RZ旋转角不参与计算（直线距离与姿态无关）；
        3.  返回值保留3位小数，单位为mm，符合机械臂实际操作精度需求；
        4.  接口独立无依赖（仅依赖CartesianPose类），可直接调用，无需机器人连接。
        
        :param pose1: 第一组法兰笛卡尔位姿（CartesianPose(x,y,z,rx,ry,rz)）
        :param pose2: 第二组法兰笛卡尔位姿（CartesianPose(x,y,z,rx,ry,rz)）
        :return: 两组位姿间的直线距离（单位：mm），保留3位小数
        """
        # 3D空间两点间直线距离公式：√[(x2-x1)² + (y2-y1)² + (z2-z1)²]
        distance = math.sqrt(
            (pose2.x - pose1.x) ** 2 +
            (pose2.y - pose1.y) ** 2 +
            (pose2.z - pose1.z) ** 2
        )
        print("distance (mm): ", distance)
        # 保留3位小数，适配机械臂操作精度
        return round(distance, 3)
    
if __name__ == "__main__":

    arm = JAKARobot("192.168.1.106", "/home/nvidia/Downloads/jaka-python-sdk")

    arm.connect()
    # arm.robot_powered()
    # arm.robot_enable()
    ## 关节运动
    current_joint = arm.get_joint_pose()
    print("当前关节角：", current_joint)

    test_joint = [91.191, 51.477, -100.181, 204.620, 90.350, -263.764]
    ret, pose = arm.kine_forward(test_joint)
    if ret == 0:
        print("正解位姿(度):", pose)
    else:
        print("正解失败:", pose)
    target_pose = CartesianPose(x=-110.809, y=634.292, z=245.709, rx=166.013, ry=65.132, rz=-103.778)
    ret, solution = arm.kine_inverse(current_joint, target_pose)
    if ret == 0:
        print("逆解成功，关节角(度):", solution)
    else:
        print("逆解失败:", solution)
    # if current_joint is not None:
    #     target_joint = current_joint.copy()
    #     target_joint[0] = target_joint[0] + 5.0  # 关节1索引是0
    #     print("目标关节角：", target_joint)
    #     arm.move_joint(target_joint)
    #     arm.wait_move_done()
    #     print("运动完成！")
    #     final_joint = arm.get_joint_pose()
    #     print("运动后关节角：", final_joint)

    ## 直线运动
    # current_pose = arm.get_tcp_pose()
    # print("当前 TCP 位姿：", current_pose)
    # # ---------------------
    # # 直线运动测试：Z 轴向上 20 mm
    # # ---------------------
    # print("开始直线运动：Z 轴 +20mm")
    # arm.move_linear_relative_block(dz=+20, speed=100)
    # print("运动完成！")
    # # 查看运动后位置
    # new_pose = arm.get_tcp_pose()
    # print("运动后位姿：", new_pose)

    ## 法兰坐标系运动
    # arm.move_relative_tool(dz=-100, mode='linear', wait=True)
    # print("工具系相对运动完成！") 
    # print("新位姿：", arm.get_tcp_pose()) 

    ## 算直线运动的距离
    # target_pose_1 = CartesianPose(x=180.300, y=369.106, z=324.999, rx=-176.139, ry=67.990, rz=-88.553)
    # target_pose_2 = CartesianPose(x=189.504, y=563.076, z=246.710, rx=-176.139, ry=67.990, rz=-88.553)
    # arm.calculate_linear_distance(target_pose_1, target_pose_2)

    arm.disconnect()

