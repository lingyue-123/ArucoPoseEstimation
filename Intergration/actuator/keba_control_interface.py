import numpy as np
import ctypes
import time
import struct
import threading
import sys
from typing import Optional, Tuple, List
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

from scipy.spatial.transform import Rotation as R

import ctypes
import logging
import math
import os



class JointPose:
    """关节角度数据类，包含 Q1~Q6 (单位：弧度，与C库保持一致)"""
    def __init__(self, Q1=0.0, Q2=0.0, Q3=0.0, Q4=0.0, Q5=0.0, Q6=0.0):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3
        self.Q4 = Q4
        self.Q5 = Q5
        self.Q6 = Q6

    def to_list(self) -> List[float]:
        return [self.Q1, self.Q2, self.Q3, self.Q4, self.Q5, self.Q6]

    def __repr__(self):
        return f"JointPose(Q1={self.Q1:.6f}, Q2={self.Q2:.6f}, Q3={self.Q3:.6f}, Q4={self.Q4:.6f}, Q5={self.Q5:.6f}, Q6={self.Q6:.6f})"


class CartesianPose:
    """笛卡尔位姿数据类，包含 X, Y, Z (米) 和 Roll, Pitch, Yaw (弧度)"""
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self):
        return f"CartesianPose(x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f}, roll={self.roll:.6f}, pitch={self.pitch:.6f}, yaw={self.yaw:.6f})"


class RobotInterface:
    # Modbus 寄存器地址常量
    REG_POWERON_WRITE = 2
    REG_START_WRITE = 3
    REG_MODE_WRITE = 7
    REG_MODE3TARGET_WRITE = 10000
    REG_JOINTDATA_WRITE = 15200
    REG_JOINTDATASENDSTATE_WRITE = 15212
    REG_JOINTDATA_READ = 12288
    REG_MODE_READ = 8
    REG_MOTIONDONE_READ = 6
    REG_POWERON_READ = 4

    def __init__(self, kin_lib_path: str = "/home/nvidia/SWG/ToCamera/RobotControlkeba.so", heartbeat_interval: float = 5.0):
        """        print("连接失败")

        初始化机器人接口
        :param kin_lib_path: 正逆运动学 C 共享库的路径
        :param kin_lib_path: 正逆运动学 C 共享库的路径
        :param heartbeat_interval: 心跳间隔(秒),默认5秒
        """
        self.client: Optional[ModbusTcpClient] = None
        self.lock = threading.Lock()
        self.robot_ip = "192.168.1.133"
        self.port = 502

        # 心跳线程
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # 加载运动学共享库
        self.kin_lib = None
        try:
            self.kin_lib = ctypes.CDLL(kin_lib_path)
            self._setup_kin_lib()
            print(f"[RobotInterface] 运动学库加载成功: {kin_lib_path}")
        except Exception as e:
            print(f"[RobotInterface] 运动学库加载失败: {e}，正逆运动学功能不可用")

    def _setup_kin_lib(self):
        """配置 ctypes 函数参数和返回类型"""
        # 定义 C 结构体（必须与库定义完全一致）
        class EulerAngles2(ctypes.Structure):
            _fields_ = [
                ("X", ctypes.c_double),
                ("Y", ctypes.c_double),
                ("Z", ctypes.c_double),
                ("roll", ctypes.c_double),
                ("pitch", ctypes.c_double),
                ("yaw", ctypes.c_double)
            ]

        class JntAngle(ctypes.Structure):
            _fields_ = [
                ("J0", ctypes.c_double),
                ("J1", ctypes.c_double),
                ("J2", ctypes.c_double),
                ("J3", ctypes.c_double),
                ("J4", ctypes.c_double),
                ("J5", ctypes.c_double)
            ]

        # 保存结构体类型供实例方法使用
        self._EulerAngles2 = EulerAngles2
        self._JntAngle = JntAngle

        # 配置 JnttoZYX 函数
        self.kin_lib.JnttoZYX.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        ]
        self.kin_lib.JnttoZYX.restype = EulerAngles2

        # 配置 ZYXtoJntAngle 函数
        self.kin_lib.ZYXtoJntAngle.argtypes = [
            ctypes.POINTER(EulerAngles2),
            ctypes.POINTER(JntAngle)
        ]
        self.kin_lib.ZYXtoJntAngle.restype = JntAngle

    # ==================== 心跳线程 ====================
    def _heartbeat(self):
        """每隔5s读一次寄存器保持连接"""
        while self.heartbeat_running:
            time.sleep(5)
            if self.client and self.client.is_socket_open():
                try:
                    self.client.read_holding_registers(address=self.REG_POWERON_READ, count = 1)
                    print("[Heartbeat] 心跳正常")
                except Exception as e:
                    print("[Heartbeat] 心跳异常")
            else:
                print("[Heartbeat] 连接已断开，尝试重连...")
                self.__reconnect()

    def _reconnect(self):
        """重连函数"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                print(f"[Reconnect] 第{attempt + 1}次重新连接...")
                if self.client:
                    self.client.close()
                self.client = ModbusTcpClient(host=self.robot_ip, port=self.port, timeout=5.0)

                if self.client.connect():
                    print(f"[Reconnect] 重连成功")
                    return True
            except Exception as e:
                print(f"[Reconnect] 重连失败: {e}")
                time.sleep(1)
        print("[Reconnect] 重连失败，请检查网络")
        return False




    # ==================== 运动学接口 ====================

    def forward_kinematics(self, joint_pose: JointPose) -> Optional[CartesianPose]:
        """
        正运动学：关节角度 -> 笛卡尔位姿
        :param joint_pose: 关节角度 (弧度)
        :return: CartesianPose 对象，失败返回 None
        """
        if self.kin_lib is None:
            print("[forward_kinematics] 运动学库未加载")
            return None

        # 创建占位矩阵 (C库可能需要)
        out_matrix = np.zeros((6, 1), dtype=np.float64, order="C")

        try:
            # 调用 C 函数
            pose = self.kin_lib.JnttoZYX(
                joint_pose.Q1, joint_pose.Q2, joint_pose.Q3,
                joint_pose.Q4, joint_pose.Q5, joint_pose.Q6,
                out_matrix
            )
            return CartesianPose(
                x=pose.X, y=pose.Y, z=pose.Z,
                roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw
            )
        except Exception as e:
            print(f"[forward_kinematics] 调用失败: {e}")
            return None

    def inverse_kinematics(self, target_pose: CartesianPose,
                           current_joint: Optional[JointPose] = None) -> Optional[JointPose]:
        """
        逆运动学：笛卡尔位姿 -> 关节角度
        :param target_pose: 目标笛卡尔位姿
        :param current_joint: 当前关节角（用于选择最优解），若为 None 则自动读取机器人当前关节角
        :return: JointPose 对象，失败返回 None
        """
        if self.kin_lib is None:
            print("[inverse_kinematics] 运动学库未加载")
            return None

        # 如果没有提供当前关节角，尝试从机器人读取
        if current_joint is None:
            current_joint = self.get_keba_joint_data()
            if current_joint is None:
                print("[inverse_kinematics] 无法获取当前关节角，且未提供")
                return None

        # 构建目标位姿结构体
        target = self._EulerAngles2()
        target.X = target_pose.x
        target.Y = target_pose.y
        target.Z = target_pose.z
        target.roll = target_pose.roll
        target.pitch = target_pose.pitch
        target.yaw = target_pose.yaw

        # 构建当前关节角结构体
        curr = self._JntAngle()
        curr.J0 = current_joint.Q1
        curr.J1 = current_joint.Q2
        curr.J2 = current_joint.Q3
        curr.J3 = current_joint.Q4
        curr.J4 = current_joint.Q5
        curr.J5 = current_joint.Q6

        try:
            result = self.kin_lib.ZYXtoJntAngle(ctypes.byref(target), ctypes.byref(curr))
            return JointPose(
                Q1=result.J0, Q2=result.J1, Q3=result.J2,
                Q4=result.J3, Q5=result.J4, Q6=result.J5
            )
        except Exception as e:
            print(f"[inverse_kinematics] 调用失败: {e}")
            return None

    # ==================== Modbus 接口（原代码保持不变） ====================

    def init_robot(self, ip: str, port: int = 502, timeout: float = 10.0) -> bool:
        """初始化 Modbus TCP 连接并完成机器人上电启动流程"""
        self.robot_ip = ip
        self.port = port

        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[Init] 第 {attempt + 1} 次尝试连接 {ip}:{port}...")
                self.client = ModbusTcpClient(host=ip, port=port, timeout=timeout)
                if self.client.connect():
                    print(f"[Init] 连接成功: {ip}:{port}")
                    
                    # ==================== 修改2: 连接成功后启动心跳线程 ====================
                    self.heartbeat_running = True
                    self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
                    self.heartbeat_thread.start()
                    print("[Heartbeat] 心跳线程已启动")
                    return True
                else:
                    print(f"[Init] 第 {attempt + 1} 次连接失败")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 等待2秒后重试
            except Exception as e:
                print(f"[Init] 连接异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        print(f"[Init] 连接失败，已重试 {max_retries} 次")
        self.client = None
        return False



        # self.client = ModbusTcpClient(host=ip, port=port, timeout=timeout)
        # if not self.client.connect():
        #     print(f"[Init] 连接失败: {ip}:{port}")
        #     self.client = None
        #     return False
        
        # else:
        #     print("连接成功")

        # with self.lock:
        #     try:
        #         self.client.write_register(self.REG_POWERON_WRITE, 1)
        #         print("上电成功")
        #     except ModbusException as e:
        #         print(f"[Init] 上电写失败: {e}")
        #         return False

        # time.sleep(0.005)

        # with self.lock:
        #     try:
        #         self.client.write_register(self.REG_START_WRITE, 1)
        #     except ModbusException as e:
        #         print(f"[Init] 启动写失败: {e}")
        #         return False

        # print(f"[Init] 连接成功: {ip}:{port}")
        return True

    def arm_PowerON(self) -> bool:
        """机器人上电"""
        if not self.client:
            print("[arm_PowerON] 未初始化连接")
            return False

        with self.lock:
            try:
                self.client.write_register(self.REG_POWERON_WRITE, 1)
            except ModbusException as e:
                print(f"[arm_PowerON] 上电失败: {e}")
                return False

        # time.sleep(0.005)

        with self.lock:
            try:
                self.client.write_register(self.REG_START_WRITE, 1)
            except ModbusException as e:
                print(f"[arm_PowerON] 启动失败: {e}")
                return False

        print("[arm_PowerON] 上电完成")
        return True
    
    def arm_PowerOff(self) -> bool:
        """机器人下电"""
        if not self.client:
            print("[arm_PowerON] 未初始化连接")
            return False

        with self.lock:
            try:
                self.client.write_register(self.REG_POWERON_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_PowerOff] 下电失败: {e}")
                return False

        # time.sleep(0.005)

        with self.lock:
            try:
                self.client.write_register(self.REG_START_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_PowerOff] 启动失败: {e}")
                return False

        print("[arm_PowerOff] 下电完成")
        return True

    def arm_close(self) -> bool:
        """关闭机器人（停止运动并断电）"""
        if not self.client:
            print("[arm_close] 未初始化连接")
            return False

        with self.lock:
            try:
                self.client.write_register(self.REG_START_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_close] 停止运动失败: {e}")
                return False

        time.sleep(0.005)

        with self.lock:
            try:
                self.client.write_register(self.REG_POWERON_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_close] 断电失败: {e}")
                return False

        print("[arm_close] 机器人已关闭")
        return True

    @staticmethod
    def float_to_regs(f_val: float) -> Tuple[int, int]:
        """将 float 转换为两个 16 位寄存器值（大端）"""
        packed = struct.pack('>f', f_val)
        high = (packed[0] << 8) | packed[1]
        low = (packed[2] << 8) | packed[3]
        return high, low

    @staticmethod
    def registers_to_float(high: int, low: int) -> float:
        """将两个 16 位寄存器值恢复为 float（大端）"""
        packed = bytes([(high >> 8) & 0xFF, high & 0xFF,
                        (low >> 8) & 0xFF, low & 0xFF])
        return struct.unpack('>f', packed)[0]

    def get_registers(self, pose: JointPose) -> List[int]:
        """将 JointPose 转换为 12 个 uint16 的列表"""
        regs = []
        for q in (pose.Q1, pose.Q2, pose.Q3, pose.Q4, pose.Q5, pose.Q6):
            high, low = self.float_to_regs(q)
            regs.extend([high, low])
        return regs

    def get_value_from_registers(self, raw_data: List[int]) -> JointPose:
        """从 12 个 uint16 的列表中恢复 JointPose"""
        if len(raw_data) < 12:
            raise ValueError("寄存器数据长度不足12")
        return JointPose(
            Q1=self.registers_to_float(raw_data[0], raw_data[1]),
            Q2=self.registers_to_float(raw_data[2], raw_data[3]),
            Q3=self.registers_to_float(raw_data[4], raw_data[5]),
            Q4=self.registers_to_float(raw_data[6], raw_data[7]),
            Q5=self.registers_to_float(raw_data[8], raw_data[9]),
            Q6=self.registers_to_float(raw_data[10], raw_data[11])
        )

    def write_pose_registers(self, pose: JointPose) -> bool:
        """将目标关节角度写入 Modbus 寄存器区"""
        if not self.client:
            print("[write_pose_registers] 未初始化连接")
            return False

        regs = self.get_registers(pose)
        with self.lock:
            try:
                self.client.write_registers(self.REG_JOINTDATA_WRITE, regs)
            except ModbusException as e:
                print(f"[write_pose_registers] 写入失败: {e}")
                return False
        return True

    def arm_move_joint(self, target: JointPose) -> bool:
        """关节运动：设置模式为关节模式，下发目标角度，触发运动"""
        if not self.client:
            print("[arm_move_joint] 未初始化连接")
            return False
        
        current_joint = self.get_keba_joint_data()
        if current_joint is None:
            print("[arm_move_joint] 获取当前关节角失败")
            return False

        # 设置模式为0（关节模式）
        # mode = 0
        with self.lock:
            try:
                self.client.write_register(self.REG_MODE_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_move_joint] 写模式失败: {e}")
                return False

        time.sleep(0.005)

        if not self.write_pose_registers(target):
            return False

        time.sleep(0.005)

        # 触发运动
        with self.lock:
            try:
                self.client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 1)
            except ModbusException as e:
                print(f"[arm_move_joint] 触发运动失败: {e}")
                return False

        time.sleep(0.5)

        # 清除触发标志
        with self.lock:
            try:
                self.client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_move_joint] 清除触发标志失败: {e}")
                return False
            
        # 轮询当前位置是否为移动到的位置
        # while True:
        #     if self.is_joint_equal(self.arm_get_current_joint(), target):
        #         break
        print("[arm_move_joint] 关节运动指令下发成功")
        return True
    def is_joint_equal(self, j1: JointPose, j2: JointPose, decimal: int = 1) -> bool:
        """
        比较两个关节角是否相等（保留指定位小数，默认2位）
        """
        if j1 is None or j2 is None:
            return False

        # 四舍五入到小数点后2位
        q1 = round(j1.Q1, decimal)
        q2 = round(j1.Q2, decimal)
        q3 = round(j1.Q3, decimal)
        q4 = round(j1.Q4, decimal)
        q5 = round(j1.Q5, decimal)
        q6 = round(j1.Q6, decimal)

        tq1 = round(j2.Q1, decimal)
        tq2 = round(j2.Q2, decimal)
        tq3 = round(j2.Q3, decimal)
        tq4 = round(j2.Q4, decimal)
        tq5 = round(j2.Q5, decimal)
        tq6 = round(j2.Q6, decimal)

        return (q1 == tq1 and
                q2 == tq2 and
                q3 == tq3 and
                q4 == tq4 and
                q5 == tq5 and
                q6 == tq6)

    def move_linear(self, target_pose: CartesianPose, wait:bool = True,) -> bool:
        """
        直线运动（基于关节运动 + 逆解实现）
        :param target_pose: 目标笛卡尔位姿（单位：米，弧度）
        :return: True=成功下发，False=失败
        """
        if not self.client:
            print("[arm_move_line] 未初始化连接")
            return False

        # 1. 获取当前关节角
        current_joint = self.get_keba_joint_data()
        if current_joint is None:
            print("[arm_move_line] 获取当前关节角失败")
            return False

        # 2. 做 KEBA 专用关节映射（必须！适配C库）
        current_joint_fk = JointPose(
            Q1=current_joint.Q1,
            Q2=current_joint.Q2 - 90.0,
            Q3=-current_joint.Q3,
            Q4=current_joint.Q4 - 90.0,
            Q5=current_joint.Q5,
            Q6=current_joint.Q6
        )

        # 3. 逆解：笛卡尔位姿 → 关节角
        target_joint = self.inverse_kinematics(target_pose, current_joint_fk)
        if target_joint is None:
            print("[arm_move_line] 逆解失败")
            return False

        # 4. 映射回机器人控制器格式
        target_joint_ctrl = JointPose(
            Q1=target_joint.Q1,
            Q2=target_joint.Q2 + 90.0,
            Q3=-target_joint.Q3,
            Q4=target_joint.Q4 + 90.0,
            Q5=target_joint.Q5,
            Q6=target_joint.Q6
        )

        # 5. 调用已有的关节运动执行直线
        # print(f"[arm_move_line] 直线运动目标: {target_pose}")
        ret = self.arm_move_joint(target_joint_ctrl)

        if wait:
            while True:
                if self.is_joint_equal(self.arm_get_current_joint(), target_joint_ctrl):
                    break
        # print('debug ---- ')
        return ret 
    


    def get_keba_joint_data(self) -> Optional[JointPose]:
        """读取当前关节角度"""
        if not self.client:
            print("[get_keba_joint_data] 未初始化连接")
            return None

        with self.lock:
            try:
                result = self.client.read_holding_registers(address = self.REG_JOINTDATA_READ, count = 12)
                if result.isError():
                    print(f"[get_keba_joint_data] 读取失败: {result}")
                    return None
                raw_data = result.registers
            except ModbusException as e:
                print(f"[get_keba_joint_data] 异常: {e}")
                return None        
                


        return self.get_value_from_registers(raw_data)

    def arm_move_force(self, mode: int, displace_target: int = 0) -> bool:
        """力控模式（1:力控，3:带目标位移的力控）"""
        if not self.client:
            print("[arm_move_force] 未初始化连接")
            return False

        if mode == 1:
            with self.lock:
                try:
                    self.client.write_register(self.REG_MODE_WRITE, 1)
                except ModbusException as e:
                    print(f"[arm_move_force] 设置模式1失败: {e}")
                    return False
        elif mode == 3:
            with self.lock:
                try:
                    self.client.write_register(self.REG_MODE3TARGET_WRITE, displace_target)
                except ModbusException as e:
                    print(f"[arm_move_force] 写目标位移失败: {e}")
                    return False
            time.sleep(0.005)
            with self.lock:
                try:
                    self.client.write_register(self.REG_MODE_WRITE, 3)
                except ModbusException as e:
                    print(f"[arm_move_force] 设置模式3失败: {e}")
                    return False
        else:
            print(f"[arm_move_force] 不支持的模式: {mode}")
            return False

        print(f"[arm_move_force] 力控模式{mode}已设置")
        return True

    def get_keba_mode_state(self) -> Optional[int]:
        """读取当前机器人模式状态"""
        if not self.client:
            print("[get_keba_mode_state] 未初始化连接")
            return None

        with self.lock:
            try:
                result = self.client.read_holding_registers(self.REG_MODE_READ, 1)
                if result.isError():
                    print(f"[get_keba_mode_state] 读取失败: {result}")
                    return None
                return result.registers[0]
            except ModbusException as e:
                print(f"[get_keba_mode_state] 异常: {e}")
                return None

    def get_keba_motion_done(self) -> Optional[int]:
        """读取运动完成标志（1:完成，0:运动中）"""
        if not self.client:
            print("[get_keba_motion_done] 未初始化连接")
            return None

        with self.lock:
            try:
                result = self.client.read_holding_registers(self.REG_MOTIONDONE_READ, 1)
                if result.isError():
                    print(f"[get_keba_motion_done] 读取失败: {result}")
                    return None
                return result.registers[0]
            except ModbusException as e:
                print(f"[get_keba_motion_done] 异常: {e}")
                return None

    def arm_get_powerStatus(self) -> Optional[int]:
        """读取电源状态（1:上电，0:断电）"""
        if not self.client:
            print("[arm_get_powerStatus] 未初始化连接")
            return None

        with self.lock:
            try:
                result = self.client.read_holding_registers(self.REG_POWERON_READ, 1)
                if result.isError():
                    print(f"[arm_get_powerStatus] 读取失败: {result}")
                    return None
                return result.registers[0]
            except ModbusException as e:
                print(f"[arm_get_powerStatus] 异常: {e}")
                return None

    def arm_get_motionStatus(self) -> Optional[int]:
        """获取运动状态（1:运动中，0:静止）"""
        return self.get_keba_motion_done()

    def arm_get_current_joint(self) -> Optional[JointPose]:
        """获取当前关节角度（同 get_keba_joint_data）"""
        return self.get_keba_joint_data()

    def close(self):
        """关闭 Modbus 连接"""
        if self.client:
            self.client.close()
            self.client = None
            print("[close] Modbus 连接已关闭")
    
        # ==================== 齐次矩阵 / 位姿变换工具（集成到 KebaRobot 类） ====================
    def pose_to_homogeneous_matrix(self, pose: CartesianPose, degrees=True) -> np.ndarray:
        """位姿 → 4x4齐次矩阵"""
        # ZYX 内旋 RPY：yaw(Z) → pitch(Y) → roll(X)
        euler_angles = [pose.yaw, pose.pitch, pose.roll]
        r = R.from_euler('ZYX', euler_angles, degrees=degrees)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = [pose.x, pose.y, pose.z]
        return T

    def homogeneous_matrix_to_pose(self, matrix: np.ndarray, degrees=True) -> CartesianPose:
        """4x4齐次矩阵 → 笛卡尔位姿"""
        if matrix.shape != (4, 4):
            raise ValueError(f"矩阵必须是4x4，当前：{matrix.shape}")
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        rot_mat = matrix[:3, :3]
        r = R.from_matrix(rot_mat)
        yaw, pitch, roll = r.as_euler('ZYX', degrees=degrees)
        return CartesianPose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
    
    def zyx_homogeneous_matrix_to_pose(self, matrix: np.ndarray, degrees=True) -> CartesianPose:
        """4x4齐次矩阵 → 笛卡尔位姿"""
        if matrix.shape != (4, 4):
            raise ValueError(f"矩阵必须是4x4，当前：{matrix.shape}")
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        rot_mat = matrix[:3, :3]
        r = R.from_matrix(rot_mat)
        yaw, pitch, roll = r.as_euler('ZXZ', degrees=degrees)
        # angles_zxz = r.as_euler('zxz', degrees=degrees)
        return CartesianPose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
    
    #
    def get_tcp_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取当前 TCP 位姿（纯关节角正解实现）
        内部直接完成 KEBA 映射：
        Q2 -= 90, Q3 = -Q3, Q4 -= 90
        返回格式：(x_m, y_m, z_m, rx_deg, ry_deg, rz_deg)
        """
        # if not self._connected:
        #     logging.debug("KEBA 未连接，无法获取 TCP 位姿")
        #     return None

        # 1. 读取关节角
        joint = self.arm_get_current_joint()
        if joint is None:
            return None

        # ===================== 在这里直接实现关节映射 =====================
        # 直接按照 KEBA C 库要求修正角度，不调用任何外部方法
        q1 = joint.Q1
        q2 = joint.Q2 - 90.0   # Q2 -= 90
        q3 = -joint.Q3         # Q3 = -Q3
        q4 = joint.Q4 - 90.0   # Q4 -= 90
        q5 = joint.Q5
        q6 = joint.Q6
        # =================================================================

        # 2. 调用正运动学 C 库
        try:
            T = np.zeros((4, 4), dtype=np.float64)
            pose = self.kin_lib.JnttoZYX(q1, q2, q3, q4, q5, q6, T)
        except Exception as exc:
            logging.error("正运动学求解失败: %s", exc)
            return None

        # 3. 返回结果：米、度，ZYX 内旋 RPY
        return (
            round(float(pose.X), 6),
            round(float(pose.Y), 6),
            round(float(pose.Z), 6),
            round(float(pose.roll), 6),
            round(float(pose.pitch), 6),
            round(float(pose.yaw), 6),
        )


    # ==================== 核心：相对运动接口 ====================
    def move_relative_base(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        drx: float = 0.0,
        dry: float = 0.0,
        drz: float = 0.0,
        wait: bool = True,
        timeout: float = 30.0
    ) -> int:
        """
        【相对法兰/基坐标系】增量运动（世界系）
        :param dx, dy, dz: 基系下位移增量，单位 mm
        :param drx, dry, drz: 基系下旋转增量，单位 度
        :param wait: 是否等待运动完成
        :param timeout: 等待超时时间
        :return: 0=成功, -1=失败, -2=超时
        """
        # 1. 获取当前TCP位姿
        current_pose_tuple = self.get_tcp_pose()
        if current_pose_tuple is None:
            logging.error("move_relative_base: 获取当前TCP位姿失败")
            return -1
        x, y, z, rx, ry, rz = current_pose_tuple
        current_pose = CartesianPose(x, y, z, rx, ry, rz)

        # 2. 基系直接叠加增量（mm → m）
        target_pose = CartesianPose(
            x=current_pose.x + dx / 1000.0,
            y=current_pose.y + dy / 1000.0,
            z=current_pose.z + dz / 1000.0,
            roll=current_pose.roll + drx,
            pitch=current_pose.pitch + dry,
            yaw=current_pose.yaw + drz
        )

        logging.info(
            f"基系相对运动 | dx={dx:.1f}mm dy={dy:.1f}mm dz={dz:.1f}mm | "
            f"drx={drx:.1f}° dry={dry:.1f}° drz={drz:.1f}°"
        )

        # 3. 执行直线运动
        ret = self.move_linear(target_pose)
        if ret != 0:
            logging.error(f"move_relative_base: 运动启动失败，错误码={ret}")
            return -1

        # 4. 等待完成
        # if wait:
        #     success = self._wait_until_target_reached(target_pose, timeout)
        #     if not success:
        #         logging.error(f"move_relative_base: 运动超时({timeout}s)")
        #         return -2
        #     logging.info("move_relative_base: 运动完成")
        if wait:
             while True:
                if self.is_joint_equal(self.arm_get_current_joint(), target_pose):
                    break
        # print('debug ---- ')
        return 0

    def move_relative_tool(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        drx: float = 0.0,
        dry: float = 0.0,
        drz: float = 0.0,
        wait: bool = True,
        timeout: float = 30.0
    ) -> int:
        """
        【相对工具/TCP坐标系】增量运动（末端系）
        :param dx, dy, dz: 工具系下位移增量，单位 mm
        :param drx, dry, drz: 工具系下旋转增量，单位 度
        :param wait: 是否等待运动完成
        :param timeout: 等待超时时间
        :return: 0=成功, -1=失败, -2=超时
        """
        # 1. 获取当前TCP位姿
        current_pose_tuple = self.get_tcp_pose()
        if current_pose_tuple is None:
            logging.error("move_relative_tool: 获取当前TCP位姿失败")
            return -1
        x, y, z, rx, ry, rz = current_pose_tuple
        current_pose = CartesianPose(x, y, z, rx, ry, rz)

        # 2. 转换为齐次矩阵
        T_current = self.pose_to_homogeneous_matrix(current_pose)

        # 3. 工具系增量变换（mm → m）
        delta_pose = CartesianPose(
            x=dx / 1000.0,
            y=dy / 1000.0,
            z=dz / 1000.0,
            roll=drx,
            pitch=dry,
            yaw=drz
        )
        T_delta = self.pose_to_homogeneous_matrix(delta_pose)

        # 4. 计算目标位姿：T_target = T_current @ T_delta（工具系相对运动）
        T_target = T_current @ T_delta
        target_pose = self.homogeneous_matrix_to_pose(T_target)

        logging.info(
            f"工具系相对运动 | dx={dx:.1f}mm dy={dy:.1f}mm dz={dz:.1f}mm | "
            f"drx={drx:.1f}° dry={dry:.1f}° drz={drz:.1f}°"
        )

        # 5. 执行直线运动
        ret = self.move_linear(target_pose, wait)
        # if ret != 0:
        #     logging.error(f"move_relative_tool: 运动启动失败，错误码={ret}")
        #     return -1

        # 6. 等待完成
        # if wait:
        #     success = self._wait_until_target_reached(target_pose, timeout)
        #     if not success:
        #         logging.error(f"move_relative_tool: 运动超时({timeout}s)")
        #         return -2
        #     logging.info("move_relative_tool: 运动完成")
        return 0
    
    # ==================== 新增：相对轨迹运动接口 ====================
    def compute_relative_transform(self, base_pose: CartesianPose, target_pose: CartesianPose) -> np.ndarray:
        """
        计算相对变换矩阵 T，使得 target_matrix = base_matrix @ T
        :param base_pose: 基准位姿
        :param target_pose: 目标位姿
        :return: 4x4 齐次变换矩阵 T
        """
        T_base = self.pose_to_homogeneous_matrix(base_pose, degrees=True)
        T_target = self.pose_to_homogeneous_matrix(target_pose, degrees=True)
        T_rel = np.linalg.inv(T_base) @ T_target
        return T_rel

    def run_relative_trajectory(
        self,
        base_pose: CartesianPose,
        target_poses: list,
        ref_pose: CartesianPose = None,
        motion_type: str = "linear",
        ):
        """
        执行相对轨迹运动（已修复 KEBA 关节映射问题）
        :param base_pose: 基准位姿
        :param target_poses: 目标位姿列表
        :param ref_pose: 参考位姿（默认当前TCP）
        :param motion_type: linear / joint
        """
        # 1. 获取参考位姿
        if ref_pose is None:
            pose_tuple = self.get_tcp_pose()
            if not pose_tuple:
                raise RuntimeError("获取当前位姿失败")
            ref_pose = CartesianPose(*pose_tuple)

        # 2. 计算所有相对变换
        rel_transforms = []
        for pose in target_poses:
            t_rel = self.compute_relative_transform(base_pose, pose)
            rel_transforms.append(t_rel)

        # 3. 参考矩阵
        ref_mat = self.pose_to_homogeneous_matrix(ref_pose, degrees=False)
        joint_path = []

        # 4. 遍历轨迹点
        for idx, t_rel in enumerate(rel_transforms):
            target_mat = ref_mat @ t_rel
            target_pose = self.homogeneous_matrix_to_pose(target_mat, degrees=False)

            if motion_type == "linear":
                print(f"[相对轨迹] 直线运动 → 点 {idx+1}")
                self.move_linear(target_pose)

            elif motion_type == "joint":
                # ==============================================
                # 获取当前关节角 + KEBA 映射
                # ==============================================
                # current_joints = self.arm_get_current_joint()
                current_joints  = JointPose(Q1=-70.6330, Q2=-39.3460, Q3=116.9291, Q4=135.6958, Q5=76.0021, Q6=-86.3680)

                # 必须映射！！！
                current_joints_fk = JointPose(
                    Q1=current_joints.Q1,
                    Q2=current_joints.Q2 - 90.0,
                    Q3=-current_joints.Q3,
                    Q4=current_joints.Q4 - 90.0,
                    Q5=current_joints.Q5,
                    Q6=current_joints.Q6
                )

                #  逆解（使用映射后的关节角）
                target_joint = self.inverse_kinematics(target_pose, current_joints_fk)
                if target_joint is None:
                    raise RuntimeError(f"逆解失败 → 点 {idx+1}")

                # ==============================================
                #  【关键修复】逆解完映射回控制器格式
                # ==============================================
                target_joint_ctrl = JointPose(
                    Q1=target_joint.Q1,
                    Q2=target_joint.Q2 + 90.0,
                    Q3=-target_joint.Q3,
                    Q4=target_joint.Q4 + 90.0,
                    Q5=target_joint.Q5,
                    Q6=target_joint.Q6
                )

                joint_path.append(target_joint_ctrl)

        # 5. 连续关节轨迹
        if motion_type == "joint" and len(joint_path) > 0:
            print("[相对轨迹] 执行连续关节轨迹")
            for jpos in joint_path:
                # self.arm_move_joint(jpos)
                print('++++++++解算结果：',jpos)
            print("[相对轨迹] 连续轨迹完成")

    # ==============================
    # ✅ 封装：世界坐标 → 关节角（自动处理 KEBA 映射，外部零感知）
    # ==============================
    def world_pose_to_joint(self, target_cartesian: CartesianPose) -> Optional[JointPose]:
        """
        外部调用专用：
        输入：世界坐标系笛卡尔位姿（单位：度 + 米）
        输出：机器人可直接使用的关节角 JointPose
        内部自动处理 KEBA Q2-90 / Q3取反 / Q4-90 映射
        """
        if not self.kin_lib:
            print("[world_pose_to_joint] 运动学库未加载")
            return None

        # 1. 获取当前机器人关节角
        current_joint_ctrl = self.arm_get_current_joint()
        if current_joint_ctrl is None:
            print("[world_pose_to_joint] 获取当前关节角失败")
            return None

        # ========================
        # ✅ 内部 KEBA 映射（对外隐藏）
        # ========================
        current_joint_kin = JointPose(
            Q1=current_joint_ctrl.Q1,
            Q2=current_joint_ctrl.Q2 - 90.0,
            Q3=-current_joint_ctrl.Q3,
            Q4=current_joint_ctrl.Q4 - 90.0,
            Q5=current_joint_ctrl.Q5,
            Q6=current_joint_ctrl.Q6
        )
        # current_joint_kin = JointPose(
        #     Q1=8.383,
        #     Q2=-34.750 - 90.0,
        #     Q3=-92.476,
        #     Q4=-11.557 - 90.0,
        #     Q5=24.334,
        #     Q6=43.124
        # )

        # 2. 逆解求解
        target_joint_kin = self.inverse_kinematics(target_cartesian, current_joint_kin)
        if target_joint_kin is None:
            print("[world_pose_to_joint] 逆解失败")
            return None

        # ========================
        # ✅ 逆解结果映射回机器人格式
        # ========================
        target_joint_ctrl = JointPose(
            Q1=target_joint_kin.Q1,
            Q2=target_joint_kin.Q2 + 90.0,
            Q3=-target_joint_kin.Q3,
            Q4=target_joint_kin.Q4 + 90.0,
            Q5=target_joint_kin.Q5,
            Q6=target_joint_kin.Q6
        )

        return target_joint_ctrl
    
    def get_distance_between_poses(self, pose1: CartesianPose, pose2: CartesianPose) -> float:
        """
        计算两个笛卡尔位姿的 3D 空间直线距离
        :param pose1: 位姿1 (CartesianPose)
        :param pose2: 位姿2 (CartesianPose)
        :return: 空间直线距离，单位：米
        """
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dz = pose2.z - pose1.z
        
        # 欧几里得距离公式
        distance_m = np.sqrt(dx**2 + dy**2 + dz**2)
        return distance_m

    def get_distance_mm(self, pose1: CartesianPose, pose2: CartesianPose) -> float:
        """
        计算空间距离，返回：毫米
        """
        print("距离：", self.get_distance_between_poses(pose1, pose2) * 1000.0)
        return self.get_distance_between_poses(pose1, pose2) * 1000.0







def main():
    # 去掉第一个元素（脚本名），得到参数列表
    args = sys.argv[1:]
    print("接收到的参数:", args)
    # 按需处理参数
    if len(args) > 0:
        print("第一个参数:", args[0])
        robot = RobotInterface(kin_lib_path="/home/nvidia/Downloads/HD/HD_0323/KEBA/RobotControlkeba.so")
    # 初始化连接（假设机器人 IP 为 192.168.1.100，端口 502）
        if robot.init_robot("192.168.1.133", 502):
            # 读取当前关节角度
            

            # 移动到目标角度
            # target = JointPose(Q1=0.1, Q2=0.2, Q3=0.3, Q4=0.4, Q5=0.5, Q6=0.6)
            # robot.arm_move_joint(target)


            if args[0] == "power_off":
                robot.arm_PowerOff()
            elif args[0] == "power_on":
                robot.arm_PowerON()
            elif args[0] == "mode_1": # 插枪
                robot.arm_move_force(1)
            elif args[0] == "mode_3":
                robot.arm_move_force(3, 70) # 拔枪
            elif args[0] == "control":
                # 关节角->位姿 
                # current_joint_new = robot.arm_get_current_joint()
                # # current_joint_new.Q1 += 2
                # current_joint_new.Q3 += 2
                # robot.arm_move_joint(JointPose(-87.5003, -18.0431, 141.3562, 139.4387, 91.8385, -92.2225))
                robot.move_relative_tool(dz=20)
                # robot.move_relative_tool(dy= 30)
                time.sleep(5)

                current_joint_new = robot.arm_get_current_joint()
                print("current_joint_new: ",current_joint_new)

                # current_joint = JointPose(Q1=-44.383205, Q2=-9.097911, Q3=141.924088, Q4=118.530670, Q5=59.957478, Q6=-67.268784)
                current_joint = JointPose(Q1=-83.9327, Q2=-23.9139, Q3=136.0814, Q4=140.0463, Q5=88.4855, Q6=-91.0072)
                print("当前关节角：", current_joint)
                current_joint.Q2 -= 90
                current_joint.Q3 = -current_joint.Q3
                current_joint.Q4 -= 90

                # 正运动学：计算当前位姿
                # cartesian = robot.forward_kinematics(current_joint) 
                cartesian = CartesianPose(x=-0.135861, y=-0.600643, z=0.284965, roll=175.7176, pitch=-70.0001, yaw=-80.9877)
                # cartesian = CartesianPose(369.938,-532.621,172.044,-171.8988,-70.8216,-92.8269)
                # cartesian = CartesianPose(369.938,-522.621,182.044,-171.8988,-70.8216,-92.8269)
                # cartesian = CartesianPose(-328.8826919628133, -259.9668874231, 489.62248492191634, -173.11256336193946, -68.93333123664308, -87.13262339367421)
                                        # -208.45962104426187, -482.5212093449478, 810.1805462287849
                # cartesian.x /= 1000
                # cartesian.y /= 1000
                # cartesian.z /= 1000

                if cartesian:
                    print("当前笛卡尔位姿:", cartesian)
                end_pose = [
                [190.122973, -535.226942, 125.527981, 156.745490, -78.082289, -58.330155],
                [382.647933, -505.202421, 109.582415, 156.735788, -78.079192,-58.321151],
                [-138.192211, -586.441301, 152.706986, 156.750615, -78.079168, -58.334033],
                [173.154770, -561.710331, 249.060161, 156.736607, -78.077408, -58.320798],
                [-241.101406, -624.874170, 274.989218, 31.697928, -81.079001, 67.479481],
                [-241.090721, -624.771663, 274.989502, -98.042803, -77.752799, -173.243464],
                [-178.881840, -620.451700, 89.855097, -111.937743, -77.755310, -173.242765],
                [517.766611, -683.821848, 296.631774, 175.563321, -72.177883, -88.355032],
                [517.962932, -683.959329, 296.557395, 140.975430, -87.892007, -22.955146],
                [624.244556, -663.016697, -214.082843, 83.739578, -52.017316, 11.890384],
                [609.741950, -543.948602, -222.133760, 83.737590, -52.018487, 11.891428],
                [556.347820, -541.608249, -110.343083, 46.667443, -84.598241, 39.580658],
                [241.490457, -681.527467, 52.832948, 90.282223, -82.979847, 1.512484],
                [649.134591, -668.871301, 2.720495, 20.305316, -69.387702, 72.694714],
                [614.600482, -670.287804, 7.056826, 129.509660, -80.473203, -37.201948],
                [179.362731, -667.770836, -154.277406, 129.523395, -80.476798, -37.217532],
                [243.350933, -716.383753, 324.907600, 129.521907, -80.478570, -37.216666],
                [-264.664781, -715.704717, 157.749507, 129.546048, -80.475841, -37.239403],
                [-192.711162, -803.551658, 358.433436, 162.276920, -65.221876, -71.119305],
                [759.855292, -760.850117, 254.856736, 113.630394, -63.457045, -20.020562],
                [618.833195, -724.337047, -227.073824, 60.218405, -77.586287, 60.071835],
                [729.414524, -751.953522, -259.624388, 94.534632, -74.024000, -18.74497],
                [-511.114743, -964.720172, -200.665125, -131.762140, -64.909608, -132.747146],
                # [-483.348309, -1086.672777, 25.140930, 171.515496, -76.487026, -72.473048]
                ]
                result = []
                for pose in end_pose:
                    matrix = robot.pose_to_homogeneous_matrix(CartesianPose(pose[0] / 1000,
                                                                          pose[1] / 1000,
                                                                          pose[2] / 1000, 
                                                                          pose[3],
                                                                          pose[4],
                                                                          pose[5]))
                    cover_pose = robot.zyx_homogeneous_matrix_to_pose(matrix)
                    result.append([float(cover_pose.x) * 1000, float(cover_pose.y) * 1000, float(cover_pose.z) * 1000, float(cover_pose.roll), float(cover_pose.pitch), float(cover_pose.yaw)])
                # print(result)
                
                # print("粗定位关节角位姿")
                # print(joint_cover)
                # robot.arm_move_joint(joint_cover)
                
                # with open("/home/nvidia/Downloads/HD/HD_0323/KEBA/actuator/log.txt",mode='+a') as f:
                #     f.write(str(cartesian)+'\n')
                #     f.write(str(current_joint_copy)+ '\n')
                #     f.write('\n')
                #     print('---debug---')
                #     f.close()
                    

                # # 逆运动学：给定目标位姿，计算关节角
                target_pose = cartesian
                target_joint = robot.inverse_kinematics(target_pose, current_joint)
                

                target_joint.Q2 += 90
                target_joint.Q3 = -target_joint.Q3
                target_joint.Q4 += 90

                if target_joint:
                    print("逆解关节角:", target_joint)

                
                # # # 运动到逆解角度
                # # # joint_pose = JointPose(-24.207, -54.139, 105.966, 125.691, 28.069, -58.998)
                # # target_joint.Q1 += 3

                # joint_pose  = target_joint

                # 关节运动
                # target_joint = JointPose(Q1=-70.6330, Q2=-39.3460, Q3=116.9291, Q4=135.6958, Q5=76.0021, Q6=-86.3680)
                # robot.arm_move_joint(target_joint)

                # 法兰坐标系运动
                # robot.move_relative_tool(dz=40)

                # current_joint_2 = robot.arm_get_current_joint()
                # print("当前关节角:", current_joint_2)

                robot.get_distance_mm(CartesianPose(x=0.350576, y=-0.686296, z=0.248208, roll=-170.456977, pitch=-61.889503, yaw=-89.599367),
                                       CartesianPose(x=0.337331, y=-0.619505, z=0.283945, roll=-170.458640, pitch=-61.889374, yaw=-89.599150) )

                robot.close()
            else:
                print("连接失败")
if __name__ == "__main__":
    main()


    # robot = RobotInterface(kin_lib_path="/home/nvidia/KEBA/RobotControlkeba.so")

    # # 初始化 Modbus 连接
    # if robot.init_robot("192.168.1.133", 502):

    #     robot.arm_PowerON()
    #     time.sleep(5)
    #     # 读取当前关节角
    #     current_joint = robot.arm_get_current_joint()
    #     print("当前关节角:", current_joint)

    #     current_joint.Q2 -= 90
    #     current_joint.Q3 = -current_joint.Q3
    #     current_joint.Q4 -= 90

    #     # 正运动学：计算当前位姿
    #     cartesian = robot.forward_kinematics(current_joint)
    #     if cartesian:
    #         print("当前笛卡尔位姿:", cartesian)

    #     # 逆运动学：给定目标位姿，计算关节角
    #     target_pose = cartesian
    #     target_joint = robot.inverse_kinematics(target_pose, current_joint)

    #     target_joint.Q2 += 90
    #     target_joint.Q3 = -target_joint.Q3
    #     target_joint.Q4 += 90

    #     if target_joint:
    #         print("逆解关节角:", target_joint)

    #     # 运动到逆解角度
    #     joint_pose = JointPose(-24.207, -54.139, 105.966, 125.691, 28.069, -58.998)

    #     robot.arm_move_joint(joint_pose)
    #     current_joint_2 = robot.arm_get_current_joint()
    #     print("当前关节角:", current_joint_2)


    #     robot.close()
