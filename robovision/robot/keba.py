"""
KEBA 机械臂接口 | KEBA Robot Interface

封装 KEBA 机械臂的 Modbus + 正运动学库，提供与 RobotBase 统一的接口：
- 连接 / 断开管理
- 获取 TCP 位姿
- get_tcp_pose() 返回格式严格遵循 RobotBase:
    (x_m, y_m, z_m, rx_deg, ry_deg, rz_deg)
    单位：米、度
    欧拉角：ZYX 内旋（RPY），Base←TCP
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import struct
import threading
import time
from typing import List, Optional, Tuple

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

from robovision.robot.base import RobotBase

logger = logging.getLogger(__name__)

class JointPose:
    """关节角数据，单位：度。"""

    def __init__(
        self,
        Q1: float = 0.0,
        Q2: float = 0.0,
        Q3: float = 0.0,
        Q4: float = 0.0,
        Q5: float = 0.0,
        Q6: float = 0.0,
    ):
        self.Q1 = float(Q1)
        self.Q2 = float(Q2)
        self.Q3 = float(Q3)
        self.Q4 = float(Q4)
        self.Q5 = float(Q5)
        self.Q6 = float(Q6)

    def to_list(self) -> List[float]:
        return [self.Q1, self.Q2, self.Q3, self.Q4, self.Q5, self.Q6]

    def __repr__(self) -> str:
        return (
            "JointPose("
            f"Q1={self.Q1:.6f}, Q2={self.Q2:.6f}, Q3={self.Q3:.6f}, "
            f"Q4={self.Q4:.6f}, Q5={self.Q5:.6f}, Q6={self.Q6:.6f})"
        )


class CartesianPose:
    """
    笛卡尔位姿。
    - x/y/z: 米
    - roll/pitch/yaw: 度，ZYX 内旋欧拉角中的 X/Y/Z 轴角（RPY）
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def __repr__(self) -> str:
        return (
            "CartesianPose("
            f"x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f}, "
            f"roll={self.roll:.6f}, pitch={self.pitch:.6f}, yaw={self.yaw:.6f})"
        )

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


class KebaRobot(RobotBase):
    """
    KEBA 机械臂接口。

    设计目标：
    1. 对外只暴露 RobotBase 统一接口
    2. 当前重点实现 get_tcp_pose()
    3. TCP 位姿来源：
       读取关节角 -> 正运动学库 JnttoZYX -> 输出 Base←TCP 位姿
    """

    # Modbus 寄存器地址
    REG_POWERON_WRITE = 2
    REG_START_WRITE = 3
    REG_MOTIONDONE_READ = 6
    REG_MODE_WRITE = 7
    REG_MODE3TARGET_WRITE = 10000
    REG_JOINTDATA_READ = 12288
    REG_JOINTDATA_WRITE = 15200
    REG_JOINTDATASENDSTATE_WRITE = 15212
    REG_POWERON_READ = 4
    ARRIVAL_TRANS_TOL_M = 0.002
    ARRIVAL_ROT_TOL_DEG = 1.0
    POWER_ON_SETTLE_SEC = 0.3

    def __init__(
        self,
        ip: str,
        port: int = 502,
        kin_lib_path: str = "/home/nvidia/SWG/ToCamera/RobotControlkeba.so",
        timeout: float = 5.0,
        apply_joint_mapping: bool = True,
    ):
        self._ip = ip
        self._port = int(port)
        self._timeout = float(timeout)
        self._apply_joint_mapping = bool(apply_joint_mapping)

        if not os.path.isabs(kin_lib_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            kin_lib_path = os.path.join(project_root, kin_lib_path)
        self._kin_lib_path = kin_lib_path

        self._client: Optional[ModbusTcpClient] = None
        self._connected = False
        self._lock = threading.Lock()

        self._kin_lib = None
        self._EulerAngles2 = None
        self._JntAngle = None

        self._load_kinematics_library()

    # ==================== SDK / Kinematics ====================

    def _load_kinematics_library(self) -> None:
        """加载正运动学 C 共享库。"""
        try:
            self._kin_lib = ctypes.CDLL(self._kin_lib_path)
            self._setup_kin_lib()
            logger.info("KEBA 运动学库加载成功: %s", self._kin_lib_path)
        except Exception as exc:
            self._kin_lib = None
            logger.error("KEBA 运动学库加载失败: %s", exc)

    def _setup_kin_lib(self) -> None:
        """配置 ctypes 函数参数与返回类型。"""

        class EulerAngles2(ctypes.Structure):
            _fields_ = [
                ("X", ctypes.c_double),
                ("Y", ctypes.c_double),
                ("Z", ctypes.c_double),
                ("roll", ctypes.c_double),
                ("pitch", ctypes.c_double),
                ("yaw", ctypes.c_double),
            ]

        class JntAngle(ctypes.Structure):
            _fields_ = [
                ("J0", ctypes.c_double),
                ("J1", ctypes.c_double),
                ("J2", ctypes.c_double),
                ("J3", ctypes.c_double),
                ("J4", ctypes.c_double),
                ("J5", ctypes.c_double),
            ]

        self._EulerAngles2 = EulerAngles2
        self._JntAngle = JntAngle

        self._kin_lib.JnttoZYX.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._kin_lib.JnttoZYX.restype = EulerAngles2

        self._kin_lib.ZYXtoJntAngle.argtypes = [
            EulerAngles2,
            JntAngle,
        ]
        self._kin_lib.ZYXtoJntAngle.restype = JntAngle

    def _forward_kinematics(self, joint_pose: JointPose) -> Optional[CartesianPose]:
        """
        正运动学：关节角 -> 笛卡尔位姿

        返回：
            CartesianPose(x/y/z 为米, roll/pitch/yaw 为度)
        """
        if self._kin_lib is None:
            logger.error("KEBA 运动学库未加载，无法计算正运动学")
            return None

        try:
            pose = self._kin_lib.JnttoZYX(
                joint_pose.Q1,
                joint_pose.Q2,
                joint_pose.Q3,
                joint_pose.Q4,
                joint_pose.Q5,
                joint_pose.Q6,
            )
            return CartesianPose(
                x=pose.X,
                y=pose.Y,
                z=pose.Z,
                roll=pose.roll,
                pitch=pose.pitch,
                yaw=pose.yaw,
            )
        except Exception as exc:
            logger.error("KEBA 正运动学调用失败: %s", exc)
            return None

    def _inverse_kinematics(
        self,
        target_pose: CartesianPose,
        current_joint: Optional[JointPose] = None,
    ) -> Optional[JointPose]:
        """逆运动学：笛卡尔位姿（米/度） -> 运动学关节角（度）。"""
        if self._kin_lib is None or self._EulerAngles2 is None or self._JntAngle is None:
            logger.error("KEBA 运动学库未加载，无法计算逆运动学")
            return None

        if current_joint is None:
            current_joint = self._read_joint_pose()
            if current_joint is None:
                logger.warning("KEBA 无法读取当前关节角，逆解失败")
                return None
            current_joint = self._map_joint_for_fk(current_joint)

        target = self._EulerAngles2()
        target.X = target_pose.x
        target.Y = target_pose.y
        target.Z = target_pose.z
        target.roll = target_pose.roll
        target.pitch = target_pose.pitch
        target.yaw = target_pose.yaw

        curr = self._JntAngle()
        curr.J0 = current_joint.Q1
        curr.J1 = current_joint.Q2
        curr.J2 = current_joint.Q3
        curr.J3 = current_joint.Q4
        curr.J4 = current_joint.Q5
        curr.J5 = current_joint.Q6

        try:
            result = self._kin_lib.ZYXtoJntAngle(target, curr)
            return JointPose(
                Q1=result.J0,
                Q2=result.J1,
                Q3=result.J2,
                Q4=result.J3,
                Q5=result.J4,
                Q6=result.J5,
            )
        except Exception as exc:
            logger.error("KEBA 逆运动学调用失败: %s", exc)
            return None

    # ==================== Modbus helpers ====================

    @staticmethod
    def _registers_to_float(high: int, low: int) -> float:
        """两个 16-bit 寄存器 -> float32（大端）。"""
        packed = bytes(
            [
                (high >> 8) & 0xFF,
                high & 0xFF,
                (low >> 8) & 0xFF,
                low & 0xFF,
            ]
        )
        return struct.unpack(">f", packed)[0]

    def _joint_from_registers(self, raw_data: List[int]) -> JointPose:
        if len(raw_data) < 12:
            raise ValueError(f"关节寄存器数据长度不足 12，实际为 {len(raw_data)}")
        return JointPose(
            Q1=self._registers_to_float(raw_data[0], raw_data[1]),
            Q2=self._registers_to_float(raw_data[2], raw_data[3]),
            Q3=self._registers_to_float(raw_data[4], raw_data[5]),
            Q4=self._registers_to_float(raw_data[6], raw_data[7]),
            Q5=self._registers_to_float(raw_data[8], raw_data[9]),
            Q6=self._registers_to_float(raw_data[10], raw_data[11]),
        )

    @staticmethod
    def _float_to_registers(value: float) -> Tuple[int, int]:
        """float32 -> 两个 16-bit 寄存器（大端）。"""
        packed = struct.pack(">f", float(value))
        high = (packed[0] << 8) | packed[1]
        low = (packed[2] << 8) | packed[3]
        return high, low

    @classmethod
    def _joint_to_registers(cls, joint: JointPose) -> List[int]:
        registers = []
        for value in joint.to_list():
            registers.extend(cls._float_to_registers(value))
        return registers

    def _ensure_connection(self) -> bool:
        """确保连接有效，断线时尝试重连。"""
        if self._connected and self._client is not None:
            try:
                # 发送心跳请求测试连接
                result = self._client.read_holding_registers(
                    address=self.REG_POWERON_READ, count=1, slave=1
                )
                if not result.isError():
                    return True
            except Exception:
                pass

        # 连接失效，尝试重连
        logger.warning("KEBA 连接失效，尝试重连...")
        self.disconnect()
        return self.connect()

    def _read_joint_pose(self) -> Optional[JointPose]:
        """读取当前 6 轴关节角，单位：度，带断线重连。"""
        for attempt in range(2):  # 首次尝试 + 重连一次
            if not self._connected and attempt > 0:
                if not self._ensure_connection():
                    return None

            if not self._connected or self._client is None:
                return None

            with self._lock:
                try:
                    result = self._client.read_holding_registers(
                        address=self.REG_JOINTDATA_READ,
                        count=12,
                    )
                except (ConnectionError, OSError, ModbusException) as exc:
                    if attempt == 0:
                        logger.debug("KEBA 读取关节寄存器失败（尝试重连）: %s", exc)
                        continue  # 尝试重连
                    logger.warning("KEBA 读取关节寄存器失败: %s", exc)
                    return None
                except Exception as exc:
                    if attempt == 0:
                        logger.debug("KEBA 读取关节寄存器异常（尝试重连）: %s", exc)
                        continue  # 尝试重连
                    logger.warning("KEBA 读取关节寄存器异常: %s", exc)
                    return None

            if result.isError():
                if attempt == 0:
                    logger.debug("KEBA 读取关节寄存器错误（尝试重连）: %s", result)
                    # 尝试重连
                    if not self._ensure_connection():
                        return None
                    continue
                logger.warning("KEBA 读取关节寄存器错误: %s", result)
                return None

            try:
                return self._joint_from_registers(result.registers)
            except Exception as exc:
                logger.warning("KEBA 解析关节寄存器失败: %s", exc)
                return None

        return None

    # ==================== Joint mapping ====================

    def _map_joint_for_fk(self, joint: JointPose) -> JointPose:
        """
        将控制器读出的关节角映射到正运动学库使用的关节定义。

        KEBA C 源码中 JnttoZYX 会把入参按“度”转为弧度，因此这里保留参考
        实现中的固定补偿：Q2 -= 90, Q3 = -Q3, Q4 -= 90。
        """
        if not self._apply_joint_mapping:
            return JointPose(*joint.to_list())

        return JointPose(
            Q1=joint.Q1,
            Q2=joint.Q2 - 90.0,
            Q3=-joint.Q3,
            Q4=joint.Q4 - 90.0,
            Q5=joint.Q5,
            Q6=joint.Q6,
        )

    def _unmap_joint_from_fk(self, joint: JointPose) -> JointPose:
        """将运动学库输出关节角映射回控制器关节定义。"""
        if not self._apply_joint_mapping:
            return JointPose(*joint.to_list())

        return JointPose(
            Q1=joint.Q1,
            Q2=joint.Q2 + 90.0,
            Q3=-joint.Q3,
            Q4=joint.Q4 + 90.0,
            Q5=joint.Q5,
            Q6=joint.Q6,
        )

    # ==================== RobotBase API overrides ====================

    def set_speed(self, speed_pct: int) -> int:
        """KEBA 不支持速度设置。"""
        logger.debug("KEBA 不支持速度设置")
        return -1  # 返回非 0 表示失败/不支持

    def set_tool_id(self, tool_id: int) -> int:
        """KEBA 不支持工具坐标系切换。"""
        logger.debug("KEBA 不支持工具坐标系切换")
        return -1  # 返回非 0 表示失败/不支持

    def get_tool_id(self) -> Optional[int]:
        """KEBA 不支持工具坐标系读取。"""
        return None  # 保持 None（已正确）

    # ==================== RobotBase required APIs ====================

    def connect(self) -> bool:
        """连接机械臂。"""
        try:
            client = ModbusTcpClient(
                host=self._ip,
                port=self._port,
                timeout=self._timeout,
            )
            ok = client.connect()
            if not ok:
                logger.warning("KEBA 机械臂连接失败: %s:%s", self._ip, self._port)
                self._client = None
                self._connected = False
                return False

            self._client = client
            self._connected = True
            logger.info("KEBA 机械臂连接成功: %s:%s", self._ip, self._port)
            return True
        except Exception as exc:
            self._client = None
            self._connected = False
            logger.error("KEBA 机械臂连接异常: %s", exc)
            return False

    def disconnect(self) -> None:
        """断开机械臂连接。"""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("KEBA 机械臂已断开连接")
            except Exception:
                pass
        self._client = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """是否已连接。"""
        return self._connected

    def get_tcp_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取当前 TCP 位姿。

        Returns:
            (x_m, y_m, z_m, rx_deg, ry_deg, rz_deg) 或 None（失败时）
            单位：米、度，ZYX 内旋欧拉角（RPY），Base←TCP
        """
        if not self._connected:
            logger.debug("KEBA 未连接，无法获取 TCP 位姿")
            return None

        joint = self._read_joint_pose()
        if joint is None:
            return None

        joint_for_fk = self._map_joint_for_fk(joint)
        pose = self._forward_kinematics(joint_for_fk)
        if pose is None:
            return None

        return (
            float(pose.x),  # C 库返回的就是米，直接返回
            float(pose.y),
            float(pose.z),
            float(pose.roll),
            float(pose.pitch),
            float(pose.yaw),
        )

    def move_linear(self, target) -> int:
        """非阻塞直线运动，target 使用脚本约定的 mm/deg。"""
        if not self._connected or self._client is None:
            logger.warning("KEBA 未连接，无法执行运动")
            return -1

        # 检查电源状态，未上电则自动上电
        power_status = self.get_power_status()
        if power_status == 0:  # 未上电
            logger.info("KEBA 未上电，自动上电...")
            if not self.power_on():
                logger.warning("KEBA 上电失败，无法执行运动")
                return -1
            time.sleep(self.POWER_ON_SETTLE_SEC)

        logger.info("KEBA move_linear: 读取当前关节...")
        current_joint = self._read_joint_pose()
        if current_joint is None:
            logger.warning("KEBA move_linear: 当前关节读取失败")
            return -1

        current_joint_for_ik = self._map_joint_for_fk(current_joint)
        target_pose = CartesianPose(
            x=float(target.x) / 1000.0,
            y=float(target.y) / 1000.0,
            z=float(target.z) / 1000.0,
            roll=float(target.rx),
            pitch=float(target.ry),
            yaw=float(target.rz),
        )
        logger.info(
            "KEBA move_linear: 逆解目标 TCP(m/deg)=X%.4f Y%.4f Z%.4f R%.2f P%.2f Y%.2f",
            target_pose.x,
            target_pose.y,
            target_pose.z,
            target_pose.roll,
            target_pose.pitch,
            target_pose.yaw,
        )
        target_joint = self._inverse_kinematics(target_pose, current_joint_for_ik)
        if target_joint is None:
            logger.warning("KEBA move_linear: 逆解失败")
            return -1

        controller_joint = self._unmap_joint_from_fk(target_joint)
        registers = self._joint_to_registers(controller_joint)

        with self._lock:
            try:
                logger.info("KEBA move_linear: 写入目标关节并触发运动")
                self._client.write_register(self.REG_MODE_WRITE, 0)
                time.sleep(0.005)
                self._client.write_registers(self.REG_JOINTDATA_WRITE, registers)
                time.sleep(0.005)
                self._client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 1)
                time.sleep(0.5)
                self._client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 0)
                return 0
            except ModbusException as exc:
                logger.warning("KEBA move_linear 写寄存器异常: %s", exc)
                return -1
            except Exception as exc:
                logger.warning("KEBA move_linear 失败: %s", exc)
                return -1

    def get_status(self) -> int:
        """读取运动状态，统一映射为 0=空闲/到位, 1=忙碌。"""
        if not self._connected or self._client is None:
            return 1

        with self._lock:
            try:
                result = self._client.read_holding_registers(
                    address=self.REG_MOTIONDONE_READ,
                    count=1,
                )
            except ModbusException as exc:
                logger.debug("KEBA 读取运动状态异常: %s", exc)
                return 1
            except Exception as exc:
                logger.debug("KEBA 读取运动状态失败: %s", exc)
                return 1

        if result.isError():
            logger.debug("KEBA 读取运动状态失败: %s", result)
            return 1

        motion_done = int(result.registers[0])
        return 0 if motion_done == 1 else 1

    def move_force(self, mode: int, displace_target: int = 0) -> bool:
        """进入 KEBA 力控模式。

        mode=1: 力控插枪。
        mode=3: 带目标位移的力控拔枪，先写目标位移再切换 mode。
        """
        if not self._connected or self._client is None:
            logger.warning("KEBA 未连接，无法执行力控模式")
            return False

        mode = int(mode)
        if mode not in (1, 3):
            logger.warning("KEBA 不支持的力控模式: %s", mode)
            return False

        with self._lock:
            try:
                if mode == 1:
                    self._client.write_register(self.REG_MODE_WRITE, 1)
                else:
                    self._client.write_register(
                        self.REG_MODE3TARGET_WRITE,
                        int(displace_target),
                    )
                    time.sleep(0.005)
                    self._client.write_register(self.REG_MODE_WRITE, 3)
            except ModbusException as exc:
                logger.warning("KEBA 力控模式%d设置异常: %s", mode, exc)
                return False
            except Exception as exc:
                logger.warning("KEBA 力控模式%d设置失败: %s", mode, exc)
                return False

        logger.info("KEBA 力控模式%d已设置", mode)
        return True

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def _target_reached(
        self,
        target,
        trans_tol_m: float = ARRIVAL_TRANS_TOL_M,
        rot_tol_deg: float = ARRIVAL_ROT_TOL_DEG,
    ) -> bool:
        """按 KEBA 的真实 TCP(m/deg) 校验脚本目标(mm/deg)是否已到位。"""
        current = self.get_tcp_pose()
        if current is None:
            return False

        target_xyz_m = (
            float(target.x) / 1000.0,
            float(target.y) / 1000.0,
            float(target.z) / 1000.0,
        )
        trans_err = math.sqrt(
            (float(current[0]) - target_xyz_m[0]) ** 2
            + (float(current[1]) - target_xyz_m[1]) ** 2
            + (float(current[2]) - target_xyz_m[2]) ** 2
        )
        rot_err = max(
            self._angle_diff_deg(current[3], target.rx),
            self._angle_diff_deg(current[4], target.ry),
            self._angle_diff_deg(current[5], target.rz),
        )
        return trans_err <= float(trans_tol_m) and rot_err <= float(rot_tol_deg)

    def _wait_until_target_reached(self, target, timeout: float) -> bool:
        deadline = time.time() + float(timeout)
        last_log = 0.0
        while time.time() < deadline:
            if self._target_reached(target):
                logger.info("KEBA move_and_wait: TCP 已确认到位")
                return True
            self.get_status()
            now = time.time()
            if now - last_log >= 2.0:
                logger.info("KEBA move_and_wait: 等待 TCP 到位...")
                last_log = now
            time.sleep(0.05)
        return self._target_reached(target)

    def move_and_wait(self, target, timeout: float = 30.0) -> bool:
        """
        直线运动并等待完成。target 使用脚本约定的 mm/deg。

        KEBA 的完成寄存器可能不能代表 TCP 已到目标，因此这里用正运动学
        读取实际 TCP 做到位确认；首次未到位时重新上电并重发一次目标。
        """
        for attempt in range(2):
            ret = self.move_linear(target)
            if ret != 0:
                if attempt == 0 and self.power_on():
                    continue
                return False

            if self._wait_until_target_reached(target, timeout):
                return True

            if attempt == 0:
                logger.warning("KEBA 运动后未确认到位，重新上电并重发目标")
                if not self.power_on():
                    break

        logger.warning("KebaRobot.move_and_wait 超时 (%.1fs)", timeout)
        return False

    def power_on(self) -> bool:
        """机器人上电。"""
        if not self._connected or self._client is None:
            logger.warning("KEBA 未连接，无法上电")
            return False

        with self._lock:
            try:
                self._client.write_register(self.REG_POWERON_WRITE, 1)
                time.sleep(0.005)
                self._client.write_register(self.REG_START_WRITE, 1)
                logger.info("KEBA 机器人上电完成")
                return True
            except ModbusException as exc:
                logger.warning("KEBA 上电失败: %s", exc)
                return False
            except Exception as exc:
                logger.warning("KEBA 上电异常: %s", exc)
                return False

    def power_off(self) -> bool:
        """机器人下电。"""
        if not self._connected or self._client is None:
            logger.warning("KEBA 未连接，无法下电")
            return False

        with self._lock:
            try:
                self._client.write_register(self.REG_POWERON_WRITE, 0)
                time.sleep(0.005)
                self._client.write_register(self.REG_START_WRITE, 0)
                logger.info("KEBA 机器人下电完成")
                return True
            except ModbusException as exc:
                logger.warning("KEBA 下电失败: %s", exc)
                return False
            except Exception as exc:
                logger.warning("KEBA 下电异常: %s", exc)
                return False

    def get_power_status(self) -> Optional[int]:
        """
        读取电源状态。

        Returns:
            1=上电, 0=断电, None=读取失败
        """
        if not self._connected or self._client is None:
            return None

        with self._lock:
            try:
                result = self._client.read_holding_registers(
                    address=self.REG_POWERON_READ,
                    count=1,
                )
            except ModbusException as exc:
                logger.debug("KEBA 读取电源状态异常: %s", exc)
                return None
            except Exception as exc:
                logger.debug("KEBA 读取电源状态失败: %s", exc)
                return None

        if result.isError():
            logger.debug("KEBA 读取电源状态失败: %s", result)
            return None

        return int(result.registers[0])

    def __enter__(self) -> "KebaRobot":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"KebaRobot(ip={self._ip!r}, port={self._port}, status={status})"

    def write_pose_registers(self, pose: JointPose) -> bool:
        """将目标关节角度写入 Modbus 寄存器区"""
        if not self._client:
            print("[write_pose_registers] 未初始化连接")
            return False

        regs = self.get_registers(pose)
        with self._lock:
            try:
                self._client.write_registers(self.REG_JOINTDATA_WRITE, regs)
            except ModbusException as e:
                print(f"[write_pose_registers] 写入失败: {e}")
                return False
        return True

    @staticmethod
    def float_to_regs(f_val: float) -> Tuple[int, int]:
        """将 float 转换为两个 16 位寄存器值（大端）"""
        packed = struct.pack('>f', f_val)
        high = (packed[0] << 8) | packed[1]
        low = (packed[2] << 8) | packed[3]
        return high, low

    def get_registers(self, pose: JointPose) -> List[int]:
        """将 JointPose 转换为 12 个 uint16 的列表"""
        regs = []
        for q in (pose.Q1, pose.Q2, pose.Q3, pose.Q4, pose.Q5, pose.Q6):
            high, low = self.float_to_regs(q)
            regs.extend([high, low])
        return regs

    def arm_move_joint(self, target: list) -> bool:
        """关节运动：设置模式为关节模式，下发目标角度，触发运动"""

        joint_target = JointPose(float(target[0]), float(target[1]), float(target[2]), float(target[3]), float(target[4]), float(target[5]))

        # 设置模式为0（关节模式）
        # mode = 0
        with self._lock:
            try:
                self._client.write_register(self.REG_MODE_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_move_joint] 写模式失败: {e}")
                return False

        time.sleep(0.005)

        if not self.write_pose_registers(joint_target):
            return False

        time.sleep(0.005)

        # 触发运动
        with self._lock:
            try:
                self._client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 1)
            except ModbusException as e:
                print(f"[arm_move_joint] 触发运动失败: {e}")
                return False

        time.sleep(0.5)

        # 清除触发标志
        with self._lock:
            try:
                self._client.write_register(self.REG_JOINTDATASENDSTATE_WRITE, 0)
            except ModbusException as e:
                print(f"[arm_move_joint] 清除触发标志失败: {e}")
                return False

        print("[arm_move_joint] 关节运动指令下发成功")
        return True