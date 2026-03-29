"""
Modbus TCP 机械臂接口 | Modbus Robot Interface

通过 Modbus TCP 协议控制机械臂，封装 robot_driver 中的 RobotInterface。
- 通信：pymodbus 库，无需专有 SDK
- 原始单位：mm + deg，ZYX 内旋欧拉角（RPY）
- get_tcp_pose() 输出格式与 JAKARobot 完全一致：(mm, mm, mm, deg, deg, deg)
"""

import logging
import time
from typing import Optional, Tuple

from robovision.robot.base import RobotBase

logger = logging.getLogger(__name__)


class ModbusRobot(RobotBase):
    """
    Modbus TCP 机械臂接口。

    用法::

        robot = ModbusRobot(ip='192.168.1.133', port=502)
        robot.connect()
        pose = robot.get_tcp_pose()
        robot.disconnect()

        # 或者用 with 语句
        with ModbusRobot(ip='192.168.1.133') as robot:
            pose = robot.get_tcp_pose()
    """

    def __init__(self, ip: str, port: int = 502, unit_id: int = 1):
        self._ip = ip
        self._port = port
        self._unit_id = unit_id
        self._driver = None  # 延迟导入和创建
        self._connected = False

    def _ensure_driver(self):
        """延迟创建 RobotInterface 实例。"""
        if self._driver is None:
            from third_party.robot_driver.robot_driver_interface import RobotInterface
            self._driver = RobotInterface(
                ip=self._ip, port=self._port, unit_id=self._unit_id
            )

    def connect(self) -> bool:
        """连接机械臂，返回是否成功。"""
        try:
            self._ensure_driver()
            result = self._driver.connect()
            self._connected = bool(result)
            if self._connected:
                logger.info("Modbus 机械臂连接成功: %s:%d", self._ip, self._port)
            else:
                logger.warning("Modbus 机械臂连接失败: %s:%d", self._ip, self._port)
            return self._connected
        except Exception as e:
            self._connected = False
            logger.error("Modbus 机械臂连接异常: %s", e)
            return False

    def disconnect(self) -> None:
        """断开机械臂连接。"""
        if self._driver is not None:
            try:
                self._driver.close()
                logger.info("Modbus 机械臂已断开连接")
            except Exception:
                pass
        self._connected = False

    def get_tcp_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取当前 TCP 位姿。

        Returns:
            (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) 或 None（失败时）
            单位：毫米、度，ZYX 内旋欧拉角（RPY）
        """
        if not self._connected or self._driver is None:
            return None
        try:
            pose = self._driver.arm_get_current_pose()
            if pose is None:
                return None
            return (
                pose.x, pose.y, pose.z,
                pose.rx, pose.ry, pose.rz,
            )
        except Exception as e:
            logger.debug("获取 TCP 位姿异常: %s", e)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ——— 扩展方法（仅 Modbus 臂有） ———

    def set_coord_sys(self, coord: int) -> int:
        """切换坐标系：0=World, 1=Camera, 2=Flange。"""
        self._ensure_driver()
        return self._driver.arm_set_coord_sys(coord)

    def get_coord_sys(self) -> int:
        """读取当前坐标系。"""
        self._ensure_driver()
        return self._driver.arm_get_coord_sys()

    def set_speed(self, speed: int) -> int:
        """设置坐标系运动速度 (0~100%)。"""
        self._ensure_driver()
        return self._driver.arm_set_coord_speed(speed)

    def move_linear(self, target) -> int:
        """笛卡尔直线运动。target: CartesianPose 实例。"""
        self._ensure_driver()
        return self._driver.arm_move_linear(target)

    def move_joint(self, target) -> int:
        """关节空间运动。target: CartesianPose 实例。"""
        self._ensure_driver()
        return self._driver.arm_move_joint(target)

    def get_status(self) -> int:
        """读取状态寄存器：0=空闲, 1=忙碌。"""
        self._ensure_driver()
        return self._driver.arm_get_status()

    def wait_until_idle(self, reg_addr, timeout: float = 10.0) -> int:
        """轮询等待目标寄存器归零。"""
        self._ensure_driver()
        return self._driver.wait_until_idle(reg_addr, timeout)

    def move_and_wait(self, target, timeout: float = 30.0) -> bool:
        """直线运动并等待完成。返回 True=成功到达。"""
        ret = self.move_linear(target)
        if ret != 0:
            logger.warning("move_linear 指令下发失败 (ret=%d)", ret)
            return False
        time.sleep(0.05)  # 等运动启动
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            if status == 0:
                return True
            time.sleep(0.05)
        logger.warning("move_and_wait 超时 (%.1fs)", timeout)
        return False

    def __enter__(self) -> 'ModbusRobot':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"ModbusRobot(ip={self._ip!r}, port={self._port}, status={status})"
