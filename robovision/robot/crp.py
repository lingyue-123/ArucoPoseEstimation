"""
CRP 机械臂接口 | CRP Robot Interface (CrobotpOS SDK)

封装 crp_robot SDK，提供与 RobotBase 统一的接口：
- 连接 / 断开管理（支持 with 语句）
- 获取 TCP 位姿（mm, deg）
- 欧拉角约定：ZYX 内旋 / RPY
"""

import logging
import os
import sys
from typing import Optional, Tuple

from robovision.robot.base import RobotBase

logger = logging.getLogger(__name__)


class CRPRobot(RobotBase):
    """
    CRP 协作机器人接口（CrobotpOS SDK）。

    用法::

        robot = CRPRobot(
            ip='192.168.1.133',
            so_path='third_party/crp_robot_sdk/libRobotService.so',
            sdk_path='third_party/crp_robot_sdk',
        )
        robot.connect()
        pose = robot.get_tcp_pose()
        robot.disconnect()

        # 或者用 with 语句
        with CRPRobot(ip='...', so_path='...', sdk_path='...') as robot:
            pose = robot.get_tcp_pose()
    """

    def __init__(self, ip: str, so_path: str, sdk_path: str,
                 disable_hardware: bool = True):
        self._ip = ip
        # 将相对路径转为绝对路径（相对于项目根目录）
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self._so_path = so_path if os.path.isabs(so_path) else os.path.join(project_root, so_path)
        self._sdk_path = sdk_path if os.path.isabs(sdk_path) else os.path.join(project_root, sdk_path)
        self._disable_hardware = disable_hardware
        self._robot = None  # crp_robot.Robot 实例
        self._connected = False

    def _ensure_robot(self):
        """延迟导入 crp_robot 并创建 Robot 实例。"""
        if self._robot is None:
            if self._sdk_path not in sys.path:
                sys.path.insert(0, self._sdk_path)
            from crp_robot import Robot
            self._robot = Robot(self._so_path)

    def connect(self) -> bool:
        """连接机械臂并使能伺服，返回是否成功。"""
        try:
            self._ensure_robot()
            ok = self._robot.connect(self._ip, self._disable_hardware)
            if not ok:
                logger.warning("CRP 机械臂连接失败: %s", self._ip)
                return False
            servo_ok = self._robot.servo_on()
            if not servo_ok:
                logger.warning("CRP 机械臂伺服使能失败: %s", self._ip)
            self._connected = True
            logger.info("CRP 机械臂连接成功: %s (servo=%s)", self._ip, servo_ok)
            return True
        except Exception as e:
            self._connected = False
            logger.error("CRP 机械臂连接异常: %s", e)
            return False

    def disconnect(self) -> None:
        """关闭伺服并断开连接。"""
        if self._robot is not None:
            try:
                self._robot.servo_off()
                self._robot.disconnect()
                logger.info("CRP 机械臂已断开连接")
            except Exception:
                pass
            try:
                self._robot.close()
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
        if not self._connected or self._robot is None:
            return None
        try:
            pos = self._robot.get_position()
            return (
                pos.x, pos.y, pos.z,
                pos.Rx, pos.Ry, pos.Rz,
            )
        except Exception as e:
            logger.debug("获取 TCP 位姿异常: %s", e)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ——— 扩展方法（仅 CRP 臂有） ———

    def servo_on(self) -> bool:
        """伺服使能。"""
        self._ensure_robot()
        return self._robot.servo_on()

    def servo_off(self) -> bool:
        """伺服关闭。"""
        self._ensure_robot()
        return self._robot.servo_off()

    def get_joint(self):
        """获取关节角度。"""
        self._ensure_robot()
        return self._robot.get_joint()

    def stop_move(self) -> bool:
        """停止运动。"""
        self._ensure_robot()
        return self._robot.stop_move()

    def is_moving(self) -> bool:
        """是否在运动中。"""
        self._ensure_robot()
        return self._robot.is_moving()

    def has_error(self) -> bool:
        """是否有报警。"""
        self._ensure_robot()
        return self._robot.has_error()

    def clear_error(self) -> bool:
        """清除报警。"""
        self._ensure_robot()
        return self._robot.clear_error()

    @property
    def io(self):
        """IO 子服务。"""
        self._ensure_robot()
        return self._robot.io

    @property
    def motion(self):
        """运动子服务。"""
        self._ensure_robot()
        return self._robot.motion

    @property
    def model(self):
        """模型子服务。"""
        self._ensure_robot()
        return self._robot.model

    def __enter__(self) -> 'CRPRobot':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"CRPRobot(ip={self._ip!r}, status={status})"
