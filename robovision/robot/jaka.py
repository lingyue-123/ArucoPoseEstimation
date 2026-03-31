"""
JAKA 机械臂接口 | JAKA Robot Interface

封装 jkrc SDK，提供：
- 连接 / 断开管理（支持 with 语句）
- 获取 TCP 位姿（mm, deg）
- 欧拉角约定：ZYX 内旋 / RPY（与 JAKA 控制器一致）
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import time

from robovision.robot.base import RobotBase

logger = logging.getLogger(__name__)


def _ensure_sdk_path(sdk_path: str) -> None:
    """确保 JAKA SDK 路径在 sys.path 和 LD_LIBRARY_PATH 中。"""
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if sdk_path not in ld_path:
        os.environ['LD_LIBRARY_PATH'] = sdk_path + ':' + ld_path
        # 重启进程以使 LD_LIBRARY_PATH 生效（SDK 要求）
        os.execv(sys.executable, [sys.executable] + sys.argv)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


class JAKARobot(RobotBase):
    """
    JAKA 协作机器人接口。

    用法::

        robot = JAKARobot(ip='192.168.1.106', sdk_path='/path/to/jaka-python-sdk')
        robot.connect()
        pose = robot.get_tcp_pose()
        robot.disconnect()

        # 或者用 with 语句
        with JAKARobot(ip='192.168.1.106', sdk_path='...') as robot:
            pose = robot.get_tcp_pose()
    """

    def __init__(self, ip: str, sdk_path: str):
        self._ip = ip
        # 将相对路径转为绝对路径（相对于项目根目录）
        if not os.path.isabs(sdk_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sdk_path = os.path.join(project_root, sdk_path)
        self._sdk_path = sdk_path
        self._robot = None
        self._connected = False
        self._powered = False
        self._enabled = False
        # 在构造时立即确保 SDK 路径，os.execv 若需重启会在此触发，
        # 避免相机已 open() 后重启导致独占句柄未释放
        _ensure_sdk_path(sdk_path)

    def connect(self) -> bool:
        """连接机械臂，返回是否成功。"""
        try:
            import jkrc  # 延迟导入，避免 SDK 路径未设置时报错
            self._robot = jkrc.RC(self._ip)
            ret = self._robot.login()
            if ret[0] == 0:
                self._connected = True
                logger.info("机械臂连接成功: %s", self._ip)
                return True
            else:
                self._connected = False
                logger.warning("机械臂登录失败，错误码: %s", ret)
                return False
        except Exception as e:
            self._connected = False
            logger.error("机械臂连接异常: %s", e)
            return False

    def robot_powered(self):
        # power
        ret_power = self._robot.power_on()
        if ret_power[0] == 0:
            self._powered = True
            logger.info("机械臂上电成功: %s", self._ip)
            return True
        else:
            self._powered = False
            logger.warning("机械臂上电失败，错误码: %s", ret_power)
            return False
        
    def robot_enable(self):
        # enable
        ret_enable = self._robot.enable_robot()
        if ret_enable[0] == 0:
            self._enabled = True
            logger.info("机械臂使能成功: %s", self._ip)
            return True
        else:
            self._enabled = False
            logger.warning("机械臂使能失败，错误码: %s", ret_enable)
            return False
        
    def get_coord_sys(self) -> Optional[int]:
        """获取当前坐标系 ID，失败时返回 None。"""
        if not self._connected or self._robot is None:
            logger.warning("机械臂未连接，无法获取坐标系")
            return None
        try:
            ret = self._robot.get_tool_id()
            if ret[0] == 0:
                coord_sys_id = ret[1]
                logger.info("当前坐标系 ID: %d", coord_sys_id)
                return coord_sys_id
            else:
                logger.warning("获取坐标系失败，错误码: %s", ret)
                return None
        except Exception as e:
            logger.error("获取坐标系异常: %s", e)
            return None
        
    def set_tool_id(self, tool_id: int) -> bool:
        """设置工具坐标系 ID，返回是否成功。"""
        if not self._connected or self._robot is None:
            logger.warning("机械臂未连接，无法设置工具坐标系")
            return False
        try:
            ret = self._robot.set_tool_id(tool_id)
            if ret[0] == 0:
                logger.info("设置工具坐标系成功: tool_id=%d", tool_id)
                return True
            else:
                logger.warning("设置工具坐标系失败，错误码: %s", ret)
                return False
        except Exception as e:
            logger.error("设置工具坐标系异常: %s", e)
            return False
        
    def move_and_wait(self, target, move_mode, move_block, move_speed, timeout: float = 30.0) -> bool:
        """直线运动并等待完成。返回 True=成功到达。"""
        ret = self._robot.linear_move(target, move_mode, move_block, move_speed)
        if ret[0] != 0:
            logger.warning("move_linear 指令下发失败 (ret=%d)", ret)
            return False
        time.sleep(0.05)  # 等运动启动
        return True

    def disconnect(self) -> None:
        """断开机械臂连接。"""
        if self._robot is not None:
            try:
                self._robot.logout()
                logger.info("机械臂已断开连接")
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_tcp_pose(self) -> Optional[tuple]:
        """
        获取当前 TCP 位姿。

        Returns:
            (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) 或 None（失败时）
            单位：毫米、度，ZYX 内旋欧拉角（RPY）
        """
        if not self._connected or self._robot is None:
            return None
        try:
            ret = self._robot.get_tcp_position()
            if ret[0] == 0:
                tcp = ret[1]
                x = tcp[0]
                y = tcp[1]
                z = tcp[2]
                rx = np.degrees(tcp[3])
                ry = np.degrees(tcp[4])
                rz = np.degrees(tcp[5])
                return (x, y, z, rx, ry, rz)
            else:
                logger.warning("获取 TCP 位姿失败，错误码: %s", ret)
                return None
        except Exception as e:
            logger.debug("获取 TCP 位姿异常: %s", e)
            return None

    def get_tcp_pose_raw_mm(self) -> Optional[list]:
        """
        获取原始 TCP 位姿（JAKA 格式，不做单位转换）。

        Returns:
            [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad] 或 None
        """
        if not self._connected or self._robot is None:
            return None
        try:
            ret = self._robot.get_tcp_position()
            if ret[0] == 0:
                return list(ret[1])
            return None
        except Exception as e:
            logger.debug("获取原始 TCP 位姿异常: %s", e)
            return None

    def __enter__(self) -> 'JAKARobot':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"JAKARobot(ip={self._ip!r}, status={status})"
