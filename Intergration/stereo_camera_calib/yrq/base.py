"""
机械臂接口基类 | Robot Base Interface

定义所有机械臂驱动必须实现的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class RobotBase(ABC):
    """机械臂接口基类。所有实现必须返回统一格式。"""

    @abstractmethod
    def connect(self) -> bool:
        """连接机械臂，返回是否成功。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开机械臂连接。"""
        ...

    @abstractmethod
    def get_tcp_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取当前 TCP 位姿。

        Returns:
            (x_m, y_m, z_m, rx_deg, ry_deg, rz_deg) 或 None（失败时）
            单位：米、度，ZYX 内旋欧拉角（RPY），Base←TCP
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""
        ...

    def move_linear(self, target) -> int:
        """非阻塞直线运动到目标位姿。子类应返回 0=成功，其它=失败。"""
        raise NotImplementedError(f"{type(self).__name__} does not implement move_linear()")

    def get_status(self) -> int:
        """读取运动状态。约定 0=空闲/到位, 1=忙碌。"""
        raise NotImplementedError(f"{type(self).__name__} does not implement get_status()")

    @abstractmethod
    def move_and_wait(self, target, timeout: float = 30.0) -> bool:
        """
        直线运动到目标位姿并等待完成。

        Args:
            target: CartesianPose 实例，单位 mm / deg
            timeout: 等待超时秒数
        Returns:
            True=成功到达，False=失败或超时
        """
        ...

    def set_speed(self, speed_pct: int) -> int:
        """设置运动速度比例 (0~100%)。默认 no-op，子类可覆盖。"""
        return 0

    def set_coord_sys(self, coord: int) -> int:
        """切换坐标系。默认 no-op，子类可覆盖。返回 0=成功。"""
        return 0

    def get_coord_sys(self) -> Optional[int]:
        """读取当前坐标系编号。默认返回 None，子类可覆盖。"""
        return None
