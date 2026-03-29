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
