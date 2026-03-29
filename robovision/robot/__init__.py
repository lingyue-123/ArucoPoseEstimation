"""
机械臂接口 | Robot Interfaces

支持的驱动：
- jaka: JAKA 协作机器人（jkrc SDK）
- modbus: Modbus TCP 通用机械臂（pymodbus）
- crp: CRP 协作机器人（CrobotpOS SDK）
"""

from robovision.robot.base import RobotBase


def build_robot(robot_cfg) -> RobotBase:
    """
    根据配置创建机械臂实例。

    Args:
        robot_cfg: RobotConfig 实例，包含 driver/ip/sdk_path/port 等字段

    Returns:
        RobotBase 子类实例
    """
    # 自动配置网络接口
    if getattr(robot_cfg, 'network', None):
        from robovision.network import ensure_interface
        n = robot_cfg.network
        ensure_interface(n.iface, n.local_ip, n.target_ip, n.delete_default_route)

    driver = getattr(robot_cfg, 'driver', 'jaka')
    if driver == 'jaka':
        from robovision.robot.jaka import JAKARobot
        return JAKARobot(ip=robot_cfg.ip, sdk_path=robot_cfg.sdk_path)
    elif driver == 'modbus':
        from robovision.robot.modbus import ModbusRobot
        port = getattr(robot_cfg, 'port', 502)
        return ModbusRobot(ip=robot_cfg.ip, port=port)
    elif driver == 'crp':
        from robovision.robot.crp import CRPRobot
        return CRPRobot(
            ip=robot_cfg.ip,
            so_path=robot_cfg.so_path,
            sdk_path=robot_cfg.sdk_path,
            disable_hardware=robot_cfg.disable_hardware,
        )
    else:
        raise ValueError(f"未知 robot driver: {driver}")
