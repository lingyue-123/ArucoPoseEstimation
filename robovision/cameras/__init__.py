"""相机抽象层 | Camera Abstraction Layer"""

from .base import CameraInterface, CameraIntrinsics
from .hikvision import HikvisionCamera
from .usb import USBCamera
from .rtsp import RTSPCamera
from .mecheye import MechEyeCamera, MechEyeImageFolderCamera

__all__ = [
    'CameraInterface',
    'CameraIntrinsics',
    'HikvisionCamera',
    'USBCamera',
    'RTSPCamera',
    'MechEyeCamera',
    'MechEyeImageFolderCamera',
]


def build_camera(camera_name=None, config=None) -> CameraInterface:
    """
    工厂函数：根据配置名称创建相机实例。

    Args:
        camera_name: config/cameras.yaml 中的相机名称，None 时使用 default_camera
        config: Config 实例，None 时使用默认配置

    Returns:
        对应的 CameraInterface 实现
    """
    from robovision.config.loader import get_config
    cfg = config or get_config()
    cam_cfg = cfg.get_camera(camera_name)
    intrinsics = CameraIntrinsics.from_config(cam_cfg.intrinsics, name=cam_cfg.name)

    # 自动配置网络接口
    if cam_cfg.network:
        from robovision.network import ensure_interface
        n = cam_cfg.network
        ensure_interface(n.iface, n.local_ip, n.target_ip, n.delete_default_route)

    if cam_cfg.type == 'hikvision':
        return HikvisionCamera(intrinsics=intrinsics)
    elif cam_cfg.type == 'usb':
        conn = cam_cfg.connection
        return USBCamera(
            intrinsics=intrinsics,
            device_id=conn.device_id or 0,
            width=conn.width,
            height=conn.height,
            fps=conn.fps,
        )
    elif cam_cfg.type == 'rtsp':
        conn = cam_cfg.connection
        return RTSPCamera(
            intrinsics=intrinsics,
            url=conn.url,
            buffer_size=conn.buffer_size,
        )
    elif cam_cfg.type == 'mecheye':
        conn = cam_cfg.connection
        return MechEyeCamera(
            intrinsics=intrinsics,
            ip=conn.ip or "",
        )
    else:
        raise ValueError(f"未知相机类型: {cam_cfg.type!r}")
