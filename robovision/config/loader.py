"""
配置加载器 | Configuration Loader

统一解析 config/ 目录下的 YAML 文件，构造强类型数据类。
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# 默认配置目录（相对于本文件的三级上目录，即项目根目录）
_DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'config')


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class NetworkConfig:
    """网络接口配置（用于自动配网）。"""
    iface: str           # e.g. "eth0"
    local_ip: str        # e.g. "192.168.1.132/24"
    target_ip: str       # e.g. "192.168.1.133"
    delete_default_route: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'NetworkConfig':
        return cls(
            iface=d['iface'],
            local_ip=d['local_ip'],
            target_ip=d['target_ip'],
            delete_default_route=d.get('delete_default_route', False),
        )


@dataclass
class CameraIntrinsicsConfig:
    """相机内参数据类（从 YAML 解析）。"""
    camera_matrix: np.ndarray   # shape (3, 3)
    dist_coeffs: np.ndarray     # shape (N,)
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraIntrinsicsConfig':
        K = np.array(d['camera_matrix'], dtype=np.float64)
        dist = np.array(d['dist_coeffs'], dtype=np.float64)
        w = d.get('width', 0)
        h = d.get('height', 0)
        return cls(camera_matrix=K, dist_coeffs=dist, width=w, height=h)


@dataclass
class CameraConnectionConfig:
    """相机连接参数（USB / RTSP / Hikvision / Mech-Eye）。"""
    device_id: Optional[int] = None
    url: Optional[str] = None
    ip: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: int = 30
    buffer_size: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraConnectionConfig':
        return cls(
            device_id=d.get('device_id'),
            url=d.get('url'),
            ip=d.get('ip'),
            width=d.get('width', 0),
            height=d.get('height', 0),
            fps=d.get('fps', 30),
            buffer_size=d.get('buffer_size', 1),
        )


@dataclass
class CameraConfig:
    """单个相机的完整配置。"""
    name: str
    type: str                          # 'hikvision' | 'usb' | 'rtsp' | 'mecheye'
    intrinsics: CameraIntrinsicsConfig
    connection: Optional[CameraConnectionConfig] = None
    network: Optional[NetworkConfig] = None


@dataclass
class MarkerConfig:
    """ArUco Marker 配置。"""
    dictionary: str
    valid_ids: List[int]
    marker_sizes: Dict[int, float]
    pose_files: Dict[int, str]


@dataclass
class ChessboardConfig:
    """棋盘格标定板配置。"""
    pattern_size: Tuple[int, int]  # (cols, rows) 内角点数
    square_size: float             # 米


@dataclass
class RobotConfig:
    """机械臂配置。"""
    ip: str
    sdk_path: str
    driver: str = 'jaka'     # 'jaka' / 'modbus' / 'crp'
    port: int = 502           # 仅 modbus 用
    so_path: str = ''         # crp: libRobotService.so 路径
    disable_hardware: bool = True  # crp: connect 参数
    timeout: int = 10
    pose_convention: str = 'ZYX'
    network: Optional[NetworkConfig] = None


@dataclass
class DetectionConfig:
    """检测与滤波参数配置。"""
    # Kalman
    kf_state_dim: int = 8
    kf_measure_dim: int = 8
    kf_process_noise: float = 0.1
    kf_measurement_noise: float = 0.05
    kf_error_cov_init: float = 1.0
    lost_threshold: int = 30
    # Pose smoother
    rot_slerp_alpha_normal: float = 0.35
    rot_slerp_alpha_cautious: float = 0.15
    trans_ema_alpha_normal: float = 0.40
    trans_ema_alpha_cautious: float = 0.15
    velocity_window_size: int = 5
    motion_threshold: float = 5.0
    # Anomaly gate
    reproj_warn_px: float = 8.0
    reproj_max_px: float = 20.0
    angle_jump_warn_deg: float = 15.0
    angle_jump_max_deg: float = 40.0
    trans_jump_warn_mm: float = 30.0
    trans_jump_max_mm: float = 120.0
    motion_threshold_multiplier: float = 2.0
    anomaly_confirm_frames: int = 3


class Config:
    """
    全局配置对象，懒加载所有 YAML 文件。

    用法：
        cfg = Config()                         # 使用默认 config/ 目录
        cfg = Config('/path/to/config')        # 自定义目录
        cam = cfg.get_camera('hikvision_normal')
    """

    def __init__(self, config_dir: Optional[str] = None):
        self._dir = os.path.abspath(config_dir or _DEFAULT_CONFIG_DIR)
        self._cameras_raw: Optional[dict] = None
        self._markers_raw: Optional[dict] = None
        self._robot_raw: Optional[dict] = None
        self._detection_raw: Optional[dict] = None

    def _cameras(self) -> dict:
        if self._cameras_raw is None:
            self._cameras_raw = _load_yaml(os.path.join(self._dir, 'cameras.yaml'))
        return self._cameras_raw

    def _markers(self) -> dict:
        if self._markers_raw is None:
            self._markers_raw = _load_yaml(os.path.join(self._dir, 'markers.yaml'))
        return self._markers_raw

    def _robot(self) -> dict:
        if self._robot_raw is None:
            self._robot_raw = _load_yaml(os.path.join(self._dir, 'robot.yaml'))
        return self._robot_raw

    def _detection(self) -> dict:
        if self._detection_raw is None:
            self._detection_raw = _load_yaml(os.path.join(self._dir, 'detection.yaml'))
        return self._detection_raw

    def get_camera(self, name: Optional[str] = None) -> CameraConfig:
        """获取指定相机配置。name=None 时使用 default_camera。"""
        data = self._cameras()
        if name is None:
            name = data.get('default_camera', 'hikvision_mono')
        raw = data['cameras']
        if name not in raw:
            available = list(raw.keys())
            raise KeyError(f"相机 '{name}' 未在配置中找到。可用: {available}")
        d = raw[name]
        intrinsics = CameraIntrinsicsConfig.from_dict(d['intrinsics'])
        connection = CameraConnectionConfig.from_dict(d['connection']) if 'connection' in d else None
        network = NetworkConfig.from_dict(d['network']) if 'network' in d else None
        return CameraConfig(name=name, type=d['type'], intrinsics=intrinsics,
                            connection=connection, network=network)

    def list_cameras(self) -> List[str]:
        """列出所有已配置的相机名称。"""
        return list(self._cameras()['cameras'].keys())

    def get_marker(self) -> MarkerConfig:
        """获取 ArUco Marker 配置。"""
        d = self._markers()['aruco']
        return MarkerConfig(
            dictionary=d['dictionary'],
            valid_ids=d['valid_ids'],
            marker_sizes={int(k): v for k, v in d['marker_sizes'].items()},
            pose_files={int(k): v for k, v in d['pose_files'].items()},
        )

    def get_chessboard(self) -> ChessboardConfig:
        """获取棋盘格配置。"""
        d = self._markers()['chessboard']
        ps = d['pattern_size']
        return ChessboardConfig(
            pattern_size=(ps[0], ps[1]),
            square_size=d['square_size'],
        )

    def get_robot(self, driver: Optional[str] = None) -> RobotConfig:
        """获取机械臂配置。

        Args:
            driver: 驱动类型 ('jaka'/'modbus')，None 时使用配置文件的 default_driver
        """
        raw = self._robot()

        # 新格式：多驱动并存（有 'drivers' 键）
        if 'drivers' in raw:
            drv = driver or raw.get('default_driver', 'modbus')
            drivers = raw['drivers']
            if drv not in drivers:
                raise KeyError(f"驱动 '{drv}' 未在 robot.yaml 中配置。可用: {list(drivers.keys())}")
            d = drivers[drv]
            network = NetworkConfig.from_dict(d['network']) if 'network' in d else None
            return RobotConfig(
                ip=d['ip'],
                sdk_path=d.get('sdk_path', ''),
                driver=drv,
                port=d.get('port', 502),
                so_path=d.get('so_path', ''),
                disable_hardware=d.get('disable_hardware', True),
                timeout=raw.get('timeout', d.get('timeout', 10)),
                pose_convention=raw.get('pose_convention', d.get('pose_convention', 'ZYX')),
                network=network,
            )

        # 兼容旧格式（顶层 key 为 'robot' 或 'jaka'）
        d = raw.get('robot', raw.get('jaka', {}))
        network = NetworkConfig.from_dict(d['network']) if 'network' in d else None
        return RobotConfig(
            ip=d['ip'],
            sdk_path=d.get('sdk_path', ''),
            driver=driver or d.get('driver', 'jaka'),
            port=d.get('port', 502),
            so_path=d.get('so_path', ''),
            disable_hardware=d.get('disable_hardware', True),
            timeout=d.get('timeout', 10),
            pose_convention=d.get('pose_convention', 'ZYX'),
            network=network,
        )

    def get_detection(self) -> DetectionConfig:
        """获取检测参数配置。"""
        d = self._detection()
        kd = d.get('kalman', {})
        pd = d.get('pose_smoother', {})
        ad = d.get('anomaly_gate', {})
        return DetectionConfig(
            kf_state_dim=kd.get('state_dim', 8),
            kf_measure_dim=kd.get('measure_dim', 8),
            kf_process_noise=kd.get('process_noise', 0.1),
            kf_measurement_noise=kd.get('measurement_noise', 0.05),
            kf_error_cov_init=kd.get('error_cov_init', 1.0),
            lost_threshold=kd.get('lost_threshold', 30),
            rot_slerp_alpha_normal=pd.get('rot_slerp_alpha_normal', 0.35),
            rot_slerp_alpha_cautious=pd.get('rot_slerp_alpha_cautious', 0.15),
            trans_ema_alpha_normal=pd.get('trans_ema_alpha_normal', 0.40),
            trans_ema_alpha_cautious=pd.get('trans_ema_alpha_cautious', 0.15),
            velocity_window_size=pd.get('velocity_window_size', 5),
            motion_threshold=pd.get('motion_threshold', 5.0),
            reproj_warn_px=ad.get('reproj_warn_px', 8.0),
            reproj_max_px=ad.get('reproj_max_px', 20.0),
            angle_jump_warn_deg=ad.get('angle_jump_warn_deg', 15.0),
            angle_jump_max_deg=ad.get('angle_jump_max_deg', 40.0),
            trans_jump_warn_mm=ad.get('trans_jump_warn_mm', 30.0),
            trans_jump_max_mm=ad.get('trans_jump_max_mm', 120.0),
            motion_threshold_multiplier=ad.get('motion_threshold_multiplier', 2.0),
            anomaly_confirm_frames=ad.get('anomaly_confirm_frames', 3),
        )


# 全局默认配置实例（懒加载）
_default_config: Optional[Config] = None


def get_config(config_dir: Optional[str] = None) -> Config:
    """获取全局默认配置实例（单例）。"""
    global _default_config
    if _default_config is None or config_dir is not None:
        _default_config = Config(config_dir)
    return _default_config
