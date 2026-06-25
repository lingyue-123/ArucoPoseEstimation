import struct
import time
import threading
import logging
from typing import Tuple, Optional, List, Any
import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R,Slerp

import math
import crobotsdk
from robovision.robot.base import RobotBase
from crobotsdk import RobotMode, InstMoveJ, InstMoveC, InstMoveL, InstMoveAbsJ, MoveStrategy, MovePathResult, MotionType,JointPosition,RobotPosition,RobotPosture,ProgramStatus
from typing import Tuple
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置日志（只输出 INFO 及以上）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotInterface")


class CartesianPose:
    """位姿数据类"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, 
                 rx: float = 0.0, ry: float = 0.0, rz: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def from_list(self, pose: List[float]):
        if len(pose) != 6:
            logger.error(f"位姿列表长度必须为6，当前长度: {len(pose)}")
            raise ValueError(f"位姿列表长度必须为6，当前长度: {len(pose)}")
        
        self.x = pose[0]
        self.y = pose[1]
        self.z = pose[2]
        self.rx = pose[3]
        self.ry = pose[4]
        self.rz = pose[5]

    def to_list(self) -> List[float]:
        return [round(self.x, 3), round(self.y, 3), round(self.z, 3), round(self.rx, 3), round(self.ry, 3), round(self.rz, 3)]

    def __repr__(self) -> str:
        return f"CartesianPose(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, rx={self.rx:.3f}, ry={self.ry:.3f}, rz={self.rz:.3f})"
    
 
        


def pose_to_homogeneous_matrix(pose: CartesianPose, degrees: bool = True) -> np.ndarray:
    euler_angles = [pose.rz, pose.ry, pose.rx]
    r = R.from_euler('ZYX', euler_angles, degrees=degrees)
    rotation_matrix = r.as_matrix()

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [pose.x, pose.y, pose.z]
    return T


def homogeneous_matrix_to_pose(matrix: np.ndarray, degrees: bool = True) -> CartesianPose:
    if matrix.shape != (4, 4):
        error_msg = f"输入矩阵维度错误，需为4x4，当前为{matrix.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    
    rot_matrix = matrix[:3, :3]
    try:
        r = R.from_matrix(rot_matrix)
    except ValueError as e:
        error_msg = f"旋转矩阵不合法：{e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    euler_angles = r.as_euler('ZYX', degrees=degrees)
    rz = euler_angles[0]
    ry = euler_angles[1]
    rx = euler_angles[2]
    
    return CartesianPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)


def get_flange_relative_move(pose1: CartesianPose, pose2: CartesianPose) -> CartesianPose:
    """
    输入：基坐标系下的两个位姿 pose1(起点), pose2(终点)
    输出：法兰坐标系（pose1姿态）下的相对位姿 CartesianPose
          dx, dy, dz = 移动增量
          drx, dry, drz = 旋转增量
    """
    # 1. 位姿 → 齐次矩阵
    T1 = pose_to_homogeneous_matrix(pose1)  # 基 -> 法兰1
    T2 = pose_to_homogeneous_matrix(pose2)  # 基 -> 法兰2

    # 2. 求逆矩阵：法兰1 -> 基
    T1_inv = np.linalg.inv(T1)

    # 3. 核心：法兰1 -> 法兰2（相对位姿矩阵）
    T_flange_rel = T1_inv @ T2

    # 4. 提取相对平移
    dx = T_flange_rel[0, 3]
    dy = T_flange_rel[1, 3]
    dz = T_flange_rel[2, 3]

    # 5. 提取相对旋转 → 转 ZYX 欧拉角
    rot_mat = T_flange_rel[:3, :3]
    r = R.from_matrix(rot_mat)
    drz, dry, drx = r.as_euler('ZYX', degrees=True)

    # 6. 返回 CartesianPose 对象
    return CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)

def interpolate_path(waypoints: List[List[float]],
                     num_points: int = 2000) -> List[List[float]]:
    n = len(waypoints)
    if n < 2:
        return []

    x = np.array([p[0] for p in waypoints])
    y = np.array([p[1] for p in waypoints])
    z = np.array([p[2] for p in waypoints])

    t = np.zeros(n)
    for i in range(1, n):
        dx, dy, dz = x[i] - x[i-1], y[i] - y[i-1], z[i] - z[i-1]
        t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy + dz*dz)
    if t[-1] > 1e-6:
        t = t / t[-1]
    else:
        t = np.linspace(0, 1, n)

    t_new = np.linspace(0, 1, num_points)

    spl_x = make_interp_spline(t, x, k=3, bc_type='natural')
    spl_y = make_interp_spline(t, y, k=3, bc_type='natural')
    spl_z = make_interp_spline(t, z, k=3, bc_type='natural')

    eulers = np.array([[p[3], p[4], p[5]] for p in waypoints])
    key_rots = R.from_euler('XYZ', eulers, degrees=True)
    slerp = Slerp(t, key_rots)
    interp_rots = slerp(t_new)
    interp_eulers = interp_rots.as_euler('XYZ', degrees=True)

    result = []
    for i in range(num_points):
        result.append([float(spl_x(t_new[i])), float(spl_y(t_new[i])),
                       float(spl_z(t_new[i])),
                       float(interp_eulers[i, 0]),
                       float(interp_eulers[i, 1]),
                       float(interp_eulers[i, 2])])
    return result



def s_curve_profile(total_distance: float, total_time: float, dt: float = 0.008):
    """
    正弦S曲线速度规划：起点/终点速度=0，中间平滑加速→减速

    Args:
        total_distance: 路径总长 (mm)
        total_time:     目标运动时间 (s)
        dt:             采样周期 (s)

    Returns:
        (dist_cum: np.ndarray, t_array: np.ndarray)
    """
    t = np.arange(0, total_time + dt, dt)
    s = np.sin(np.pi * t / total_time)
    s = s / np.sum(s)
    dist_cum = np.cumsum(s) * total_distance
    return dist_cum, t


def trapezoidal_profile(total_distance: float, total_time: float,
                        dt: float = 0.008, accel_frac: float = 0.25):
    """
    梯形速度规划：起点/终点速度=0，线性加速→匀速→线性减速

    Args:
        total_distance: 路径总长 (mm)
        total_time:     目标运动时间 (s)
        dt:             采样周期 (s)
        accel_frac:     加速段占总时间的比例，自动裁剪至 (0, 0.5]，
                        0.5 表示无匀速段（三角形速度曲线）

    Returns:
        (dist_cum: np.ndarray, t_array: np.ndarray)
    """
    accel_frac = float(min(max(accel_frac, 1e-3), 0.5))
    t = np.arange(0, total_time + dt, dt)
    t_acc = accel_frac * total_time
    t_dec = total_time - t_acc

    v = np.ones_like(t)
    ramp_up = t < t_acc
    v[ramp_up] = t[ramp_up] / t_acc
    ramp_dn = t > t_dec
    v[ramp_dn] = np.clip((total_time - t[ramp_dn]) / t_acc, 0.0, 1.0)

    total = np.sum(v)
    if total <= 1e-9:
        return np.linspace(0, total_distance, len(t)), t
    v = v / total
    dist_cum = np.cumsum(v) * total_distance
    return dist_cum, t

def visualize_trajectory(trajectory: List[List[float]],
                         waypoints: Optional[List[List[float]]] = None,
                         title: str = "轨迹可视化",
                         filepath: Optional[str] = None,
                         dt: float = 0.008) -> None:
    if not trajectory:
        logger.error("轨迹为空，无法可视化")
        return

    traj = np.array(trajectory)
    n = len(traj)
    t = np.arange(n) * dt

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=1.5, label='Path')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=80, marker='o', label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=80, marker='^', label='End')
    if waypoints and len(waypoints) > 0:
        wp = np.array(waypoints)
        ax1.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c='orange', s=40, marker='s', label='Waypoints')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.legend(fontsize=8)
    ax1.set_title('3D Path')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, traj[:, 0], 'r-', label='X')
    ax2.plot(t, traj[:, 1], 'g-', label='Y')
    ax2.plot(t, traj[:, 2], 'b-', label='Z')
    if waypoints and len(waypoints) > 0:
        wp = np.array(waypoints)
        t_wp = np.linspace(0, t[-1], len(wp))
        ax2.plot(t_wp, wp[:, 0], 'r^', markersize=6)
        ax2.plot(t_wp, wp[:, 1], 'g^', markersize=6)
        ax2.plot(t_wp, wp[:, 2], 'b^', markersize=6)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (mm)')
    ax2.legend(fontsize=8)
    ax2.set_title('Position vs Time')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, traj[:, 3], 'r-', label='Rx')
    ax3.plot(t, traj[:, 4], 'g-', label='Ry')
    ax3.plot(t, traj[:, 5], 'b-', label='Rz')
    if waypoints and len(waypoints) > 0:
        wp = np.array(waypoints)
        t_wp = np.linspace(0, t[-1], len(wp))
        ax3.plot(t_wp, wp[:, 3], 'r^', markersize=6)
        ax3.plot(t_wp, wp[:, 4], 'g^', markersize=6)
        ax3.plot(t_wp, wp[:, 5], 'b^', markersize=6)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (deg)')
    ax3.legend(fontsize=8)
    ax3.set_title('Orientation vs Time')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    if n > 1:
        vx = np.gradient(traj[:, 0], dt)
        vy = np.gradient(traj[:, 1], dt)
        vz = np.gradient(traj[:, 2], dt)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        ax4.plot(t, speed, 'k-', linewidth=1.5, label='Speed')
        ax4.fill_between(t, speed, alpha=0.2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (mm/s)')
    ax4.legend(fontsize=8)
    ax4.set_title('Speed vs Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"轨迹图已保存至 {filepath}")
    else:
        plt.show()
    plt.close(fig)


def smooth_cartesian_traj(waypoints: List[List[float]], total_time: float,
                          dt: float = 0.008, profile: str = 'scurve',
                          accel_frac: float = 0.25) -> List[List[float]]:
    """
    几何路径插值 + 时间参数化 → 输出时间确定的平滑轨迹

    Args:
        waypoints:  笛卡尔路径点 [[x,y,z,rx,ry,rz], ...]
        total_time: 目标运动总时间 (s)
        dt:         采样周期/插补周期 (s)，默认 8ms
        profile:    速度规划类型: 'scurve' 正弦S曲线 / 'trapezoid' 梯形
        accel_frac: 梯形规划加速段占比 (仅 profile='trapezoid' 时有效)

    Returns:
        List[List[float]]  时间参数化轨迹点，点数 ≈ total_time / dt
    """
    if len(waypoints) < 2:
        logger.error("路径点至少需要2个")
        return []

    N_dense = 2000
    path = interpolate_path(waypoints, N_dense)

    dists = [0.0]
    for i in range(1, N_dense):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dz = path[i][2] - path[i - 1][2]
        dists.append(dists[-1] + np.sqrt(dx * dx + dy * dy + dz * dz))
    dists = np.array(dists)

    if profile == 'trapezoid':
        s_curve_dist, _ = trapezoidal_profile(dists[-1], total_time, dt, accel_frac)
    elif profile == 'scurve':
        s_curve_dist, _ = s_curve_profile(dists[-1], total_time, dt)
    else:
        logger.error(f"未知速度规划类型: {profile}，回退至 scurve")
        s_curve_dist, _ = s_curve_profile(dists[-1], total_time, dt)

    xs = np.interp(s_curve_dist, dists, [p[0] for p in path])
    ys = np.interp(s_curve_dist, dists, [p[1] for p in path])
    zs = np.interp(s_curve_dist, dists, [p[2] for p in path])

    rxs = np.interp(s_curve_dist, dists, [p[3] for p in path])
    rys = np.interp(s_curve_dist, dists, [p[4] for p in path])
    rzs = np.interp(s_curve_dist, dists, [p[5] for p in path])

    return [[float(xs[i]), float(ys[i]), float(zs[i]),float(rxs[i]), float(rys[i]), float(rzs[i])]
            for i in range(len(xs))]

def load_trajectory(filepath: str) -> List[List[float]]:
    """
    从 txt 文件加载轨迹点，自动识别空格/逗号分隔，跳过空行

    Args:
        filepath: 文件路径，6列格式: x y z rx ry rz

    Returns:
        List[List[float]]
    """
    trajectory = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 6:
                trajectory.append([float(v) for v in parts[:6]])
    logger.info(f"从 {filepath} 加载 {len(trajectory)} 个轨迹点")
    return trajectory


def save_trajectory(trajectory: List[List[float]], filepath: str,
                    separator: str = ' ') -> None:
    """
    将轨迹点保存到 txt 文件

    Args:
        trajectory: 轨迹点 [[x,y,z,rx,ry,rz], ...]
        filepath:   保存路径
        separator:  分隔符，默认空格
    """
    with open(filepath, 'w') as f:
        for p in trajectory:
            f.write(f"{p[0]:.6f}{separator}{p[1]:.6f}{separator}"
                    f"{p[2]:.6f}{separator}{p[3]:.6f}{separator}"
                    f"{p[4]:.6f}{separator}{p[5]:.6f}\n")
    logger.info(f"保存 {len(trajectory)} 个轨迹点至 {filepath}")

class CRobot(RobotBase):
    """机械臂控制类"""
    
    def __init__(self, ip: str = "192.168.1.12", port: int = 502, unit_id: int = 1):
        super().__init__()
        
        self.ip = ip
        self.port = port
        self.unit_id = unit_id
        self.is_connected_flag = False
        
        logger.info(f"初始化机械臂控制器: IP={ip}, Port={port}, UnitID={unit_id}")
        
        self.a_list = [0, 621.620, 559.133, 0, 0, 0]
        # self.d_list = [0, 0, 0, -160.949, 119.425, 115.0] 0522
        self.d_list = [0, 0, 0, -164.261, 119.327, 115.0]
        self.alpha_deg_list = [90, 0, 0, 90, 90, 0]
        self.offset_deg_list = [0, 0, -90, 90, -90, 0]
        
        try:
            self.sdk = crobotsdk.CRobotSDK(crobotsdk._get_sdk_lib_path())
            self.model_service = self.sdk.getModelService()
            self.file_service = self.sdk.getFileService()
            self.param_service = self.sdk.getModelService()
            self.motion_service = self.sdk.getMotionService()
            self.ut_service = self.sdk.getUTService()
            self.robot_service = self.sdk.getRobotService()
            logger.info("SDK服务初始化成功")
        except Exception as e:
            logger.error(f"SDK服务初始化失败: {e}")
            raise
        
    def connect(self) -> bool:
        try:
            logger.info(f"尝试连接到机械臂 {self.ip}:{self.port}")
            connection = self.robot_service.connect(self.ip)
            if not connection:
                logger.error(f"无法连接到机械臂 {self.ip}:{self.port}")
                self.is_connected_flag = False
                return False
            
            self.is_connected_flag = True
            logger.info(f"成功连接到机械臂 {self.ip}:{self.port} (unit_id={self.unit_id})")
            return True
        except Exception as e:
            logger.error(f"连接异常: {e}")
            self.is_connected_flag = False
            return False

    def disconnect(self) -> None:
        try:
            logger.info("正在断开机械臂连接")
            self.robot_service.disconnect()
            self.is_connected_flag = False
            logger.info("机械臂连接已断开")
        except Exception as e:
            logger.error(f"断开连接时发生异常: {e}")

    def get_tcp_pose(self) -> Optional[List[float]]:
        try:
            pose = self.robot_service.get_current_position(1, 6)
            if not pose:
                logger.warning("获取TCP位姿失败，返回空值")
            return pose
        except Exception as e:
            logger.error(f"获取TCP位姿时发生异常: {e}")
            return None
    
    def get_joint_pose(self) -> Optional[List[float]]:
        try:
            joint_pose = self.robot_service.get_current_joint().body
            if joint_pose:
                joint_pose = [round(angle, 3) for angle in joint_pose]
            else:
                logger.warning("获取关节角度失败，返回空值")
            return joint_pose
        except Exception as e:
            logger.error(f"获取关节角度时发生异常: {e}")
            return None
        
    def get_current_coord(self) -> Any:
        try:
            return self.robot_service.get_coord_sys()
        except Exception as e:
            logger.error(f"获取当前坐标系时发生异常: {e}")
            return None

    def is_moving(self) -> bool:
        try:
            return self.robot_service.is_moving()
        except Exception as e:
            logger.error(f"检查运动状态时发生异常: {e}")
            return False
    
    def is_connected(self) -> bool:
        try:
            connected = self.robot_service.is_connected()
            return connected
        except Exception as e:
            logger.error(f"检查连接状态时发生异常: {e}")
            return self.is_connected_flag

    def move_linear(self, target: CartesianPose, speed: int = 100, start:bool = True,end: bool = True) -> int:
        logger.info(f"执行直线运动到目标: {target}, 速度: {speed}")
        
        try:
            if start:
                self.robot_service.set_work_mode(RobotMode.PLAYING)
                
                if self.robot_service.has_error():
                    if self.robot_service.has_emergency_error():
                        logger.error("请手动清除紧急错误")
                        return 0   
                    logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                    self.robot_service.clear_error()
                
                if not self.robot_service.is_servo_on():
                    logger.info("伺服未上电，正在上电")
                    power = self.robot_service.servo_power_on()
                    if not power:
                        logger.error("伺服上电失败")
                        return 0
                    logger.info("伺服上电成功")
            
            inst_movel = InstMoveL()
            inst_movel.target_pos.x = target.x
            inst_movel.target_pos.y = target.y
            inst_movel.target_pos.z = target.z
            inst_movel.target_pos.rx = target.rx
            inst_movel.target_pos.ry = target.ry
            inst_movel.target_pos.rz = target.rz
            inst_movel.strategy = MoveStrategy.DISTANCE_FIRST
            inst_movel.param.acc = 1
            inst_movel.param.dec = 1
            inst_movel.param.pl = 1
            inst_movel.param.speed = speed
            
            p_status = self.robot_service.get_program_status()
            if p_status == ProgramStatus.STOP:
                if not self.robot_service.start_program("guidanceInst.pro", 0):
                    logger.error("=====启动程序失败=====")
                    return 0
            elif p_status == ProgramStatus.PAUSE:
                if not self.robot_service.resume_program("guidanceInst.pro"):
                    logger.error("=====恢复程序失败=====")
                    return 0
            else:
                logger.info("=====程序正在运行=====")
            
            while not self.motion_service.is_ready(MotionType.INSTRUCTION):
                time.sleep(5)
            
            self.motion_service.move_l(0, inst_movel)
            # time.sleep(0.1)
            if end:
                self.motion_service.finalize(MotionType.INSTRUCTION)
            
            while self.robot_service.is_moving():
                logger.info("机械臂正在运动...")
                time.sleep(1)
            
            logger.info("直线运动完成")
            return 1
            
        except Exception as e:
            logger.error(f"直线运动时发生异常: {e}")
            return 0
    
    def move_circular(self,mid:CartesianPose,target:CartesianPose,speed:float,seq:int = 0,acc:int = 1 ,dec:int = 1,pl:int = 0,start:bool = False,end:bool=False):
        logger.info(f"执行圆弧运动到目标:中点{mid}，终点{target}, 速度: {speed}")
        try:
            if start:
                self.robot_service.set_work_mode(RobotMode.REMOTE)
                
                if self.robot_service.has_error():
                    if self.robot_service.has_emergency_error():
                        logger.error("请手动清除紧急错误")
                        return 0   
                    logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                    self.robot_service.clear_error()
                
                if not self.robot_service.is_servo_on():
                    logger.info("伺服未上电，正在上电")
                    power = self.robot_service.servo_power_on()
                    if not power:
                        logger.error("伺服上电失败")
                        return 0
                    logger.info("伺服上电成功")
            inst_movec = InstMoveC()
            inst_movec.p2.x = mid.x
            inst_movec.p2.y = mid.y
            inst_movec.p2.z = mid.z
            inst_movec.p2.rx = mid.rx
            inst_movec.p2.ry = mid.ry
            inst_movec.p2.rz = mid.rz
            inst_movec.p3.x = target.x
            inst_movec.p3.y = target.y
            inst_movec.p3.z = target.z
            inst_movec.p3.rx = target.rx
            inst_movec.p3.ry = target.ry
            inst_movec.p3.rz = target.rz
            inst_movec.strategy = MoveStrategy.DISTANCE_FIRST
            inst_movec.param.acc = acc
            inst_movec.param.dec = dec
            inst_movec.param.pl = pl
            inst_movec.param.speed = speed

            p_status = self.robot_service.get_program_status()
            if p_status == ProgramStatus.STOP:
                if not self.robot_service.start_program("guidanceInst.pro", 0):
                    logger.error("=====启动程序失败=====")
                    return 0
            elif p_status == ProgramStatus.PAUSE:
                if not self.robot_service.resume_program("guidanceInst.pro"):
                    logger.error("=====恢复程序失败=====")
                    return 0
            else:
                logger.info("=====程序正在运行=====")
            
            while not self.motion_service.is_ready(MotionType.INSTRUCTION):
                time.sleep(0.1)
            
            self.motion_service.move_c(seq, inst_movec)
            time.sleep(0.1)
            if end:
                self.motion_service.finalize(MotionType.INSTRUCTION)
            
            while self.robot_service.is_moving():
                logger.info("机械臂正在运动...")
                time.sleep(1)
            
            logger.info("圆弧运动完成")
            return 1

        except Exception as e:
            logger.error(f"圆弧运动时发生异常: {e}")
            return 0
    
    def move_circular_session(self,mid:CartesianPose,target:CartesianPose,speed:float,seq:int = 0,acc:int = 1 ,dec:int = 1,pl:int = 1):
        logger.info(f"执行圆弧运动到目标:中点{mid}，终点{target}, 速度: {speed}")
        try:
            inst_movec = InstMoveC()
            inst_movec.p2.x = mid.x
            inst_movec.p2.y = mid.y
            inst_movec.p2.z = mid.z
            inst_movec.p2.rx = mid.rx
            inst_movec.p2.ry = mid.ry
            inst_movec.p2.rz = mid.rz
            inst_movec.p3.x = target.x
            inst_movec.p3.y = target.y
            inst_movec.p3.z = target.z
            inst_movec.p3.rx = target.rx
            inst_movec.p3.ry = target.ry
            inst_movec.p3.rz = target.rz
            inst_movec.strategy = MoveStrategy.TIME_FIRST
            inst_movec.param.acc = acc
            inst_movec.param.dec = dec
            inst_movec.param.pl = pl
            inst_movec.param.speed = speed

            self.motion_service.move_c(seq, inst_movec)
        except Exception as e:
            logger.error(f"圆弧运动时发生异常: {e}")
            return 0

    
    def move_joint(self, joint: List[float], speed: int = 100, start: bool = True, end: bool = True) -> int:
        if len(joint) != 6:
            logger.error(f"关节列表长度必须为6，当前长度: {len(joint)}")
            return 0
        
        logger.info(f"执行关节运动到目标: {joint}, 速度: {speed}")
        
        try:
            if start:
                self.robot_service.set_work_mode(RobotMode.PLAYING)
                
                if self.robot_service.has_error():
                    if self.robot_service.has_emergency_error():
                        logger.error("请手动清除紧急错误")
                        return 0    
                    logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                    self.robot_service.clear_error()
                
                if not self.robot_service.is_servo_on():
                    logger.info("伺服未上电，正在上电")
                    power = self.robot_service.servo_power_on()
                    if not power:
                        logger.error("伺服上电失败")
                        return 0
                    logger.info("伺服上电成功")
            
            inst_move = InstMoveAbsJ()
            inst_move.joint.body = joint
            inst_move.param.acc = 1
            inst_move.param.dec = 1
            inst_move.param.pl = 1
            inst_move.param.speed = speed

            p_status = self.robot_service.get_program_status()
            if p_status == ProgramStatus.STOP:
                if not self.robot_service.start_program("guidanceInst.pro", 0):
                    logger.error("=====启动程序失败=====")
                    return 0
            elif p_status == ProgramStatus.PAUSE:
                if not self.robot_service.resume_program("guidanceInst.pro"):
                    logger.error("=====恢复程序失败=====")
                    return 0
            else:
                logger.info("=====程序正在运行=====")
            
            
            while not self.motion_service.is_ready(MotionType.INSTRUCTION):
                time.sleep(5)
            
            
            self.motion_service.move_abs_j(0, inst_move)

            if end:
                self.motion_service.finalize(MotionType.INSTRUCTION)

            
            while self.robot_service.is_moving():
                # logger.info("机械臂正在运动...")
                # time.sleep(1)
                pass
            
            logger.info("关节运动完成")
            return 1
            
        except Exception as e:
            logger.error(f"关节运动时发生异常: {e}")
            return 0
        
    # 连续直线/关节运动
    def move_by_pose_list(self, poses: List[List[float]], speeds: List[int],end: bool = True) -> int:
        """
        按笛卡尔位姿列表连续运动（直线运动，点之间不停顿）
        
        Args:
            poses: 位姿点列表，每个元素为长度为6的列表 [x, y, z, rx, ry, rz]
            speeds: 速度列表，每个元素为速度值，与poses一一对应
        
        Returns:
            int: 成功返回1，失败返回0
        """
        if not poses:
            logger.error("位姿点列表为空")
            return 0
        
        if len(poses) != len(speeds):
            logger.error(f"位姿数量({len(poses)})与速度数量({len(speeds)})不匹配")
            return 0
        
        # 验证每个点的参数
        for idx, pose in enumerate(poses):
            if len(pose) != 6:
                logger.error(f"第{idx}个位姿参数长度必须为6，当前长度: {len(pose)}，需要格式: [x,y,z,rx,ry,rz]")
                return 0
            if not isinstance(speeds[idx], int) or speeds[idx] <= 0:
                logger.error(f"第{idx}个速度必须是正整数，当前: {speeds[idx]}")
                return 0
        
        logger.info(f"开始执行直线运动序列，共{len(poses)}个点")
        
        try:
            # 设置工作模式
            self.robot_service.set_work_mode(RobotMode.PLAYING)
            
            # 检查紧急错误
            if self.robot_service.has_error():
                if self.robot_service.has_emergency_error():
                    logger.error("请手动清除紧急错误")
                    return 0   
                logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                self.robot_service.clear_error()
            
            # 检查并上电
            if not self.robot_service.is_servo_on():
                logger.info("伺服未上电，正在上电")
                power = self.robot_service.servo_power_on()
                if not power:
                    logger.error("伺服上电失败")
                    return 0
                logger.info("伺服上电成功")
            
            # 启动程序
            p_status = self.robot_service.get_program_status()
            if p_status == ProgramStatus.STOP:
                if not self.robot_service.start_program("guidanceInst.pro", 0):
                    logger.error("=====启动程序失败=====")
                    return 0
            elif p_status == ProgramStatus.PAUSE:
                if not self.robot_service.resume_program("guidanceInst.pro"):
                    logger.error("=====恢复程序失败=====")
                    return 0
            else:
                logger.info("=====程序正在运行=====")
            
            
            # 等待运动服务就绪
            wait_count = 0
            while not self.motion_service.is_ready(MotionType.INSTRUCTION):
                time.sleep(0.1)
                wait_count += 1
                if wait_count > 50:  # 5秒超时
                    logger.error("等待运动服务就绪超时")
                    return 0
            
            # 按顺序连续执行直线运动（不等待，不停顿）
            for idx, pose in enumerate(poses):
                x, y, z, rx, ry, rz = pose
                speed = speeds[idx]
                
                logger.info(f"发送第{idx + 1}/{len(poses)}个直线运动: 位置=({x},{y},{z}), 姿态=({rx},{ry},{rz}), 速度={speed}")
                
                inst_movel = InstMoveL()
                inst_movel.target_pos.x = x
                inst_movel.target_pos.y = y
                inst_movel.target_pos.z = z
                inst_movel.target_pos.rx = rx
                inst_movel.target_pos.ry = ry
                inst_movel.target_pos.rz = rz
                inst_movel.strategy = MoveStrategy.DISTANCE_FIRST
                inst_movel.param.acc = 1
                inst_movel.param.dec = 1
                inst_movel.param.pl = 0
                inst_movel.param.speed = speed
                
                # 直接发送运动指令，不等待，不sleep
                self.motion_service.move_l(idx, inst_movel)

            
            # 完成所有运动
            if end:
                self.motion_service.finalize(MotionType.INSTRUCTION)
            
            # 等待所有运动完成
            while self.robot_service.is_moving():
                logger.info("机械臂正在执行直线运动序列...")
                time.sleep(0.5)
            
            logger.info("直线运动序列完成")
            return 1
            
        except Exception as e:
            logger.error(f"直线运动序列执行异常: {e}")
            return 0

    def move_by_joint_list(self, joints: List[List[float]], speeds: List[int]) -> int:
        """
        按关节角列表连续运动（点之间不停顿）
        
        Args:
            joints: 关节角点列表，每个元素为长度为6的列表 [j1, j2, j3, j4, j5, j6]
            speeds: 速度列表，每个元素为速度值，与joints一一对应
        
        Returns:
            int: 成功返回1，失败返回0
        """
        if not joints:
            logger.error("关节角点列表为空")
            return 0
        
        if len(joints) != len(speeds):
            logger.error(f"关节角数量({len(joints)})与速度数量({len(speeds)})不匹配")
            return 0
        
        # 验证每个点的参数
        for idx, joint in enumerate(joints):
            if len(joint) != 6:
                logger.error(f"第{idx}个关节角参数长度必须为6，当前长度: {len(joint)}，需要格式: [j1,j2,j3,j4,j5,j6]")
                return 0
            if not isinstance(speeds[idx], int) or speeds[idx] <= 0:
                logger.error(f"第{idx}个速度必须是正整数，当前: {speeds[idx]}")
                return 0
        
        logger.info(f"开始执行关节运动序列，共{len(joints)}个点")
        
        try:
            # 设置工作模式
            self.robot_service.set_work_mode(RobotMode.PLAYING)
            
            # 检查紧急错误
            if self.robot_service.has_error():
                if self.robot_service.has_emergency_error():
                    logger.error("请手动清除紧急错误")
                    return 0    
                logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                self.robot_service.clear_error()
            
            # 检查并上电
            if not self.robot_service.is_servo_on():
                logger.info("伺服未上电，正在上电")
                power = self.robot_service.servo_power_on()
                if not power:
                    logger.error("伺服上电失败")
                    return 0
                logger.info("伺服上电成功")
            
            # 启动程序
            p_status = self.robot_service.get_program_status()
            if p_status == ProgramStatus.STOP:
                if not self.robot_service.start_program("guidanceInst.pro", 0):
                    logger.error("启动程序失败")
                    return 0
            elif p_status == ProgramStatus.PAUSE:
                if not self.robot_service.resume_program("guidanceInst.pro"):
                    logger.error("恢复程序失败")
                    return 0
            else:
                logger.info("=====程序正在运行=====")
            

            # 等待运动服务就绪
            wait_count = 0
            while not self.motion_service.is_ready(MotionType.INSTRUCTION):
                time.sleep(0.1)
                wait_count += 1
                if wait_count > 100:  # 5秒超时
                    logger.error("等待运动服务就绪超时")
                    return 0
            
            # 按顺序连续执行关节运动（不等待，不停顿）
            for idx, joint in enumerate(joints):
                speed = speeds[idx]
                
                logger.info(f"发送第{idx + 1}/{len(joints)}个关节运动: 关节={joint}, 速度={speed}")
                
                inst_move = InstMoveAbsJ()
                inst_move.joint.body = joint
                inst_move.param.acc = 1
                inst_move.param.dec = 1
                inst_move.param.pl = 0
                inst_move.param.speed = speed
                
                # 直接发送运动指令，不等待，不sleep
                self.motion_service.move_abs_j(idx, inst_move)
            
            # 完成所有运动
            self.motion_service.finalize(MotionType.INSTRUCTION)
            
            # 等待所有运动完成
            while self.robot_service.is_moving():
                logger.info("机械臂正在执行关节运动序列...")
                time.sleep(0.5)
            
            logger.info("关节运动序列完成")
            return 1
            
        except Exception as e:
            logger.error(f"关节运动序列执行异常: {e}")
            return 0

    def move_by_joint_path(self, joints: List[List[float]], ratio: int = 1,
                           start: bool = True, end: bool = True) -> int:
        """
        按关节角路径连续运动（路径模式，区别于指令模式的 move_by_joint_list）

        Args:
            joints: 关节角点列表，每个元素为长度为6的列表 [j1, j2, j3, j4, j5, j6]
            ratio:  插补周期倍率 (movePath参数)，插补周期 = 2ms × ratio，取值范围 1~50
            start:  是否在运动前执行初始化（上电/清错/启动程序 guidancePos.pro）
            end:    是否在运动后执行 finalize

        Returns:
            int: 成功返回1，失败返回0
        """
        if ratio < 1 or ratio > 50:
            logger.error(f"ratio 取值范围 1~50，当前值: {ratio}")
            return 0

        if not joints:
            logger.error("关节角点列表为空")
            return 0

        for idx, joint in enumerate(joints):
            if len(joint) != 6:
                logger.error(f"第{idx}个关节角参数长度必须为6，当前长度: {len(joint)}，需要格式: [j1,j2,j3,j4,j5,j6]")
                return 0

        logger.info(f"开始执行关节路径运动，共{len(joints)}个点, ratio={ratio}")

        try:
            if start:
                self.robot_service.set_work_mode(RobotMode.PLAYING)

                if self.robot_service.has_error():
                    if self.robot_service.has_emergency_error():
                        logger.error("请手动清除紧急错误")
                        return 0
                    logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                    self.robot_service.clear_error()

                if not self.robot_service.is_servo_on():
                    logger.info("伺服未上电，正在上电")
                    power = self.robot_service.servo_power_on()
                    if not power:
                        logger.error("伺服上电失败")
                        return 0
                    logger.info("伺服上电成功")

                p_status = self.robot_service.get_program_status()
                if p_status == ProgramStatus.STOP:
                    if not self.robot_service.start_program("guidancePos.pro", 0):
                        logger.error("启动程序失败")
                        return 0
                elif p_status == ProgramStatus.PAUSE:
                    if not self.robot_service.resume_program("guidancePos.pro"):
                        logger.error("恢复程序失败")
                        return 0
                else:
                    logger.info("=====程序正在运行=====")

            wait_count = 0
            while not self.motion_service.is_ready(MotionType.PATH):
                time.sleep(0.1)
                wait_count += 1
                if wait_count > 100:
                    logger.error("等待路径运动服务就绪超时")
                    return 0

            joint_positions = []
            for joint in joints:
                jp = JointPosition()
                jp.body = joint
                jp.cfg = [0, 0, 0, 0]
                jp.ext = [0, 0, 0, 0, 0, 0]
                joint_positions.append(jp)

            self.motion_service.send_joint_path(joint_positions)

            result = self.motion_service.move_path(ratio)
            if result != MovePathResult.SUCCESS:
                logger.error(f"启动路径运动失败: {result}")
                return 0
            logger.info("关节路径运动启动成功")

            if end:
                self.motion_service.finalize(MotionType.PATH)

            while self.robot_service.is_moving():
                logger.info("机械臂正在执行关节路径运动...")
                time.sleep(0.5)

            logger.info("关节路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"关节路径运动执行异常: {e}")
            return 0

    def move_by_position_path(self, positions: List[List[float]], ratio: int = 1,
                               tool_no: int = 10, user_no: int = 0,
                               start: bool = True, end: bool = True) -> int:
        """
        按笛卡尔位姿路径连续运动（路径模式，区别于指令模式的 move_by_pose_list）

        Args:
            positions: 位姿点列表，每个元素为 [x, y, z, rx, ry, rz]
            ratio:     插补周期倍率 (movePath参数)，插补周期 = 2ms × ratio，取值范围 1~50
            tool_no:   工具坐标系编号（对应C++中的 toolNo=10）
            user_no:   用户坐标系编号（对应C++中的 userNo=0）
            start:     是否在运动前执行初始化
            end:       是否在运动后执行 finalize

        Returns:
            int: 成功返回1，失败返回0
        """
        if ratio < 1 or ratio > 50:
            logger.error(f"ratio 取值范围 1~50，当前值: {ratio}")
            return 0

        if not positions:
            logger.error("位姿点列表为空")
            return 0

        for idx, pos in enumerate(positions):
            if len(pos) != 6:
                logger.error(f"第{idx}个位姿参数长度必须为6，当前长度: {len(pos)}，需要格式: [x,y,z,rx,ry,rz]")
                return 0

        logger.info(f"开始执行笛卡尔路径运动，共{len(positions)}个点, ratio={ratio}")

        try:
            if start:
                self.robot_service.set_work_mode(RobotMode.PLAYING)

                if self.robot_service.has_error():
                    if self.robot_service.has_emergency_error():
                        logger.error("请手动清除紧急错误")
                        return 0
                    logger.warning(f"检测到错误，尝试清除: {self.robot_service.get_error_message(0)}")
                    self.robot_service.clear_error()

                if not self.robot_service.is_servo_on():
                    logger.info("伺服未上电，正在上电")
                    power = self.robot_service.servo_power_on()
                    if not power:
                        logger.error("伺服上电失败")
                        return 0
                    logger.info("伺服上电成功")

                p_status = self.robot_service.get_program_status()
                if p_status == ProgramStatus.STOP:
                    if not self.robot_service.start_program("guidancePos.pro", 0):
                        logger.error("启动程序失败")
                        return 0
                elif p_status == ProgramStatus.PAUSE:
                    if not self.robot_service.resume_program("guidancePos.pro"):
                        logger.error("恢复程序失败")
                        return 0
                else:
                    logger.info("=====程序正在运行=====")

            wait_count = 0
            while not self.motion_service.is_ready(MotionType.PATH):
                time.sleep(0.1)
                wait_count += 1
                if wait_count > 100:
                    logger.error("等待路径运动服务就绪超时")
                    return 0

            robot_positions = []
            for pos in positions:
                rp = RobotPosition()
                rp.x = pos[0]
                rp.y = pos[1]
                rp.z = pos[2]
                rp.rx = pos[3]
                rp.ry = pos[4]
                rp.rz = pos[5]
                rp.cfg = [0, 0, 0, 0]
                rp.ext_joint = [0, 0, 0, 0, 0, 0]
                robot_positions.append(rp)

            self.motion_service.send_cartesian_path(robot_positions, tool_no, user_no)
            print("正在发送点位")
            print("当前缓存区大小：",self.motion_service.get_avail_path_buffer_size())
            # time.sleep(2)

            result = self.motion_service.move_path(ratio)
            if result != MovePathResult.SUCCESS:
                logger.error(f"启动路径运动失败: {result}")
                return 0
            logger.info("笛卡尔路径运动启动成功")
            self.motion_service.finalize(MotionType.PATH)
            
            while self.motion_service.get_avail_path_buffer_size() < 2047:
                time.sleep(0.1)
            self.motion_service.finalize(MotionType.PATH)
            time.sleep(0.1)
            while self.robot_service.is_moving():
                logger.info("机械臂正在执行笛卡尔路径运动...")
                print("实时缓存区大小：",self.motion_service.get_avail_path_buffer_size())
                time.sleep(0.5)
            self.motion_service.finalize(MotionType.PATH)
            # self.robot_service.stop_program()
            
            # self.robot_service.set_work_mode(RobotMode.MANUAL)
            # self.robot_service.stop_program()
            # self.robot_service.set_work_mode(RobotMode.PLAYING)
            # # time.sleep(1)
            # self.robot_service.stop_program()
            # p_status = self.robot_service.get_program_status()
            # if p_status == ProgramStatus.STOP:
            #     if not self.robot_service.start_program("guidanceInst.pro", 0):
            #         logger.error("启动程序失败")
            #         return 0
            # elif p_status == ProgramStatus.PAUSE:
            #     if not self.robot_service.resume_program("guidanceInst.pro"):
            #         logger.error("恢复程序失败")
            #         return 0
            # else:
            #     logger.info("=====程序正在运行=====")
        # self.robot_service.clear_error()

            logger.info("笛卡尔路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"笛卡尔路径运动执行异常: {e}")
            return 0

    def plan_cartesian_traj(self, waypoints: List[List[float]],
                            total_time: float, dt: float = 0.008,
                            profile: str = 'scurve',
                            accel_frac: float = 0.25) -> List[List[float]]:
        """
        对笛卡尔路点做时间参数化轨迹规划 (S曲线 / 梯形)

        Args:
            waypoints:  笛卡尔路径点 [[x,y,z,rx,ry,rz], ...]，需 >= 2 个点
            total_time: 目标运动总时间 (s)
            dt:         插补周期 (s)，默认 8ms
            profile:    速度规划类型: 'scurve' 正弦S曲线 / 'trapezoid' 梯形
            accel_frac: 梯形规划加速段占比 (仅 profile='trapezoid' 时有效)

        Returns:
            List[List[float]]  时间参数化轨迹，点数 ≈ total_time / dt，
                               可直接传给 move_by_position_path()
        """
        return smooth_cartesian_traj(waypoints, total_time, dt,
                                     profile=profile, accel_frac=accel_frac)

    def cartesian_to_joint_traj(self, cartesian_traj: List[List[float]],
                                initial_joints: Optional[List[float]] = None
                                ) -> List[List[float]]:
        """
        将笛卡尔轨迹逐点逆解为关节角轨迹
        每点 IK 以前一点结果为初始值，保证相邻点逆解连续

        Args:
            cartesian_traj: 笛卡尔轨迹 [[x,y,z,rx,ry,rz], ...]
            initial_joints: IK 初始关节角，None 则取当前关节角

        Returns:
            List[List[float]]  关节角轨迹，可传给 move_by_joint_path()
        """
        if not cartesian_traj:
            logger.error("笛卡尔轨迹为空")
            return []

        ik_ref = initial_joints
        if ik_ref is None:
            ik_ref = self.get_joint_pose()
            if ik_ref is None:
                logger.error("无法获取当前关节角作为 IK 初始值")
                return []

        joint_traj = []
        for i, pose in enumerate(cartesian_traj):
            result = self.inverse_kinematics(pose, ik_ref)
            if result is None or len(result) != 6:
                logger.error(f"第{i}个点逆解失败: {pose}")
                return []
            joint_traj.append(result)
            ik_ref = result

        logger.info(f"逆解完成，{len(joint_traj)}个关节角点")
        return joint_traj

    def plan_and_move_position(self, waypoints: List[List[float]],
                                total_time: float, dt: float = 0.008,
                                tool_no: int = 10, user_no: int = 0,
                                start: bool = True, end: bool = True,
                                profile: str = 'scurve',
                                accel_frac: float = 0.25) -> int:
        """
        规划笛卡尔轨迹 → 自动执行笛卡尔路径运动

        Args:
            waypoints:  笛卡尔路径点
            total_time: 目标运动总时间 (s)
            dt:         插补周期 (s)，默认 8ms → movePath ratio = dt / 0.002
            tool_no:    工具坐标系编号
            user_no:    用户坐标系编号
            start:      是否执行运动前初始化
            end:        是否执行 finalize
            profile:    速度规划类型: 'scurve' 正弦S曲线 / 'trapezoid' 梯形
            accel_frac: 梯形规划加速段占比 (仅 profile='trapezoid' 时有效)

        Returns:
            int: 成功返回1，失败返回0
        """
        traj = self.plan_cartesian_traj(waypoints, total_time, dt,
                                        profile=profile, accel_frac=accel_frac)
        if not traj:
            return 0
        ratio = int(dt / 0.002)
        return self.move_by_position_path(traj, ratio=ratio,
                                           tool_no=tool_no, user_no=user_no,
                                           start=start, end=end)

    def plan_and_move_joint(self, waypoints: List[List[float]],
                             total_time: float, dt: float = 0.008,
                             initial_joints: Optional[List[float]] = None,
                             start: bool = True, end: bool = True) -> int:
        """
        规划笛卡尔轨迹 → 逆解关节角 → 自动执行关节路径运动

        Args:
            waypoints:      笛卡尔路径点
            total_time:     目标运动总时间 (s)
            dt:             插补周期 (s)，默认 8ms → movePath ratio = dt / 0.002
            initial_joints: IK 初始关节角，None 则取当前关节角
            start:          是否执行运动前初始化
            end:            是否执行 finalize

        Returns:
            int: 成功返回1，失败返回0
        """
        traj = self.plan_cartesian_traj(waypoints, total_time, dt)
        if not traj:
            return 0
        joint_traj = self.cartesian_to_joint_traj(traj, initial_joints)
        if not joint_traj:
            return 0
        ratio = int(dt / 0.002)
        return self.move_by_joint_path(joint_traj, ratio=ratio,
                                        start=start, end=end)
    
    def switch_motion_model(self):
            self.robot_service.stop_program()
            self.robot_service.set_work_mode(RobotMode.MANUAL)
            self.robot_service.stop_program()

    def move_and_wait(self, target, timeout: float = 30) -> bool:
        logger.info(f"移动并等待，超时时间: {timeout}秒")
        try:
            start_time = time.time()
            if self.move_linear(target, start=True, end=True) != 1:
                logger.error("发起运动失败")
                return False
            while self.is_moving():
                if time.time() - start_time > timeout:
                    logger.warning(f"运动超时 ({timeout}s)")
                    return False
                time.sleep(0.1)
            logger.info("运动完成")
            return True
        except Exception as e:
            logger.error(f"move_and_wait 异常: {e}")
            return False
        
    def move_joint_and_wait(self, target,speed, timeout: float = 30) -> bool:
        logger.info(f"移动并等待，超时时间: {timeout}秒")
        try:
            start_time = time.time()
            ik_ref_joint = self.get_joint_pose()
            joint_pose = self.inverse_kinematics(target.to_list(), ik_ref_joint)
            if self.move_by_joint_list(joints=[joint_pose], speeds=[speed]) != 1:
                logger.error("发起运动失败")
                return False
            while self.is_moving():
                if time.time() - start_time > timeout:
                    logger.warning(f"运动超时 ({timeout}s)")
                    return False
                time.sleep(0.1)
            logger.info("运动完成")
            return True
        except Exception as e:
            logger.error(f"move_and_wait 异常: {e}")
            return False
    
    def set_speed(self, speed_pct: int) -> bool:
        if not 0 <= speed_pct <= 100:
            logger.error(f"速度百分比必须在0-100之间，当前值: {speed_pct}")
            return False
        
        try:
            logger.info(f"设置运动速度: {speed_pct}%")
            result = self.robot_service.set_speed_ratio(speed_pct)
            if result:
                logger.info(f"速度设置成功: {speed_pct}%")
            else:
                logger.error(f"速度设置失败: {speed_pct}%")
            return result
        except Exception as e:
            logger.error(f"设置速度时发生异常: {e}")
            return False
            
    def move_relative_tool(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                          drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
                          mode: str = 'linear', speed: int = 100, start: bool = False, end: bool = False) -> int:
        logger.info(f"执行工具坐标系相对移动: dX={dx}, dY={dy}, dZ={dz}, dRX={drx}, dRY={dry}, dRZ={drz}")
        
        try:
            current_pose = self.get_tcp_pose()
            if current_pose is None:
                logger.error("无法获取当前位姿，停止相对移动")
                return -1
            
            current_carpose = CartesianPose()
            current_carpose.from_list(current_pose)
            
            T_base_tcp = pose_to_homogeneous_matrix(current_carpose, degrees=True)
            delta_pose = CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
            T_tool_delta = pose_to_homogeneous_matrix(delta_pose, degrees=True)
            T_target_base = np.dot(T_base_tcp, T_tool_delta)
            target_pose_base = homogeneous_matrix_to_pose(T_target_base, degrees=True)
            
            if mode == 'linear':
                res = self.move_linear(target_pose_base, speed, start, end)
            elif mode == 'joint':
                joints = self.inverse_kinematics(target_pose_base.to_list())
                res = self.move_joint(joints, speed, start, end)
            else:
                logger.error(f"不支持的运动模式: {mode}")
                return -1
                
            
            if res != 1:
                logger.error("相对移动失败")
                return -1
            
            logger.info("工具坐标系相对移动完成")
            return 0
            
        except Exception as e:
            logger.error(f"相对移动时发生异常: {e}")
            return -1

    def move_relative_tool_v1(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                          drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
                          mode: str = 'linear', speed: int = 100, wait: bool = True) -> int:
        logger.info(f"执行工具坐标系相对移动: dX={dx}, dY={dy}, dZ={dz}, dRX={drx}, dRY={dry}, dRZ={drz}")
        
      
        current_pose = self.get_tcp_pose()
        if current_pose is None:
            logger.error("无法获取当前位姿，停止相对移动")
            return -1
        
        current_carpose = CartesianPose()
        current_carpose.from_list(current_pose)
        
        T_base_tcp = pose_to_homogeneous_matrix(current_carpose, degrees=True)
        delta_pose = CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
        T_tool_delta = pose_to_homogeneous_matrix(delta_pose, degrees=True)
        T_target_base = np.dot(T_base_tcp, T_tool_delta)
        target_pose_base = homogeneous_matrix_to_pose(T_target_base, degrees=True)

        return target_pose_base  

    # 法兰位姿->世界位姿
    def relative_tool_pose(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                          drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,init_pose: list = [0, 0, 0, 0, 0, 0],
                          mode: str = 'linear', speed: int = 100, wait: bool = True) :
        logger.info(f"执行工具坐标系相对移动: dX={dx}, dY={dy}, dZ={dz}, dRX={drx}, dRY={dry}, dRZ={drz}")
        
        try:
            # current_pose = self.get_tcp_pose()
            if init_pose is None:
                logger.error("无法获取当前位姿，停止相对移动")
                return -1
            
            current_carpose = init_pose
            # print('++++++++inint_pose ', init_pose)
            current_carpose = CartesianPose(current_carpose[0], current_carpose[1],current_carpose[2], current_carpose[3],current_carpose[4],current_carpose[5])
            
            T_base_tcp = pose_to_homogeneous_matrix(current_carpose, degrees=True)
            delta_pose = CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
            T_tool_delta = pose_to_homogeneous_matrix(delta_pose, degrees=True)
            T_target_base = np.dot(T_base_tcp, T_tool_delta)
            target_pose_base = homogeneous_matrix_to_pose(T_target_base, degrees=True)
            
            
            return target_pose_base
            
        except Exception as e:
            logger.error(f"相对移动时发生异常: {e}")
            return -1
        
    def move_joint_noservo(self,joint:List) -> bool:
        if(self.robot_service.get_work_mode!=0):
            self.robot_service.set_work_mode(RobotMode.MANUAL)
        movejoint = JointPosition()
        movejoint.body = joint
        movejoint.cfg = [0,0,0,0]
        movejoint.ext = [0,0,0,0,0,0]
        return self.robot_service.move_j(movejoint)
    def move_linear_noservo(self,pos:List) -> bool:
        if(self.robot_service.get_work_mode!=0):
            self.robot_service.set_work_mode(RobotMode.MANUAL)
        targetpos = RobotPosition()
        targetpos.x = pos[0]
        targetpos.y = pos[1]
        targetpos.z = pos[2]
        targetpos.rx = pos[3]
        targetpos.ry = pos[4]
        targetpos.rz = pos[5]
        targetpos.ext_joint = [0,0,0,0,0,0]
        targetpos.cfg = [0,0,0,0]
        
        
        return self.robot_service.move_l(targetpos)
        
    
    def calculate_pose_distance(self, pose1: CartesianPose, pose2: CartesianPose) -> float:
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dz = pose2.z - pose1.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        return distance
    
    def dh_transform(self, a: float, alpha_rad: float, d: float, theta_rad: float) -> np.ndarray:
        sa = np.sin(alpha_rad)
        ca = np.cos(alpha_rad)
        st = np.sin(theta_rad)
        ct = np.cos(theta_rad)
        
        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [ 0,       sa,      ca,      d],
            [ 0,        0,       0,      1]
        ])
        return T

    def forward_kinematics(self, joints: List[float], representation: str = 'euler'):
        if len(joints) != 6:
            logger.error(f"关节数量必须为6，当前: {len(joints)}")
            raise ValueError(f"关节数量必须为6，当前: {len(joints)}")
        
        N = len(joints)
        T = np.eye(4)
        for i in range(N):
            a = self.a_list[i]
            d = self.d_list[i]
            alpha = np.deg2rad(self.alpha_deg_list[i])
            offset = np.deg2rad(self.offset_deg_list[i])
            theta = np.deg2rad(joints[i]) + offset
            T = T @ self.dh_transform(a, alpha, d, theta)

        if representation == 'matrix':
            return T
        
        pos = T[:3, 3]
        rot = T[:3, :3]
        
        if representation == 'euler':
            rpy = R.from_matrix(rot).as_euler('ZYX', degrees=True)
            result = [pos[0], pos[1], pos[2], rpy[2], rpy[1], rpy[0]]
        elif representation == 'rotvec':
            rotvec = R.from_matrix(rot).as_rotvec(degrees=True)
            result = [pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2]]
        else:
            error_msg = f'不支持的表示方式: {representation}'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return result

    def inverse_kinematics(self, target_pose: List[float], initial_joints: Optional[List[float]] = None,
                          representation: str = 'euler', max_iter: int = 200, tol: float = 1e-6) -> List[float]:
        N = len(self.a_list)
        if initial_joints is None:
            # joints = np.zeros(N)
            joints = self.get_joint_pose()

        else:
            joints = np.array(initial_joints, dtype=float)

        target_pos = np.array(target_pose[:3])
        if representation == 'euler':
            rx, ry, rz = target_pose[3], target_pose[4], target_pose[5]
            target_R = R.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
        elif representation == 'rotvec':
            target_R = R.from_rotvec(target_pose[3:6], degrees=True).as_matrix()
        else:
            error_msg = f'不支持的表示方式: {representation}'
            logger.error(error_msg)
            raise ValueError(error_msg)

        for it in range(max_iter):
            T = np.eye(4)
            T_list = []
            for i in range(N):
                a = self.a_list[i]
                d = self.d_list[i]
                alpha = np.deg2rad(self.alpha_deg_list[i])
                offset = np.deg2rad(self.offset_deg_list[i])
                theta = np.deg2rad(joints[i]) + offset
                T_i = self.dh_transform(a, alpha, d, theta)
                T = T @ T_i
                T_list.append(T)

            current_pos = T[:3, 3]
            current_R = T[:3, :3]

            err_pos = target_pos - current_pos
            R_diff = target_R @ current_R.T
            err_rot = R.from_matrix(R_diff).as_rotvec()
            error = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(error) < tol:
                break

            J = np.zeros((6, N))
            p_end = current_pos
            for i in range(N):
                if i == 0:
                    z = np.array([0, 0, 1])
                    p = np.array([0., 0., 0.])
                else:
                    z = T_list[i-1][:3, 2]
                    p = T_list[i-1][:3, 3]
                J[:3, i] = np.cross(z, p_end - p)
                J[3:, i] = z

            lamda = 0.1
            I = np.eye(6)
            delta_theta = J.T @ np.linalg.solve(J @ J.T + lamda * I, error)
            joints += np.rad2deg(delta_theta)
        else:
            logger.warning(f"IK未收敛，最大迭代次数: {max_iter}, 最终误差: {np.linalg.norm(error):.6f}")
        
        return joints.tolist()

    def get_flange_relative_move(self, pose1: CartesianPose, pose2: CartesianPose) -> Tuple[float, float, float]:
        T1 = pose_to_homogeneous_matrix(pose1)
        T2 = pose_to_homogeneous_matrix(pose2)
        T1_inv = np.linalg.inv(T1)
        T_rel = T1_inv @ T2
        return T_rel[0,3], T_rel[1,3], T_rel[2,3]

    def close(self):
        logger.info("正在关闭机械臂连接")
        self.robot_service.disconnect()
        self.is_connected_flag = False
        logger.info("连接已关闭")



def main():
    arm = CRobot(ip='192.168.1.12')
    
    if arm.connect():
        arm.set_speed(100)
        pose = arm.get_tcp_pose()
        joint = arm.get_joint_pose()
        print("当前关节角度:", joint)
        carpose = CartesianPose(*pose).to_list()
        print("当前TCP位姿:", carpose)

        mid = [-350.499, -611.45, 161.834, 105.313, 17.017, -15.831]
        mid = CartesianPose(*mid)
        print(mid)
        target = [-517.47, -478.475, 161.869, 105.305, 17.023, -33.247]
        target = CartesianPose(*target)
        print(target)
        
        arm.move_circular(mid,target,speed=100,start=True,end=True)

def func():
    arm = CRobot(ip='192.168.1.12')
    if arm.connect():
        arm.set_speed(50)
        arm.move_relative_tool(dz=5, mode='joint', start=True, end=True)

        time0 = time.time()
        arm.move_relative_tool(dz=200, mode='joint', start=True, end=True)
        last0 = time.time() - time0
        print(f'单段耗时：{last0}')

        time1 = time.time()
        arm.move_relative_tool(dz=-100, mode='joint', start=True, end=True)
        arm.move_relative_tool(dz=-100, mode='joint', start=True, end=True)
        last1 = time.time() - time1
        print(f'两段耗时：{last1}')

        print(f'差值：{last1 - last0}')

        arm.move_relative_tool(dz=-5, mode='joint', start=True, end=True)

def func1():
    arm = CRobot(ip='192.168.1.12')
    if arm.connect():
        arm.set_speed(50)
        current_joint = arm.get_joint_pose()
        arm.move_by_joint_list([current_joint], speeds=[20])
        current_joint = arm.get_joint_pose()
        print('current joint: ', current_joint)
        current_joint[0] -= 90

        time0 = time.time()
        arm.move_by_joint_list([current_joint], speeds=[20])
        last0 = time.time() - time0
        print(f'单段耗时：{last0}')

        joint1 = current_joint.copy()
        joint1[0] -= 45
        joint2 = joint1.copy()
        joint2[0] -= 45
        time1 = time.time()
        arm.move_by_joint_list([joint1, joint2], speeds=[20, 20])

        last1 = time.time() - time1
        print(f'两段耗时：{last1}')
        print(f'差值：{last1 - last0}')

        current_joint = arm.get_joint_pose()
        print('current joint: ', current_joint)

if __name__ == "__main__":
    # main()
    # func()
    func1()

