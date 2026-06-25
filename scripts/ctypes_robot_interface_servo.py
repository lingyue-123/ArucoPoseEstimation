"""
ctypes_robot_interface -- 基于 crp_robot ctypes SDK 的高层运动接口

只使用路径模式（send_path_pos / send_path_joint + move_path），不使用指令模式。

Usage:
    from ctypes_robot_interface import CRobotCtypes

    robot = CRobotCtypes("libRobotService.so")
    robot.connect("192.168.1.1")

    # 笛卡尔路径运动
    robot.move_servo_path([[300,0,400,180,0,0], [400,100,400,180,0,0]], ratio=4)

    # 规划+执行
    robot.plan_and_servo(waypoints, total_time=3.0)

    robot.disconnect()
"""
import time
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from crp_robot import Robot
# from crp_robot import (RobotPosition, JointPosition, MotionParam,
#                        RobotMode, ProgramStatus, MotionType,
#                        MovePathResult, MoveStrategy)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CtypesRobotInterface")


# ── 辅助：笛卡尔位姿 ─────────────────────────────────────────────────────

class CartesianPose:
    """位姿数据类"""
    def __init__(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        self.x, self.y, self.z = x, y, z
        self.rx, self.ry, self.rz = rx, ry, rz

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @staticmethod
    def from_list(p: List[float]) -> "CartesianPose":
        return CartesianPose(*p[:6])

    def __repr__(self):
        return (f"CPose(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, "
                f"rx={self.rx:.2f}, ry={self.ry:.2f}, rz={self.rz:.2f})")


# ── 辅助：轨迹规划函数 ───────────────────────────────────────────────────

def interpolate_path(waypoints: List[List[float]],
                     num_points: int = 2000) -> List[List[float]]:
    """弧长参数化 + 三次B样条插值(xyz) + SLERP(姿态)"""
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
    key_rots = Rotation.from_euler('XYZ', eulers, degrees=True)
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


def s_curve_profile(total_distance: float, total_time: float,
                    dt: float = 0.008):
    """正弦S曲线速度规划：起点/终点速度=0"""
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

    rx0, ry0, rz0 = path[0][3], path[0][4], path[0][5]

    return [[float(xs[i]), float(ys[i]), float(zs[i]), rx0, ry0, rz0]
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


def visualize_trajectory(trajectory: List[List[float]],
                         waypoints: Optional[List[List[float]]] = None,
                         title: str = "轨迹可视化",
                         filepath: Optional[str] = None,
                         dt: float = 0.008) -> None:
    """
    轨迹可视化：3D路径 + 姿态曲线 + 速度

    Args:
        trajectory: 轨迹点 [[x,y,z,rx,ry,rz], ...]
        waypoints:  原始路点（可选，用于对比显示）
        title:      图表标题
        filepath:   保存图片路径，None 则显示窗口
        dt:         插补周期 (s)，用于计算时间轴
    """
    if not trajectory:
        logger.error("轨迹为空，无法可视化")
        return

    traj = np.array(trajectory)
    n = len(traj)
    t = np.arange(n) * dt

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── 3D 路径 ──
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

    # ── 位置 xyz vs 时间 ──
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

    # ── 姿态 rx,ry,rz vs 时间 ──
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

    # ── 速度（xyz差分） ──
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


# ── 主类 ──────────────────────────────────────────────────────────────────

class CRobotCtypes:
    """
    基于 crp_robot ctypes SDK 的机械臂控制类（路径模式）。

    Parameters
    ----------
    so_path : str
        libRobotService.so 路径
    bridge_path : str, optional
        robot_bridge.so 路径，默认同目录下
    """

    def __init__(self, so_path: str, bridge_path: str = None):
        self.robot = Robot(so_path, bridge_path)
        logger.info(f"SDK 初始化完成: {so_path}")

    def connect(self, ip: str) -> bool:
        ok = self.robot.connect(ip)
        if ok:
            logger.info(f"连接成功: {ip}")
        else:
            logger.error(f"连接失败: {ip}")
        return ok

    def disconnect(self) -> bool:
        ok = self.robot.disconnect()
        logger.info("已断开连接")
        return ok

    # ── 状态查询 ──────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> Optional[List[float]]:
        try:
            pos = self.robot.get_position()
            return pos.to_pos6()
        except Exception as e:
            logger.error(f"获取笛卡尔位姿失败: {e}")
            return None

    def get_joint_pose(self) -> Optional[List[float]]:
        try:
            jp = self.robot.get_joint()
            return jp.body[:6]
        except Exception as e:
            logger.error(f"获取关节角失败: {e}")
            return None

    def is_moving(self) -> bool:
        return self.robot.is_moving()

    # ── 可视化 ────────────────────────────────────────────────────────────

    def visualize(self, trajectory: List[List[float]],
                  waypoints: Optional[List[List[float]]] = None,
                  title: str = "轨迹可视化",
                  filepath: Optional[str] = None,
                  dt: float = 0.008) -> None:
        """
        可视化轨迹：3D路径 + 姿态曲线 + 速度

        Args:
            trajectory: 轨迹点 [[x,y,z,rx,ry,rz], ...]
            waypoints:  原始路点（可选，对比显示）
            title:      图表标题
            filepath:   保存图片路径，None 则显示窗口
            dt:         插补周期 (s)
        """
        visualize_trajectory(trajectory, waypoints, title, filepath, dt)

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _ensure_ready(self, program: str = "guidancePos.pro") -> bool:
        """统一初始化：set_work_mode → clear_error → servo_on → start_program → is_ready"""
        try:
            self.robot.set_work_mode(RobotMode.Playing)

            if self.robot.has_error():
                if self.robot.has_emergency_error():
                    logger.error("请手动清除紧急错误")
                    return False
                logger.warning("检测到错误，尝试清除")
                self.robot.clear_error()

            if not self.robot.is_servo_on():
                logger.info("伺服未上电，正在上电")
                if not self.robot.servo_on():
                    logger.error("伺服上电失败")
                    return False

            status = self.robot.get_program_status()
            if status == ProgramStatus.Stop:
                if not self.robot.start_program(program, 0):
                    logger.error(f"启动程序 {program} 失败")
                    return False
            elif status == ProgramStatus.Pause:
                if not self.robot.resume_program(program):
                    logger.error(f"恢复程序 {program} 失败")
                    return False

            wait_count = 0
            while not self.robot.motion.is_ready(MotionType.Path):
                time.sleep(0.1)
                wait_count += 1
                if wait_count > 100:
                    logger.error("等待路径运动服务就绪超时")
                    return False

            return True

        except Exception as e:
            logger.error(f"初始化异常: {e}")
            return False

    def _wait_done(self) -> None:
        """等待运动完成"""
        while self.robot.is_moving():
            time.sleep(0.5)

    # ── 路径运动 ──────────────────────────────────────────────────────────

    def move_servo_path(self, positions: List[List[float]],
                        ratio: int = 1,
                        tool_no: int = 10,
                        user_no: int = 0) -> int:
        """
        笛卡尔路径运动（路径模式）

        Args:
            positions: 位姿点列表 [[x,y,z,rx,ry,rz], ...]
            ratio:     插补周期倍率，周期=2ms*ratio，范围1~50
            tool_no:   工具坐标系编号
            user_no:   用户坐标系编号

        Returns:
            int: 1=成功, 0=失败
        """
        if ratio < 1 or ratio > 50:
            logger.error(f"ratio 取值范围 1~50，当前: {ratio}")
            return 0
        if not positions:
            logger.error("位姿点列表为空")
            return 0
        for idx, pos in enumerate(positions):
            if len(pos) != 6:
                logger.error(f"第{idx}个位姿长度必须为6，当前: {len(pos)}")
                return 0

        logger.info(f"笛卡尔路径运动，{len(positions)}个点, ratio={ratio}")

        if not self._ensure_ready():
            return 0

        try:
            robot_positions = []
            for pos in positions:
                rp = RobotPosition(x=pos[0], y=pos[1], z=pos[2],
                                   Rx=pos[3], Ry=pos[4], Rz=pos[5])
                robot_positions.append(rp)

            if not self.robot.motion.send_path_pos(robot_positions, tool_no, user_no):
                logger.error("发送笛卡尔路径数据失败")
                return 0

            result = self.robot.motion.move_path(ratio)
            if result != MovePathResult.Success:
                logger.error(f"启动路径运动失败: {result}")
                return 0

            self.robot.motion.finalize(MotionType.Path)
            self._wait_done()

            logger.info("笛卡尔路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"笛卡尔路径运动异常: {e}")
            return 0

    def move_joint_path(self, joints: List[List[float]],
                        ratio: int = 1) -> int:
        """
        关节路径运动（路径模式）

        Args:
            joints: 关节角点列表 [[j1,j2,j3,j4,j5,j6], ...]
            ratio:  插补周期倍率，周期=2ms*ratio，范围1~50

        Returns:
            int: 1=成功, 0=失败
        """
        if ratio < 1 or ratio > 50:
            logger.error(f"ratio 取值范围 1~50，当前: {ratio}")
            return 0
        if not joints:
            logger.error("关节角点列表为空")
            return 0
        for idx, joint in enumerate(joints):
            if len(joint) != 6:
                logger.error(f"第{idx}个关节角长度必须为6，当前: {len(joint)}")
                return 0

        logger.info(f"关节路径运动，{len(joints)}个点, ratio={ratio}")

        if not self._ensure_ready():
            return 0

        try:
            joint_positions = []
            for joint in joints:
                jp = JointPosition(body=joint)
                joint_positions.append(jp)

            if not self.robot.motion.send_path_joint(joint_positions):
                logger.error("发送关节路径数据失败")
                return 0

            result = self.robot.motion.move_path(ratio)
            if result != MovePathResult.Success:
                logger.error(f"启动路径运动失败: {result}")
                return 0

            self.robot.motion.finalize(MotionType.Path)
            self._wait_done()

            logger.info("关节路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"关节路径运动异常: {e}")
            return 0

    # ── 轨迹规划 ──────────────────────────────────────────────────────────

    def plan_cartesian(self, waypoints: List[List[float]],
                       total_time: float,
                       dt: float = 0.008) -> List[List[float]]:
        """
        S曲线笛卡尔轨迹规划

        Args:
            waypoints:  笛卡尔路径点 [[x,y,z,rx,ry,rz], ...]，需>=2个点
            total_time: 目标运动总时间 (s)
            dt:         插补周期 (s)，默认8ms

        Returns:
            时间参数化轨迹点，可传给 move_servo_path
        """
        return smooth_cartesian_traj(waypoints, total_time, dt)

    def plan_joint(self, waypoints: List[List[float]],
                   total_time: float,
                   dt: float = 0.008,
                   initial_joints: Optional[List[float]] = None) -> List[List[float]]:
        """
        S曲线笛卡尔规划 + 逐点逆解为关节角轨迹

        Args:
            waypoints:      笛卡尔路径点 [[x,y,z,rx,ry,rz], ...]
            total_time:     目标运动总时间 (s)
            dt:             插补周期 (s)，默认8ms
            initial_joints: IK 初始关节角，None 则取当前关节角

        Returns:
            关节角轨迹 [[j1..j6], ...]，可传给 move_joint_path
        """
        traj = self.plan_cartesian(waypoints, total_time, dt)
        if not traj:
            return []

        ik_ref = initial_joints
        if ik_ref is None:
            ik_ref = self.get_joint_pose()
            if ik_ref is None:
                logger.error("无法获取当前关节角作为IK初始值")
                return []

        joint_traj = []
        for i, pose in enumerate(traj):
            result = self._inverse_kinematics(pose, ik_ref)
            if result is None:
                logger.error(f"第{i}个点逆解失败: {pose}")
                return []
            joint_traj.append(result)
            ik_ref = result

        logger.info(f"逆解完成，{len(joint_traj)}个关节角点")
        return joint_traj

    def _inverse_kinematics(self, target_pose: List[float],
                            initial_joints: List[float]) -> Optional[List[float]]:
        """
        数值逆运动学（阻尼最小二乘法）

        Args:
            target_pose:   目标位姿 [x,y,z,rx,ry,rz]
            initial_joints: 初始关节角 [j1..j6]

        Returns:
            关节角 [j1..j6] 或 None
        """
        try:
            joints = np.array(initial_joints, dtype=float)
            target = np.array(target_pose[:3], dtype=float)

            a = [0, 621.620, 559.133, 0, 0, 0]
            d = [0, 0, 0, -164.261, 119.327, 115.0]
            alpha_deg = [90, 0, 0, 90, 90, 0]
            offset_deg = [0, 0, -90, 90, -90, 0]

            def dh_transform(a_i, alpha_i, d_i, theta_i):
                ca, sa = np.cos(alpha_i), np.sin(alpha_i)
                ct, st = np.cos(theta_i), np.sin(theta_i)
                return np.array([
                    [ct, -st*ca,  st*sa, a_i*ct],
                    [st,  ct*ca, -ct*sa, a_i*st],
                    [0,   sa,     ca,    d_i],
                    [0,   0,      0,     1]
                ])

            def fk(joint_angles):
                T = np.eye(4)
                for i in range(6):
                    theta = np.radians(joint_angles[i] + offset_deg[i])
                    alpha = np.radians(alpha_deg[i])
                    T = T @ dh_transform(a[i], alpha, d[i], theta)
                return T[:3, 3]

            def jacobian(joint_angles, eps=1e-6):
                pos0 = fk(joint_angles)
                J = np.zeros((3, 6))
                for i in range(6):
                    j_backup = joint_angles[i]
                    joint_angles[i] += eps
                    pos1 = fk(joint_angles)
                    joint_angles[i] = j_backup
                    J[:, i] = (pos1 - pos0) / eps
                return J

            for _ in range(50):
                pos = fk(joints)
                err = target - pos
                if np.linalg.norm(err) < 1e-4:
                    break
                J = jacobian(joints)
                joints += np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ err)

            if np.linalg.norm(target - fk(joints)) > 1e-3:
                return None

            return joints.tolist()

        except Exception as e:
            logger.error(f"逆解异常: {e}")
            return None

    # ── 规划+执行 ─────────────────────────────────────────────────────────

    def plan_and_servo(self, waypoints: List[List[float]],
                       total_time: float,
                       dt: float = 0.008,
                       tool_no: int = 10,
                       user_no: int = 0) -> int:
        """
        S曲线规划 + 笛卡尔路径执行

        Args:
            waypoints:  笛卡尔路径点
            total_time: 目标运动总时间 (s)
            dt:         插补周期 (s)
            tool_no:    工具坐标系编号
            user_no:    用户坐标系编号

        Returns:
            int: 1=成功, 0=失败
        """
        traj = self.plan_cartesian(waypoints, total_time, dt)
        if not traj:
            return 0
        ratio = max(1, int(dt / 0.002))
        return self.move_servo_path(traj, ratio=ratio,
                                    tool_no=tool_no, user_no=user_no)

    def plan_and_joint(self, waypoints: List[List[float]],
                       total_time: float,
                       dt: float = 0.008,
                       initial_joints: Optional[List[float]] = None) -> int:
        """
        S曲线规划 + 逆解 + 关节路径执行

        Args:
            waypoints:      笛卡尔路径点
            total_time:     目标运动总时间 (s)
            dt:             插补周期 (s)
            initial_joints: IK 初始关节角

        Returns:
            int: 1=成功, 0=失败
        """
        joint_traj = self.plan_joint(waypoints, total_time, dt, initial_joints)
        if not joint_traj:
            return 0
        ratio = max(1, int(dt / 0.002))
        return self.move_joint_path(joint_traj, ratio=ratio)

    # ── 轨迹文件读写 + 便捷方法 ───────────────────────────────────────────

    def plan_and_save(self, waypoints: List[List[float]],
                      total_time: float, filepath: str,
                      dt: float = 0.008) -> int:
        """
        规划轨迹并保存到文件，不执行运动

        Args:
            waypoints:  笛卡尔路径点
            total_time: 目标运动总时间 (s)
            filepath:   保存路径
            dt:         插补周期 (s)

        Returns:
            int: 轨迹点数量，0=失败
        """
        traj = self.plan_cartesian(waypoints, total_time, dt)
        if not traj:
            return 0
        save_trajectory(traj, filepath)
        return len(traj)

    def load_and_servo(self, filepath: str, ratio: int = 1,
                       tool_no: int = 10, user_no: int = 0) -> int:
        """
        从文件加载笛卡尔轨迹并执行

        Args:
            filepath: 轨迹文件路径
            ratio:    插补周期倍率
            tool_no:  工具坐标系编号
            user_no:  用户坐标系编号

        Returns:
            int: 1=成功, 0=失败
        """
        traj = load_trajectory(filepath)
        if not traj:
            return 0
        return self.move_servo_path(traj, ratio=ratio,
                                    tool_no=tool_no, user_no=user_no)

    def load_and_joint(self, filepath: str, ratio: int = 1) -> int:
        """
        从文件加载关节轨迹并执行

        Args:
            filepath: 关节轨迹文件路径
            ratio:    插补周期倍率

        Returns:
            int: 1=成功, 0=失败
        """
        traj = load_trajectory(filepath)
        if not traj:
            return 0
        return self.move_joint_path(traj, ratio=ratio)


# ── 使用示例 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ============================================================
    # 注意：以下示例需要真实机器人控制器连接
    # 无连接时可运行"轨迹规划+可视化"部分（不需要机器人）
    # ============================================================

    # ── 示例1：轨迹规划 + 可视化（无需连接机器人） ──────────────
    print("=" * 50)
    print("示例1：轨迹规划 + 可视化")
    print("=" * 50)

    waypoints = [
        [300, 0, 400, 0, 0, 0],      # 起点
        [400, 50, 450, 0, 0, 30],    # 中间点1
        [500, 100, 400, 0, 0, 60],   # 中间点2
        [400, 150, 350, 0, 0, 90],   # 终点
    ]

    # 笛卡尔轨迹规划
    from ctypes_robot_interface import smooth_cartesian_traj, visualize_trajectory
    traj = smooth_cartesian_traj(waypoints, total_time=3.0, dt=0.008)
    # traj = smooth_cartesian_traj(waypoints, total_time=3.0, dt=0.008, profile='trapezoid', accel_frac=0.1)
    
    print(f"规划生成 {len(traj)} 个轨迹点")

    # 可视化
    visualize_trajectory(traj, waypoints,
                         title="S-curve Trajectory",)
                        #  filepath="trajectory_demo.png")
    print("轨迹图已保存至 trajectory_demo.png")

    # 保存轨迹文件
    save_trajectory(traj, "trajectory.txt")
    print("轨迹数据已保存至 trajectory.txt")

    # ── 示例2：连接机器人并执行运动（需要真实连接） ────────────
    # print()
    # print("=" * 50)
    # print("示例2：连接机器人 + 执行运动")
    # print("=" * 50)

    # SO_PATH = "libRobotService.so"
    # ROBOT_IP = "192.168.1.1"

    # try:
    #     robot = CRobotCtypes(SO_PATH)
    #     if not robot.connect(ROBOT_IP):
    #         print("连接失败，请检查网络和机器人状态")
    #         exit(1)

    #     print(f"已连接到机器人: {ROBOT_IP}")

    #     # 获取当前位姿
    #     pose = robot.get_tcp_pose()
    #     print(f"当前笛卡尔位姿: {pose}")

    #     joints = robot.get_joint_pose()
    #     print(f"当前关节角: {joints}")

    #     # ── 示例2a：笛卡尔路径运动 ──
    #     print()
    #     print("--- 笛卡尔路径运动 ---")
    #     positions = [
    #         [300, 0, 400, 180, 0, 0],
    #         [350, 50, 420, 180, 0, 15],
    #         [400, 100, 400, 180, 0, 30],
    #     ]
    #     robot.move_servo_path(positions, ratio=4)

    #     # ── 示例2b：关节路径运动 ──
    #     print()
    #     print("--- 关节路径运动 ---")
    #     joint_path = [
    #         [0, -30, 90, 0, 60, 0],
    #         [10, -45, 80, 0, 45, 0],
    #         [0, -30, 90, 0, 60, 0],
    #     ]
    #     robot.move_joint_path(joint_path, ratio=4)

    #     # ── 示例2c：S曲线规划 + 执行（笛卡尔） ──
    #     print()
    #     print("--- S曲线规划+执行（笛卡尔） ---")
    #     robot.plan_and_servo(waypoints, total_time=3.0, dt=0.008)

    #     # ── 示例2d：S曲线规划 + IK + 执行（关节） ──
    #     print()
    #     print("--- S曲线规划+IK+执行（关节） ---")
    #     robot.plan_and_joint(waypoints, total_time=3.0, dt=0.008)

    #     # ── 示例2e：规划 + 保存 + 加载执行 ──
    #     print()
    #     print("--- 规划+保存+加载执行 ---")
    #     count = robot.plan_and_save(waypoints, total_time=2.0,
    #                                 filepath="planned_traj.txt")
    #     print(f"规划并保存 {count} 个点")

    #     robot.load_and_servo("planned_traj.txt", ratio=4)

    #     robot.disconnect()
    #     print("已断开连接")

    # except Exception as e:
    #     print(f"运行异常: {e}")
