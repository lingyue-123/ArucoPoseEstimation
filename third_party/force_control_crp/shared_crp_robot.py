import logging
import os
import time
from typing import List, Optional

import numpy as np
from scipy.interpolate import make_interp_spline

from scipy.spatial.transform import Rotation as R,Slerp

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scripts.crobot_driver_interface import (
    CartesianPose,
    pose_to_homogeneous_matrix,
    homogeneous_matrix_to_pose,
)
from third_party.crp_robot_sdk.crp_robot import Robot
from third_party.crp_robot_sdk.crp_robot._types import JointPosition, RobotPosition, MotionParam, DHParam
from third_party.crp_robot_sdk.crp_robot._enums import MotionType, MoveStrategy, ProgramStatus, RobotMode, MovePathResult

from contextlib import contextmanager


logger = logging.getLogger("BridgeCRobotAdapter")

_POLL_INTERVAL_S = 0.5
_INSTRUCTION_READY_TIMEOUT_S = 30.0
_MOTION_COMPLETE_TIMEOUT_S = 60.0
import threading
import socket
import json
class MotionPauseController:
    def __init__(self, sock_path: str = "/tmp/gun_pause.sock"):
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        self._sock_path = sock_path
        self._thread = None
        self._running = False
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="uds-pause-ctrl")
        self._thread.start()
        logger.info(f"MotionPauseController started, listening on {self._sock_path}")
    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
    def _listen_loop(self):
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.settimeout(1.0)
        try:
            srv.bind(self._sock_path)
            srv.listen(1)
        except Exception as e:
            logger.error(f"MotionPauseController: bind failed: {e}")
            self._running = False
            return
        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"MotionPauseController: accept failed: {e}")
                    time.sleep(1)
                continue
            try:
                with conn:
                    conn.settimeout(5.0)
                    try:
                        data = conn.recv(4096)
                        if data:
                            for line in data.decode(errors="ignore").split("\n"):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    payload = json.loads(line)
                                    cmd = payload.get("gun_pause", 0)
                                    if cmd == 1:
                                        self.pause_event.set()
                                        self.resume_event.clear()
                                        logger.info("MotionPauseController: pause signal received")
                                    elif cmd == 0:
                                        self.resume_event.set()
                                        self.pause_event.clear()
                                        logger.info("MotionPauseController: resume signal received")
                                except json.JSONDecodeError:
                                    logger.warning("MotionPauseController: JSON parse failed: %s", line)
                    except socket.timeout:
                        pass
            except Exception as e:
                if self._running:
                    logger.error(f"MotionPauseController: connection error: {e}")
        try:
            srv.close()
        except Exception:
            pass
        try:
            if os.path.exists(self._sock_path):
                os.unlink(self._sock_path)
        except Exception:
            pass
    def is_paused(self) -> bool:
        return self.pause_event.is_set()
    def wait_resume(self, timeout: float = None) -> bool:
        return self.resume_event.wait(timeout)
    



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


def resample_s_curve(path: List[List[float]], total_time: float,
                     dt: float = 0.008) -> List[List[float]]:
    """
    对密集路径点重新做 S 曲线时间重参数化。

    用于恢复运动时，将剩余路径点按 S 曲线速度规划重新采样，
    保证恢复段也有平滑的加减速。

    Args:
        path:       密集路径点 [[x,y,z,rx,ry,rz], ...]，至少 2 个点
        total_time: 该段的目标运动时间 (s)
        dt:         插补周期 (s)，默认 8ms

    Returns:
        S 曲线重参数化后的路径点列表
    """
    if len(path) < 2:
        return path

    # 累积弧长（仅用 xyz 位置）
    dists = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dz = path[i][2] - path[i - 1][2]
        dists.append(dists[-1] + np.sqrt(dx * dx + dy * dy + dz * dz))
    dists = np.array(dists)

    if dists[-1] < 1e-6:
        return path

    # S 曲线距离序列
    s_curve_dist, _ = s_curve_profile(dists[-1], total_time, dt)

    # 按 S 曲线距离重新采样六维位姿
    xs = np.interp(s_curve_dist, dists, [p[0] for p in path])
    ys = np.interp(s_curve_dist, dists, [p[1] for p in path])
    zs = np.interp(s_curve_dist, dists, [p[2] for p in path])
    rxs = np.interp(s_curve_dist, dists, [p[3] for p in path])
    rys = np.interp(s_curve_dist, dists, [p[4] for p in path])
    rzs = np.interp(s_curve_dist, dists, [p[5] for p in path])

    return [[float(xs[i]), float(ys[i]), float(zs[i]),
             float(rxs[i]), float(rys[i]), float(rzs[i])]
            for i in range(len(xs))]


def resample_joint_s_curve(joints: List[List[float]], total_time: float,
                            dt: float = 0.008) -> List[List[float]]:
    """
    对密集关节轨迹点重新做 S 曲线时间重参数化（关节空间）。

    Args:
        joints:     密集关节角点 [[j1,...,j6], ...]，至少 2 个点
        total_time: 该段的目标运动时间 (s)
        dt:         插补周期 (s)，默认 8ms

    Returns:
        S 曲线重参数化后的关节角点列表
    """
    if len(joints) < 2:
        return joints

    # 累积弧长（六维关节空间）
    dists = [0.0]
    for i in range(1, len(joints)):
        d = np.array(joints[i]) - np.array(joints[i - 1])
        dists.append(dists[-1] + np.linalg.norm(d))
    dists = np.array(dists)

    if dists[-1] < 1e-6:
        return joints

    s_curve_dist, _ = s_curve_profile(dists[-1], total_time, dt)

    result = []
    for dim in range(6):
        values = np.interp(s_curve_dist, dists, [j[dim] for j in joints])
        result.append(values)

    return [[float(result[d][i]) for d in range(6)]
            for i in range(len(s_curve_dist))]


def load_trajectory(filepath: str) -> List[List[float]]:
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
    with open(filepath, 'w+') as f:
        for p in trajectory:
            s = f"{p[0]:.6f}{separator}{p[1]:.6f}{separator}{p[2]:.6f}{separator}{p[3]:.6f}{separator}{p[4]:.6f}{separator}{p[5]:.6f}\n"
            # print(s)
            f.write(s)
    logger.info(f"保存 {len(trajectory)} 个轨迹点至 {filepath}")


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


class BridgeCRobotAdapter:
    def __init__(self, ip: str, so_path: str):
        self.ip = ip
        self.so_path = so_path
        self.bridge_robot = Robot(so_path)
        self._connected = False

        self.a_list = [0, 621.620, 559.133, 0, 0, 0]
        self.d_list = [0, 0, 0, -164.261, 119.327, 115.0]
        self.alpha_deg_list = [90, 0, 0, 90, 90, 0]
        self.offset_deg_list = [0, 0, -90, 90, -90, 0]

        # 预计算固定值（避免重复调用 deg2rad 和三角函数）
        self.alpha_rad_list = np.deg2rad(self.alpha_deg_list)
        self.sin_alpha = np.sin(self.alpha_rad_list)
        self.cos_alpha = np.cos(self.alpha_rad_list)
        self.offset_rad_list = np.deg2rad(self.offset_deg_list)

            # 关节限位（度）
        self.joint_limits = [
            (-360, 360),   # J1
            (0, 180),      # J2
            (-75, 250),    # J3
            (-360, 360),   # J4
            (-50, 120),   # J5
            (-360, 360)    # J6
        ]

        self._pause_ctrl = MotionPauseController()
        self._pause_ctrl.start()

    def _dh_matrix(self, a, sin_alpha, cos_alpha, d, theta_rad):
        """构建单个 DH 变换矩阵（不重复计算 sin/cos alpha）"""
        st = np.sin(theta_rad)
        ct = np.cos(theta_rad)
        return np.array([
            [ct, -st * cos_alpha,  st * sin_alpha, a * ct],
            [st,  ct * cos_alpha, -ct * sin_alpha, a * st],
            [0,          sin_alpha,        cos_alpha,      d],
            [0,                  0,               0,      1]
        ])
    
    @staticmethod
    def _rotmat_to_rotvec(R):
        """从旋转矩阵计算旋转向量（轴角），返回 (3,) 向量，模长为转角（弧度）"""
        # 公式: theta = arccos((trace-1)/2), 轴 = (R32-R23, R13-R31, R21-R12)/(2*sin(theta))
        trace = R[0,0] + R[1,1] + R[2,2]
        theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if theta < 1e-6:
            return np.zeros(3)
        r = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
        r = r / (2.0 * np.sin(theta)) * theta
        return r

    def connect(self) -> bool:
        ok = self.bridge_robot.connect(self.ip, disable_hardware=True)
        self._connected = bool(ok)
        return self._connected

    def disconnect(self) -> None:
        self._pause_ctrl.stop()
        if self._connected:
            try:
                self.bridge_robot.disconnect()
            finally:
                self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self.bridge_robot.is_connected()

    def _ensure_ready_for_motion(self):
        self.bridge_robot.set_work_mode(RobotMode.Playing)
        if self.bridge_robot.has_error():
            if self.bridge_robot.has_emergency_error():
                errors = []
                try:
                    errors = self.bridge_robot.get_errors()
                except Exception as exc:
                    logger.error("get_errors failed while handling emergency error: %s", exc)
                if errors:
                    raise RuntimeError(f"robot has emergency error: {errors}")
                raise RuntimeError("robot has emergency error")
            self.bridge_robot.clear_error()
        if not self.bridge_robot.is_servo_on():
            if not self.bridge_robot.servo_on():
                raise RuntimeError("servo on failed")

    def _ensure_instruction_program(self):
        status = self.bridge_robot.get_program_status()
        if status == ProgramStatus.Stop:
            if not self.bridge_robot.start_program("guidanceInst.pro", 0):
                raise RuntimeError("start guidanceInst.pro failed")
        elif status == ProgramStatus.Pause:
            if not self.bridge_robot.resume_program("guidanceInst.pro"):
                raise RuntimeError("resume guidanceInst.pro failed")
        start = time.time()
        while not self.bridge_robot.motion.is_ready(MotionType.Instruction):
            if time.time() - start > _INSTRUCTION_READY_TIMEOUT_S:
                raise TimeoutError("waiting for instruction motion ready timed out")
            time.sleep(_POLL_INTERVAL_S)

    def _ensure_guidance_program(self):
        status = self.bridge_robot.get_program_status()
        if status == ProgramStatus.Stop:
            if not self.bridge_robot.start_program("guidancePos.pro", 0):
                raise RuntimeError("start guidancePos.pro failed")
        elif status == ProgramStatus.Pause:
            if not self.bridge_robot.resume_program("guidancePos.pro"):
                raise RuntimeError("resume guidancePos.pro failed")
        start = time.time()
        while not self.bridge_robot.motion.is_ready(MotionType.Path):
            if time.time() - start > _INSTRUCTION_READY_TIMEOUT_S:
                raise TimeoutError("waiting for guidance motion ready timed out")
            time.sleep(_POLL_INTERVAL_S)
            
    @contextmanager
    def instruction_session(self):
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            yield
        except Exception as exc:
            logger.error("instruction session error: %s", exc)
        finally:
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            while(self.bridge_robot.is_moving()):
                logger.info("Wait for robot to stop...")
                time.sleep(0.1)
    
    @contextmanager
    def path_session(self):
        try:
            self._ensure_ready_for_motion()
            self._ensure_guidance_program()
            yield
        except Exception as exc:
            logger.error("path session error: %s", exc)
        finally:
            self.bridge_robot.motion.finalize(MotionType.Path)
            while(self.bridge_robot.is_moving()):
                logger.info("Wait for robot to stop...")
                time.sleep(0.1)
            

    def switch_motion_model(self):
        """切换运动模式：停止程序 → 手动模式"""
        try:
            logger.info("切换运动模式：停止程序 → 手动模式")
            self.bridge_robot.stop_program()
            self.bridge_robot.set_work_mode(RobotMode.Manual)
            self.bridge_robot.stop_program()
            logger.info("运动模式切换完成")
        except Exception as exc:
            logger.error("切换运动模式失败: %s", exc)

    def _wait_for_motion_complete(self,cb, timeout: float = _MOTION_COMPLETE_TIMEOUT_S) -> None:
        start = time.time()
        time.sleep(0.1)
        sleep_time = 0
        state = 1
        while state:
            if sleep_time > 4:
                logger.info("机械臂恢复")
                logger.info(f"当前状态：{self.bridge_robot.get_program_status()}")
                self.bridge_robot.resume_program("guidanceInst.pro")
                ok = cb()
                
                # self.bridge_robot.get_program_status()
                sleep_time = 0
                time.sleep(1)
                logger.info(f"当前状态：{self.bridge_robot.get_program_status()}")
                if self.is_moving():
                    state = 1
                    logger.info("机械臂确实恢复运动")
                    continue
                else:
                    state = 1
            if sleep_time > 2:
                logger.info("机械臂暂停")
                self.bridge_robot.stop_program()
                time.sleep(0.1)
                sleep_time += 0.1
                continue
            logger.info("机械臂正在执行运动序列...")
            sleep_time += 0.1
            if time.time() - start > timeout:
                raise TimeoutError("waiting for robot motion complete timed out")
            # time.sleep(_POLL_INTERVAL_S)
            time.sleep(0.1)
        logger.info("机械臂运动执行完成...")

    def _wait_for_motion_complete(self, timeout: float = _MOTION_COMPLETE_TIMEOUT_S) -> None:
        start = time.time()
        time.sleep(0.1)
        while self.is_moving():
            logger.info("机械臂正在执行运动序列...")
            if time.time() - start > timeout:
                raise TimeoutError("waiting for robot motion complete timed out")
            time.sleep(_POLL_INTERVAL_S)
        logger.info("机械臂运动执行完成...")

    def set_speed(self, speed_pct: int) -> bool:
        return self.bridge_robot.set_speed_ratio(speed_pct)

    def get_tcp_pose(self) -> Optional[List[float]]:
        try:
            pos = self.bridge_robot.get_position()
            return [pos.x, pos.y, pos.z, pos.Rx, pos.Ry, pos.Rz]
        except Exception as exc:
            logger.error("get_tcp_pose failed: %s", exc)
            return None

    def get_joint_pose(self) -> Optional[List[float]]:
        try:
            joints = self.bridge_robot.get_joint()
            return [round(v, 3) for v in joints.body]
        except Exception as exc:
            logger.warning("get_joint_pose first attempt failed: %s", exc)

        try:
            self._ensure_ready_for_motion()
            time.sleep(_POLL_INTERVAL_S)
            joints = self.bridge_robot.get_joint()
            return [round(v, 3) for v in joints.body]
        except Exception as exc:
            logger.error("get_joint_pose failed after retry: %s", exc)
            return None

    def is_moving(self) -> bool:
        try:
            return self.bridge_robot.is_moving()
        except Exception as exc:
            logger.error("is_moving failed: %s", exc)
            return False

    def move_linear(self, target: CartesianPose, speed: int = 100, start: bool = True, end: bool = True) -> int:
        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_instruction_program()
            rp = RobotPosition(x=target.x, y=target.y, z=target.z, Rx=target.rx, Ry=target.ry, Rz=target.rz)
            ok = self.bridge_robot.motion.move_l(
                0,
                rp,
                MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1),
                MoveStrategy.DistanceFirst,
            )
            if not ok:
                return 0
            if end:
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_linear failed: %s", exc)
            return 0
        
    def move_linear_new(self, target: CartesianPose, speed: int = 100,
                    pause_timeout: Optional[float] = None) -> int:
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            rp = RobotPosition(x=target.x, y=target.y, z=target.z,
                            Rx=target.rx, Ry=target.ry, Rz=target.rz)
            param = MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1)
            ok = self.bridge_robot.motion.move_l(
                0, rp, param, MoveStrategy.DistanceFirst)
            if not ok:
                return 0
            pause_start = None
            time.sleep(0.1)
            while self.is_moving():
                if self._pause_ctrl.pause_event.is_set():
                    logger.info("move_linear_new: pause signal received, stopping motion")
                    self.bridge_robot.stop_program()
                    self._pause_ctrl.pause_event.clear()
                    pause_start = time.time()
                    while not self._pause_ctrl.resume_event.is_set():
                        if pause_timeout and (time.time() - pause_start > pause_timeout):
                            logger.error("move_linear_new: pause timeout")
                            return 0
                        if not self.is_connected():
                            logger.error("move_linear_new: disconnected during pause")
                            return 0
                        time.sleep(0.1)
                    logger.info("move_linear_new: resume signal received, resuming motion")
                    self.bridge_robot.resume_program("guidanceInst.pro")
                    self.bridge_robot.motion.move_l(
                        0, rp, param, MoveStrategy.DistanceFirst)
                    self._pause_ctrl.resume_event.clear()
                    pause_start = None
                time.sleep(0.1)
            while(self.bridge_robot.is_moving()):
                logger.info("等待停止")
                pass
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            logger.info("move_linear_new: motion completed")
            return 1
        except Exception as exc:
            logger.error("move_linear_new failed: %s", exc)
            return 0

    def move_joint(self, joint: List[float], speed: int = 100, start: bool = True, end: bool = True) -> int:
        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_instruction_program()
            jp = JointPosition(body=list(joint))
            ok = self.bridge_robot.motion.move_abs_j(
                0,
                jp,
                MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1),
            )
            if not ok:
                return 0
            if end:
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_joint failed: %s", exc)
            return 0

    def move_by_pose_list(self, poses: List[List[float]], speeds: List[int], end: bool = True) -> int:
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            for idx, pose in enumerate(poses):
                rp = RobotPosition(x=pose[0], y=pose[1], z=pose[2], Rx=pose[3], Ry=pose[4], Rz=pose[5])
                ok = self.bridge_robot.motion.move_l(
                    idx,
                    rp,
                    MotionParam(speed=float(speeds[idx]), pl=0.0, smooth=0, acc=1, dec=1),
                    MoveStrategy.DistanceFirst,
                )
                if not ok:
                    return 0
            if end:
                logger.info('pose moving...')
                self.bridge_robot.motion.finalize(MotionType.Instruction)
                self._wait_for_motion_complete()
                # self._wait_for_motion_complete(lambda: self.bridge_robot.motion.move_l(
                #     idx,                          # 捕获当前的 idx
                #     rp,                           # 捕获当前的 RobotPosition
                #     MotionParam(speed=float(speeds[idx]), pl=0.0, smooth=0, acc=1, dec=1),
                #     MoveStrategy.DistanceFirst
                # ))
                self.bridge_robot.motion.finalize(MotionType.Instruction)
            return 1
        except Exception as exc:
            logger.error("move_by_pose_list failed: %s", exc)
            return 0

    def move_by_joint_list(self, joints: List[List[float]], speeds: List[int]) -> int:
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            for idx, joint in enumerate(joints):
                speed = speeds[idx]
                logger.info(f"发送第{idx + 1}/{len(joints)}个关节运动: 关节={joint}, 速度={speed}")
                ok = self.bridge_robot.motion.move_abs_j(
                    idx,
                    JointPosition(body=list(joint)),
                    MotionParam(speed=float(speeds[idx]), pl=0.0, smooth=0, acc=1, dec=1),
                )
                if not ok:
                    return 0
            self.bridge_robot.motion.finalize(MotionType.Instruction)
            self._wait_for_motion_complete()
            return 1
        except Exception as exc:
            logger.error("move_by_joint_list failed: %s", exc)
            return 0
    
    
        
    def move_circular(self,mid:CartesianPose,target:CartesianPose,speed:float,seq:int = 0,acc:int = 1 ,dec:int = 1,pl:int = 0):
        logger.info(f"执行圆弧运动到目标:中点{mid}，终点{target}, 速度: {speed}")
        try:
            self._ensure_ready_for_motion()
            self._ensure_instruction_program()
            p1 = RobotPosition()
            p1.set_pose(mid.to_list())
            
            p2 = RobotPosition(target)
            p2.set_pose(target.to_list())
            param = MotionParam(speed=float(speed), pl=1.0, smooth=0, acc=1, dec=1)
            

            # p_status = self.robot_service.get_program_status()
            # if p_status == ProgramStatus.STOP:
            #     if not self.robot_service.start_program("guidanceInst.pro", 0):
            #         logger.error("=====启动程序失败=====")
            #         return 0
            # elif p_status == ProgramStatus.PAUSE:
            #     if not self.robot_service.resume_program("guidanceInst.pro"):
            #         logger.error("=====恢复程序失败=====")
            #         return 0
            # else:
            #     logger.info("=====程序正在运行=====")
            
            # while not self.motion_service.is_ready(MotionType.INSTRUCTION):
            #     time.sleep(0.1)
            
            self.bridge_robot.motion.move_c(seq,p1,p2, param,MoveStrategy.DistanceFirst)
            time.sleep(0.1)

            
            while self.bridge_robot.is_moving():
                logger.info("机械臂正在运动...")
                time.sleep(1)
            
            logger.info("圆弧运动完成")
            return 1

        except Exception as e:
            logger.error(f"圆弧运动时发生异常: {e}")
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
            logger.error(f"ratio 取值范围 1~50，当前: {ratio}")
            return 0
        if not positions:
            logger.error("位姿点列表为空")
            return 0

        for idx, pos in enumerate(positions):
            if len(pos) != 6:
                logger.error(f"第{idx}个位姿参数长度必须为6，当前长度: {len(pos)}")
                return 0

        logger.info(f"开始执行笛卡尔路径运动，共{len(positions)}个点, ratio={ratio}")

        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_guidance_program()

            robot_positions = []
            for pos in positions:
                rp = RobotPosition(x=pos[0], y=pos[1], z=pos[2],
                                   Rx=pos[3], Ry=pos[4], Rz=pos[5])
                robot_positions.append(rp)

            if not self.bridge_robot.motion.send_path_pos(robot_positions, tool_no, user_no):
                logger.error("发送笛卡尔路径数据失败")
                return 0
            print("当前缓存区大小：",self.bridge_robot.motion.get_avail_buffer())
            result = self.bridge_robot.motion.move_path(ratio)
            if result != MovePathResult.Success:
                logger.error(f"启动路径运动失败: {result}")
                return 0
            
            
            logger.info("笛卡尔路径运动启动成功")

            if end:
                self.bridge_robot.motion.finalize(MotionType.Path)
                self._wait_for_motion_complete()
                # while self.bridge_robot.motion.get_avail_buffer() < 2047:
                #     time.sleep(0.1)
                # self.bridge_robot.motion.finalize(MotionType.PATH)

            logger.info("笛卡尔路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"笛卡尔路径运动执行异常: {e}")
            return 0

    def _get_consumed_index_from_buffer(self, total_sent: int) -> int:
        """
        通过缓冲区剩余容量推算已消费的轨迹点数量。

        缓冲区大小为 get_max_buffer()（默认 2048），
        send_path_* 填满后，控制器逐点消费释放空间。

        公式：consumed = total_sent - max_buf + avail_buf
        """
        max_buf = self.bridge_robot.motion.get_max_buffer()
        avail = self.bridge_robot.motion.get_avail_buffer()
        consumed = total_sent - max_buf + avail
        return max(0, min(consumed, total_sent))

    def _get_closest_index_by_position(self, positions: List[List[float]],
                                        tool_no: int = 10,
                                        user_no: int = 0) -> int:
        """
        通过当前 TCP 物理位置在轨迹中搜索最近点，返回其索引。

        使用 motion.current_user_pos 获取当前用户坐标系下的位姿，
        然后对轨迹点做最近邻搜索（仅比较 xyz 位置）。
        """
        try:
            cur = self.bridge_robot.motion.current_user_pos(tool_no, user_no)
            cur_xyz = np.array([cur.x, cur.y, cur.z])
        except Exception:
            tcp = self.get_tcp_pose()
            if tcp is None:
                logger.warning("无法获取当前位置，回退到索引 0")
                return 0
            cur_xyz = np.array(tcp[:3])

        traj_xyz = np.array([p[:3] for p in positions])
        distances = np.linalg.norm(traj_xyz - cur_xyz, axis=1)
        idx = int(np.argmin(distances))
        logger.info("当前位置最近匹配: 索引 %d, 距离 %.2f mm", idx, distances[idx])
        return idx

    def _get_closest_index_by_joint(self, joints: List[List[float]]) -> int:
        """
        通过当前关节角在关节轨迹中搜索最近点，返回其索引。
        """
        cur_joints = self.get_joint_pose()
        if cur_joints is None:
            logger.warning("无法获取当前关节角，回退到索引 0")
            return 0

        cur = np.array(cur_joints)
        traj = np.array(joints)
        distances = np.linalg.norm(traj - cur, axis=1)
        idx = int(np.argmin(distances))
        logger.info("当前关节最近匹配: 索引 %d, 距离 %.3f deg", idx, distances[idx])
        return idx

    def _send_and_start_position_path(self, positions: List[List[float]],
                                       ratio: int, tool_no: int, user_no: int) -> None:
        robot_positions = [
            RobotPosition(x=p[0], y=p[1], z=p[2], Rx=p[3], Ry=p[4], Rz=p[5])
            for p in positions
        ]
        if not self.bridge_robot.motion.send_path_pos(robot_positions, tool_no, user_no):
            raise RuntimeError("send_path_pos failed")
        result = self.bridge_robot.motion.move_path(ratio)
        if result != MovePathResult.Success:
            raise RuntimeError(f"move_path failed: {result}")

    def _send_and_start_joint_path(self, joints: List[List[float]],
                                    ratio: int) -> None:
        joint_positions = [JointPosition(body=j) for j in joints]
        if not self.bridge_robot.motion.send_path_joint(joint_positions):
            raise RuntimeError("send_path_joint failed")
        result = self.bridge_robot.motion.move_path(ratio)
        if result != MovePathResult.Success:
            raise RuntimeError(f"move_path failed: {result}")

    def move_by_position_path_with_pause(self, positions: List[List[float]],
                                          ratio: int = 1,
                                          tool_no: int = 10, user_no: int = 0,
                                          start: bool = True, end: bool = True,
                                          total_time: Optional[float] = None,
                                          pause_timeout: Optional[float] = None) -> int:
        """
        笛卡尔路径运动（带暂停/恢复），通过 MotionPauseController 接收外部信号。
        恢复时从暂停位置继续，并对剩余段重新做 S 曲线速度规划。

        Args:
            positions:     位姿点列表 [[x,y,z,rx,ry,rz], ...]
            ratio:         插补周期倍率，1~50
            tool_no:       工具坐标系编号
            user_no:       用户坐标系编号
            start:         是否在运动前执行初始化
            end:           是否在运动完成后 finalize
            total_time:    原始轨迹的总运动时间 (s)，传入后恢复段会重新做 S 曲线规划
            pause_timeout: 暂停超时 (s)，None 表示无超时

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
                logger.error(f"第{idx}个位姿参数长度必须为6，当前长度: {len(pos)}")
                return 0

        original_positions = positions
        logger.info(f"开始执行笛卡尔路径运动（带暂停），共{len(positions)}个点, ratio={ratio}")

        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_guidance_program()

            self._send_and_start_position_path(positions, ratio, tool_no, user_no)
            logger.info("笛卡尔路径运动启动成功")
            time.sleep(0.1)

            while self.is_moving():
                if self._pause_ctrl.pause_event.is_set():
                    paused_at_index = self._get_consumed_index_from_buffer(
                        len(original_positions))
                    logger.info("路径运动暂停信号，当前索引 %d / %d (buffer推算)",
                                paused_at_index, len(original_positions))

                    self.bridge_robot.stop_program()
                    self._pause_ctrl.pause_event.clear()


                    pause_start = time.time()
                    while not self._pause_ctrl.resume_event.is_set():
                        if pause_timeout and (time.time() - pause_start > pause_timeout):
                            logger.error("路径运动暂停超时")
                            return 0
                        if not self.is_connected():
                            logger.error("暂停期间连接断开")
                            return 0
                        time.sleep(0.1)

                    if paused_at_index >= len(original_positions):
                        logger.info("暂停时运动已完成")
                        break

                    logger.info("路径运动恢复，从索引 %d 继续", paused_at_index)
                    # self._ensure_guidance_program()
                    self._ensure_ready_for_motion()
                    self._ensure_guidance_program()
                    # self.bridge_robot.resume_program("guidancePos.pro")
                    remaining = original_positions[paused_at_index:]

                    # 对剩余段重新做 S 曲线速度规划
                    if total_time is not None and len(remaining) >= 2:
                        remaining_ratio = len(remaining) / max(len(original_positions), 1)
                        remaining_time = total_time * remaining_ratio
                        dt = ratio * 0.002
                        remaining = resample_s_curve(remaining, remaining_time, dt)
                        logger.info("剩余段 S 曲线重规划: %d 个点, 时间 %.2fs",
                                    len(remaining), remaining_time)

                    logger.info("重新发送剩余 %d 个点", len(remaining))
                    self._send_and_start_position_path(remaining, ratio, tool_no, user_no)
                    self._pause_ctrl.resume_event.clear()

                if not self.is_connected():
                    logger.error("运动期间连接断开")
                    return 0
                time.sleep(0.1)

            if end:
                self.bridge_robot.motion.finalize(MotionType.Path)
            logger.info("笛卡尔路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"笛卡尔路径运动（带暂停）执行异常: {e}")
            return 0

    def move_servo_path(self, positions: List[List[float]],
                        ratio: int = 1, tool_no: int = 10, user_no: int = 0) -> int:
        """
        笛卡尔路径运动（路径模式）— 便捷方法，始终执行完整初始化和finalize
        """
        return self.move_by_position_path(positions, ratio=ratio,
                                          tool_no=tool_no, user_no=user_no,
                                          start=True, end=True)

    def move_by_joint_path(self, joints: List[List[float]], ratio: int = 1,
                        start: bool = True, end: bool = True) -> int:
        """
        关节路径运动（路径模式）

        Args:
            joints: 关节角点列表 [[j1,j2,j3,j4,j5,j6], ...]
            ratio:  插补周期倍率，周期=2ms*ratio，范围1~50
            start:  是否在运动前执行初始化
            end:    是否在运动后执行 finalize

        Returns:
            int: 成功返回1，失败返回0
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

        logger.info(f"开始执行关节路径运动，共{len(joints)}个点, ratio={ratio}")

        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_guidance_program()

            joint_positions = []
            for joint in joints:
                jp = JointPosition(body=joint)
                joint_positions.append(jp)

            if not self.bridge_robot.motion.send_path_joint(joint_positions):
                logger.error("发送关节路径数据失败")
                return 0

            result = self.bridge_robot.motion.move_path(ratio)
            if result != MovePathResult.Success:
                logger.error(f"启动路径运动失败: {result}")
                return 0
            logger.info("关节路径运动启动成功")

            if end:
                self.bridge_robot.motion.finalize(MotionType.PATH)
                self._wait_for_motion_complete()

            logger.info("关节路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"关节路径运动执行异常: {e}")
            return 0

    def move_by_joint_path_with_pause(self, joints: List[List[float]],
                                       ratio: int = 1,
                                       start: bool = True, end: bool = True,
                                       total_time: Optional[float] = None,
                                       pause_timeout: Optional[float] = None) -> int:
        """
        关节路径运动（带暂停/恢复），通过 MotionPauseController 接收外部信号。
        恢复时从暂停位置继续，并对剩余段重新做 S 曲线速度规划。

        Args:
            joints:        关节角点列表 [[j1,j2,j3,j4,j5,j6], ...]
            ratio:         插补周期倍率，1~50
            start:         是否在运动前执行初始化
            end:           是否在运动完成后 finalize
            total_time:    原始轨迹的总运动时间 (s)，传入后恢复段会重新做 S 曲线规划
            pause_timeout: 暂停超时 (s)，None 表示无超时

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

        original_joints = joints
        logger.info(f"开始执行关节路径运动（带暂停），共{len(joints)}个点, ratio={ratio}")

        try:
            if start:
                self._ensure_ready_for_motion()
                self._ensure_guidance_program()

            self._send_and_start_joint_path(joints, ratio)
            logger.info("关节路径运动启动成功")

            while self.is_moving():
                if self._pause_ctrl.pause_event.is_set():
                    paused_at_index = self._get_consumed_index_from_buffer(
                        len(original_joints))
                    logger.info("关节路径运动暂停信号，当前索引 %d / %d (buffer推算)",
                                paused_at_index, len(original_joints))

                    self.bridge_robot.stop_program()
                    self._pause_ctrl.pause_event.clear()

                    pause_start = time.time()
                    while not self._pause_ctrl.resume_event.is_set():
                        if pause_timeout and (time.time() - pause_start > pause_timeout):
                            logger.error("关节路径运动暂停超时")
                            return 0
                        if not self.is_connected():
                            logger.error("暂停期间连接断开")
                            return 0
                        time.sleep(0.1)

                    if paused_at_index >= len(original_joints):
                        logger.info("暂停时运动已完成")
                        break

                    logger.info("关节路径运动恢复，从索引 %d 继续", paused_at_index)
                    self._ensure_guidance_program()
                    remaining = original_joints[paused_at_index:]

                    # 对剩余段重新做 S 曲线速度规划（关节空间）
                    if total_time is not None and len(remaining) >= 2:
                        remaining_ratio = len(remaining) / max(len(original_joints), 1)
                        remaining_time = total_time * remaining_ratio
                        dt = ratio * 0.002
                        remaining = resample_joint_s_curve(remaining, remaining_time, dt)
                        logger.info("剩余段关节 S 曲线重规划: %d 个点, 时间 %.2fs",
                                    len(remaining), remaining_time)

                    logger.info("重新发送剩余 %d 个点", len(remaining))
                    self._send_and_start_joint_path(remaining, ratio)
                    self._pause_ctrl.resume_event.clear()

                if not self.is_connected():
                    logger.error("运动期间连接断开")
                    return 0
                time.sleep(0.1)

            if end:
                self.bridge_robot.motion.finalize(MotionType.Path)
            logger.info("关节路径运动完成")
            return 1

        except Exception as e:
            logger.error(f"关节路径运动（带暂停）执行异常: {e}")
            return 0

    # ── 轨迹规划 ──────────────────────────────────────────────────────────

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
        S曲线规划 + 逆解 + 关节路径执行

        Returns:
            int: 1=成功, 0=失败
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

    def plan_and_save(self, waypoints: List[List[float]],
                      total_time: float, filepath: str,
                      dt: float = 0.008) -> int:
        """
        规划轨迹并保存到文件，不执行运动

        Returns:
            int: 轨迹点数量，0=失败
        """
        traj = self.plan_cartesian_traj(waypoints, total_time, dt)
        if not traj:
            return 0
        save_trajectory(traj, filepath)
        return len(traj)

    def load_and_servo(self, filepath: str, ratio: int = 1,
                       tool_no: int = 10, user_no: int = 0) -> int:
        """
        从文件加载笛卡尔轨迹并执行

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

        Returns:
            int: 1=成功, 0=失败
        """
        traj = load_trajectory(filepath)
        if not traj:
            return 0
        return self.move_by_joint_path(traj, ratio=ratio) 

    def move_and_wait(self, target, timeout: float = 30) -> bool:
        if self.move_linear(target, start=True, end=True) != 1:
            return False
        start = time.time()
        while self.is_moving():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def move_joint_and_wait(self, target, speed, timeout: float = 30) -> bool:
        ik_ref_joint = self.get_joint_pose()
        joint_pose = self.inverse_kinematics(target.to_list(), ik_ref_joint)
        if self.move_by_joint_list(joints=[joint_pose], speeds=[speed]) != 1:
            return False
        start = time.time()
        while self.is_moving():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def relative_tool_pose(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                           drx: float = 0.0, dry: float = 0.0, drz: float = 0.0, init_pose: list = None,
                           mode: str = 'linear', speed: int = 100, wait: bool = True):
        if init_pose is None:
            return -1
        current_carpose = CartesianPose(*init_pose[:6])
        T_base_tcp = pose_to_homogeneous_matrix(current_carpose, degrees=True)
        delta_pose = CartesianPose(x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
        T_tool_delta = pose_to_homogeneous_matrix(delta_pose, degrees=True)
        T_target_base = np.dot(T_base_tcp, T_tool_delta)
        return homogeneous_matrix_to_pose(T_target_base, degrees=True)

    def dh_transform(self, a: float, alpha_rad: float, d: float, theta_rad: float) -> np.ndarray:
        sa = np.sin(alpha_rad)
        ca = np.cos(alpha_rad)
        st = np.sin(theta_rad)
        ct = np.cos(theta_rad)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1],
        ])

    def forward_kinematics(self, joints: List[float], representation: str = 'euler'):
        T = np.eye(4)
        for i in range(len(joints)):
            T = T @ self.dh_transform(
                self.a_list[i],
                np.deg2rad(self.alpha_deg_list[i]),
                self.d_list[i],
                np.deg2rad(joints[i]) + np.deg2rad(self.offset_deg_list[i]),
            )
        if representation == 'matrix':
            return T
        pos = T[:3, 3]
        rot = T[:3, :3]
        rpy = R.from_matrix(rot).as_euler('ZYX', degrees=True)
        return [pos[0], pos[1], pos[2], rpy[2], rpy[1], rpy[0]]

    def inverse_kinematics_no_limit(self, target_pose: List[float], initial_joints: Optional[List[float]] = None,
                           representation: str = 'euler', max_iter: int = 200, tol: float = 1e-6) -> List[float]:
        joint_count = len(self.a_list)
        if initial_joints is None:
            joints = np.zeros(joint_count)
        else:
            joints = np.array(initial_joints, dtype=float)

        target_pos = np.array(target_pose[:3], dtype=float)
        if representation == 'euler':
            rx, ry, rz = target_pose[3], target_pose[4], target_pose[5]
            target_rot = R.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
        elif representation == 'rotvec':
            target_rot = R.from_rotvec(target_pose[3:6], degrees=True).as_matrix()
        else:
            raise ValueError(f'Unsupported representation: {representation}')

        for _ in range(max_iter):
            transform = np.eye(4)
            transform_list = []
            for i in range(joint_count):
                transform = transform @ self.dh_transform(
                    self.a_list[i],
                    np.deg2rad(self.alpha_deg_list[i]),
                    self.d_list[i],
                    np.deg2rad(joints[i]) + np.deg2rad(self.offset_deg_list[i]),
                )
                transform_list.append(transform.copy())

            current_pos = transform[:3, 3]
            current_rot = transform[:3, :3]

            err_pos = target_pos - current_pos
            rot_diff = target_rot @ current_rot.T
            err_rot = R.from_matrix(rot_diff).as_rotvec()
            error = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(error) < tol:
                break

            jacobian = np.zeros((6, joint_count))
            p_end = current_pos
            for i in range(joint_count):
                if i == 0:
                    z_axis = np.array([0.0, 0.0, 1.0])
                    p_axis = np.array([0.0, 0.0, 0.0])
                else:
                    z_axis = transform_list[i - 1][:3, 2]
                    p_axis = transform_list[i - 1][:3, 3]
                jacobian[:3, i] = np.cross(z_axis, p_end - p_axis)
                jacobian[3:, i] = z_axis

            lamda = 0.1
            identity = np.eye(6)
            delta_theta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lamda * identity, error)
            joints = joints + np.rad2deg(delta_theta)

        return joints.tolist()
    
    def inverse_kinematics(self, target_pose: List[float], initial_joints: List[float] , representation='euler',
                       max_iter=200, tol=1e-6):
        joint_count = len(self.a_list)
        
        # 初始化关节角（度）
        if initial_joints is None:
            joints = np.zeros(joint_count, dtype=float)
        else:
            # init_arr = initial_joints
            joints = np.array(initial_joints, dtype=float)
            # init_arr = np.array(initial_joints, dtype=float)
            # init_arr = joints.copy()
            # init_arr = initial_joints

        # joint_count = len(self.a_list)
    
        # if initial_joints is None:
        #     joints = np.zeros(joint_count, dtype=np.float64)
        #     init_arr = None
        # else:
        #     # 手动构造数组，避免 np.array 可能的内存问题
        #     joints = np.empty(joint_count, dtype=np.float64)
        #     for i in range(joint_count):
        #         try:
        #             joints[i] = float(initial_joints[i])
        #         except (TypeError, ValueError, IndexError) as e:
        #             raise ValueError(f"Invalid value at index {i}: {e}")
        #     init_arr = joints.copy()   # 备份初始值
        
        # 解析目标位姿
        target_pos = np.array(target_pose[:3], dtype=float)
        if representation == 'euler':
            rx, ry, rz = target_pose[3], target_pose[4], target_pose[5]
            # 注意：正运动学返回 [x, y, z, roll, pitch, yaw] 对应 ZYX 欧拉角 (yaw, pitch, roll)
            target_rot = R.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
        elif representation == 'rotvec':
            target_rot = R.from_rotvec(target_pose[3:6], degrees=True).as_matrix()
        else:
            raise ValueError(f'Unsupported representation: {representation}')
        
        # 阻尼系数初始值
        lamda = 0.5   # 适当增大初始阻尼，抑制远距离的大步长
        I6 = np.eye(6)
        
        for _ in range(max_iter):
            # ========== 1. 正运动学 + 保存各关节的位姿（用于雅可比） ==========
            T = np.eye(4)
            transforms = []   # 保存每个关节后的变换矩阵（第i个关节后的位姿）
            for i in range(joint_count):
                theta_rad = np.deg2rad(joints[i]) + self.offset_rad_list[i]
                # 使用预计算的 sin/cos alpha
                T = T @ self._dh_matrix(self.a_list[i], self.sin_alpha[i], self.cos_alpha[i],
                                        self.d_list[i], theta_rad)
                transforms.append(T.copy())   # 保存副本（雅可比需要）
            
            current_pos = T[:3, 3]
            current_rot = T[:3, :3]
            
            # ========== 2. 计算误差 ==========
            err_pos = target_pos - current_pos
            rot_diff = target_rot @ current_rot.T
            err_rot = self._rotmat_to_rotvec(rot_diff)   # 高效转换
            error = np.concatenate([err_pos, err_rot])
            error_norm = np.linalg.norm(error)
            
            if error_norm < tol:
                break
            
            # ========== 3. 计算雅可比矩阵（同时利用 transforms） ==========
            jacobian = np.zeros((6, joint_count))
            p_end = current_pos
            for i in range(joint_count):
                if i == 0:
                    z_axis = np.array([0.0, 0.0, 1.0])
                    p_axis = np.array([0.0, 0.0, 0.0])
                else:
                    z_axis = transforms[i-1][:3, 2]   # 第i-1个变换后的z轴方向
                    p_axis = transforms[i-1][:3, 3]   # 第i-1个变换的原点位置
                jacobian[:3, i] = np.cross(z_axis, p_end - p_axis)
                jacobian[3:, i] = z_axis
            
            # ========== 4. 阻尼最小二乘（DLS）求解增量 ==========
            JtJ = jacobian.T @ jacobian
            Jte = jacobian.T @ error
            delta_theta = np.linalg.solve(JtJ + lamda * I6, Jte)   # 优化：直接解正规方程
            
            # ========== 5. 更新关节角（度） ==========
            new_joints = joints + np.rad2deg(delta_theta)
            
            # ----- 动态调整阻尼（根据误差变化） -----
            # 试算新误差（只做一次正运动学前向传播？为了性能，快速评估误差太耗时，改用启发式）
            # 简单方法：若误差增大则增大阻尼，否则减小
            # 计算新误差的近似：快速计算末端位置变化很麻烦，这里采用幅度比较
            # 若 delta_theta 过大（超过30度），则增大阻尼下次迭代减速
            if np.max(np.abs(delta_theta)) > 0.5:   # 单步变化超过30度？0.5 rad ≈ 28.6°
                lamda = min(10.0, lamda * 1.2)
            else:
                lamda = max(0.01, lamda * 0.9)
            
            # ========== 6. 关节限位处理（核心修复） ==========
            for i, (low, high) in enumerate(self.joint_limits):
                # 先归一化到 [-360,360) 对于全周关节
                if low == -360 and high == 360:
                    # 映射到 [-360, 360)
                    new_joints[i] = ((new_joints[i] + 360) % 720) - 360
                else:
                    # 有限位关节：直接钳位，并可选映射到周期范围（但一般不需要模运算）
                    if new_joints[i] < low:
                        new_joints[i] = low
                    elif new_joints[i] > high:
                        new_joints[i] = high
            joints = new_joints
        
        # 最终将关节角限制在物理范围内（确保输出有效）
        for i, (low, high) in enumerate(self.joint_limits):
            if low == -360 and high == 360:
                joints[i] = ((joints[i] + 360) % 720) - 360
            else:
                joints[i] = np.clip(joints[i], low, high)
        
        # for i, (low, high) in enumerate(self.joint_limits):
        #     if low == -360 and high == 360:   # 仅对全周关节处理
        #         # 归一化到 [-360, 360)
        #         joints[i] = ((joints[i] + 360) % 720) - 360
        #         # 调整到与初始值最近的周期
        #         diff = joints[i] - init_arr[i]
        #         if diff > 180:
        #             joints[i] -= 360
        #         elif diff < -180:
        #             joints[i] += 360
        #         # 双重保险，确保在限位内（实际已在范围内）
        #         joints[i] = np.clip(joints[i], low, high)


        
        return joints.tolist()
    
def main():

    bridge_robot = BridgeCRobotAdapter(ip='192.168.1.12', so_path='/home/nvidia/Downloads/HD/HD_0701/third_party/crp_robot_sdk/libRobotService.so')
    bridge_robot.connect()
    bridge_robot.set_speed(5)
    pose = bridge_robot.get_tcp_pose()
    joint = bridge_robot.get_joint_pose()
    print('当前关节角：',joint)
    print('当前位姿：',pose)

    # if pose and joint:
    #     base_pose = pose
    #     waypoints = [
    #         base_pose,
    #         [base_pose[0] + 50.0, base_pose[1],         base_pose[2],
    #             base_pose[3],         base_pose[4],         base_pose[5]],
    #         [base_pose[0] + 50.0, base_pose[1] , base_pose[2] + 50.0,
    #             base_pose[3],         base_pose[4],         base_pose[5]],
    #         base_pose,
    #     ]
    #     traj = bridge_robot.plan_cartesian_traj(waypoints=waypoints,total_time=5)
        
        # save_trajectory(trajectory=traj, filepath='/home/nvidia/Downloads/HD/HD_0701/third_party/force_control_crp/traj.txt ')

        # 逆解测试
        # bridge_robot.move_by_pose_list(poses=waypoints, speeds=[50, 50, 50, 50])
        # positions = bridge_robot.plan_cartesian_traj(waypoints=waypoints, total_time=15,profile='trapezoid', accel_frac=0.2)
        # print("轨迹点：",len(positions))
        # visualize_trajectory(trajectory=positions)
        # bridge_robot.move_by_position_path_with_pause(positions=positions,ratio=4,tool_no=10, user_no=0,total_time=5)
        # bridge_robot.plan_and_move_position(waypoints = waypoints,total_time=10, profile='trapezoid',accel_frac=0.2)

        # bridge_robot.plan_and_move_position(waypoints, total_time=10.0, dt=0.008, tool_no=10, user_no=0)
        # bridge_robot.move_linear_new(CartesianPose(*waypoints[1]))
    
    #     forward_pose = bridge_robot.relative_tool_pose(dz = 50, init_pose = pose).to_list()
    #     bridge_robot.move_by_pose_list(poses=[forward_pose,[-51.35, 411.64, 259.08, 107.54, 16.32, -154.04] pose], speeds=[50, 50])
    #     start_time = time.time()
    #     bridge_robot.switch_motion_model()
    #     end_time = time.time()
    #     print("time: ", end_time - start_time)
    #     bridge_robot.plan_and_move_position(waypoints, total_time=2.0, dt=0.008, tool_no=10, user_no=0)
    #     bridge_robot.switch_motion_model()
    #     bridge_robot.move_by_pose_list(poses=[forward_pose,pose], speeds=[50, 50])

    # target_pose =  [-775.3937383948448, 781.6924120558483, -69.80050462569193, 99.87115777295497, 19.744255861168373, -158.38591186339445]
    # bridge_robot.move_by_pose_list(poses=[target_pose],speeds=[50])
    joint = [200.528, 86.671, -59.181, 144.788, -3.059, -17.912]
    pose = [-51.35, 411.64, 259.08, 107.54, 16.32, -154.04]
    target_joint = bridge_robot.inverse_kinematics(target_pose = pose, initial_joints=joint)
    print("目标关节角: ",target_joint)
    # bridge_robot.move_joint([112.196, 99.884, -56.131, 128.446, -5.912, -17.849], speed=40)


        # bridge_robot.close()
          
    # else:
    #     logger.error("连接失败")




if __name__=='__main__':
    main()
    
