"""
视觉伺服通用运行辅助 | Shared Servo Runner Helpers
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

import cv2

from robovision.geometry.transforms import matrix_to_pose
from robovision.servo.core import compute_step_pose, check_step_safety, execute_move
from robovision.servo.osd import put_text, draw_servo_status, draw_status_bar

logger = logging.getLogger(__name__)


@dataclass
class ServoFrameState:
    frame: Optional[object]
    aruco_result: dict
    target_data: Optional[dict]
    tcp: Optional[tuple]
    T_g2b_cur: Optional[object]
    quality: dict
    T_g2b_target: Optional[object]
    trans_err: Optional[float]
    rot_err: Optional[float]
    target_pose: Optional[list]


def build_frame_state(session) -> ServoFrameState:
    """读取一帧并构造当前伺服状态。"""
    frame, aruco_result, target_data, tcp, T_g2b_cur = session.read_and_detect()
    quality = session.evaluate_target_quality(target_data)

    if quality['allow_control']:
        T_g2b_target, trans_err, rot_err = session.compute_target(target_data, tcp, T_g2b_cur)
    else:
        T_g2b_target, trans_err, rot_err = (None, None, None)
    target_pose = matrix_to_pose(T_g2b_target) if T_g2b_target is not None else None

    return ServoFrameState(
        frame=frame,
        aruco_result=aruco_result,
        target_data=target_data,
        tcp=tcp,
        T_g2b_cur=T_g2b_cur,
        quality=quality,
        T_g2b_target=T_g2b_target,
        trans_err=trans_err,
        rot_err=rot_err,
        target_pose=target_pose,
    )


def render_frame(session, state: ServoFrameState, gain: float, moving: bool,
                 step_history: List[Tuple[float, float]], keys_help: str,
                 auto_mode: bool = False, auto_step: int = 0, auto_max_steps: int = 0):
    """绘制统一 OSD，并返回缩放后的显示图。"""
    vis = session.draw_aruco(state.frame, state.aruco_result, state.tcp)
    h, w = vis.shape[:2]
    y = 35 + len(state.aruco_result) * 100 + 80

    if not state.target_data:
        put_text(vis, f"ID{session.target_id} NOT DETECTED", y, (0, 0, 255))
        y += 40

    if state.target_data is not None:
        quality_color = {
            'OK': (0, 255, 0),
            'WARN': (0, 165, 255),
            'REJECT': (0, 0, 255),
        }[state.quality['level']]
        put_text(vis, f"Target Quality: {state.quality['level']} | {state.quality['reason']}",
                 y, quality_color)
        y += 30

    if state.T_g2b_target is None and state.target_data is not None and session.no_robot:
        put_text(vis, "ArUco detected (no-robot mode)", y, (0, 200, 0))
        y += 30

    y = draw_servo_status(
        vis, y, gain, state.trans_err, state.rot_err, state.target_pose,
        moving, step_history, ref_set=session.ref_set,
        auto_mode=auto_mode, auto_step=auto_step, auto_max_steps=auto_max_steps,
    )
    draw_status_bar(vis, session.ref_set, session.robot_status_str, gain,
                    keys_help, auto_mode=auto_mode)

    scale = 0.4
    return cv2.resize(vis, (int(w * scale), int(h * scale)))


def run_pbvs_step(session, state: ServoFrameState, gain: float, no_robot: bool = False):
    """执行一步 PBVS，返回 (ok, reason, trans_err, rot_err)。"""
    if not state.quality['allow_manual_move']:
        return False, state.quality['reason'], state.trans_err, state.rot_err
    if state.T_g2b_target is None or state.T_g2b_cur is None:
        return False, "无法计算目标（检查参考位姿/ArUco检测/机械臂连接）", state.trans_err, state.rot_err

    T_step, step_trans, step_rot = compute_step_pose(state.T_g2b_cur, state.T_g2b_target, gain)
    step_pose = matrix_to_pose(T_step)

    logger.info("步进计算: 平移=%.2f mm, 旋转=%.2f deg (gain=%.2f)",
                step_trans, step_rot, gain)
    logger.info("  目标 TCP: X=%.2f Y=%.2f Z=%.2f Rx=%.2f Ry=%.2f Rz=%.2f",
                *step_pose)

    safe, reason = check_step_safety(step_trans, step_rot)
    if not safe:
        return False, reason, state.trans_err, state.rot_err

    if no_robot:
        logger.info("[DRY RUN] 不执行运动")
        return True, "dry-run", state.trans_err, state.rot_err

    ok = execute_move(session.robot, step_pose)
    return ok, ("ok" if ok else "运动失败"), state.trans_err, state.rot_err
