"""
视觉伺服 OSD 绘制 | Visual Servo OSD

在画面上叠加伺服状态信息。
"""

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICK = 2


def put_text(img, text, y, color=(200, 200, 200)):
    """基础文字绘制。"""
    cv2.putText(img, text, (20, y), FONT, FONT_SCALE, color, FONT_THICK)


def draw_servo_status(vis, y, gain, trans_err, rot_err, target_pose,
                      moving, step_history, ref_set=True, auto_mode=False,
                      auto_step=0, auto_max_steps=0, trans_xyz=None):
    """
    绘制伺服状态 OSD 块。

    Args:
        trans_xyz: ndarray (3,) — 参考坐标系下 [dx, dy, dz] (mm)，可选

    Returns:
        new_y: 绘制结束后的 y 坐标
    """
    # Gain
    gain_color = (0, 255, 0) if not moving else (100, 100, 100)
    put_text(vis, f"Gain: {gain:.2f}", y, gain_color)
    y += 30

    # 误差
    if trans_err is not None:
        if trans_err < 2.0:
            err_color = (0, 255, 0)
        elif trans_err < 10.0:
            err_color = (0, 200, 255)
        else:
            err_color = (0, 0, 255)
        put_text(vis, f"Error: {trans_err:.2f} mm  {rot_err:.2f} deg", y, err_color)
        y += 30

        # 参考坐标系下分轴误差
        if trans_xyz is not None:
            put_text(vis, f"  dX={trans_xyz[0]:+.2f}  dY={trans_xyz[1]:+.2f}  dZ={trans_xyz[2]:+.2f} mm",
                     y, err_color)
            y += 28

        # 目标 TCP
        if target_pose is not None:
            put_text(vis, f"Target: X={target_pose[0]:.1f} Y={target_pose[1]:.1f} Z={target_pose[2]:.1f}",
                     y, (255, 200, 0))
            y += 28
            put_text(vis, f"        Rx={target_pose[3]:.2f} Ry={target_pose[4]:.2f} Rz={target_pose[5]:.2f}",
                     y, (255, 200, 0))
            y += 30
    elif not ref_set:
        put_text(vis, "Press r at ref position to set reference", y, (0, 165, 255))
        y += 30

    # 自动模式状态
    if auto_mode:
        put_text(vis, f"AUTO  step {auto_step}/{auto_max_steps}", y, (0, 255, 255))
        y += 30

    # 运动状态
    if moving:
        put_text(vis, "MOVING...", y, (0, 100, 255))
        y += 30

    # 步进历史（最近5步）
    if step_history:
        put_text(vis, "Step history (trans_mm, rot_deg):", y, (180, 180, 180))
        y += 25
        recent = step_history[-5:]
        start_idx = len(step_history) - len(recent) + 1
        for i, (t_e, r_e) in enumerate(recent):
            put_text(vis, f"  #{start_idx + i}: {t_e:.2f} mm  {r_e:.2f} deg", y, (160, 160, 160))
            y += 22

    return y


def draw_status_bar(vis, ref_set, robot_str, gain, keys_help, auto_mode=False):
    """在画面底部绘制状态栏。"""
    h = vis.shape[0]
    ref_str = "Ref:SET" if ref_set else "Ref:NONE"
    mode_str = "AUTO" if auto_mode else "MANUAL"
    status = f"{mode_str} | {ref_str} | {robot_str} | Gain={gain:.2f} | {keys_help}"
    put_text(vis, status, h - 20, (140, 140, 140))
