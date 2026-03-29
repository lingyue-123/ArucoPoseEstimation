"""
ArUco 检测结果可视化 | ArUco Result Overlay

整合自 aruco_pose_from_hkws_cam_multi_aruco.py 的 draw_aruco()。
"""

import cv2
import numpy as np
from typing import Optional

from robovision.cameras.base import CameraIntrinsics


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.8
FONT_SCALE_MID = 1.0
CORNER_RADIUS = 8
LINE_THICKNESS = 2


def draw_aruco_result(
    frame: np.ndarray,
    aruco_result: dict,
    intrinsics: CameraIntrinsics,
    robot_pose: Optional[tuple] = None,
    use_kalman: bool = True,
    robot_connected: bool = False,
    saved_counts: Optional[dict] = None,
    saved_images: int = 0,
) -> np.ndarray:
    """
    在图像上绘制 ArUco 检测结果。

    Args:
        frame: 原始 BGR 图像
        aruco_result: ArucoDetector.detect() 的返回值
        intrinsics: 相机内参（用于绘制坐标轴）
        robot_pose: (x, y, z, rx, ry, rz) 机械臂 TCP 位姿，None 表示未连接
        use_kalman: 是否启用卡尔曼滤波（用于状态显示）
        robot_connected: 机械臂是否已连接
        saved_counts: {marker_id: count} 已保存位姿计数
        saved_images: 已保存图片数量

    Returns:
        绘制后的 BGR 图像
    """
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    # 灰度输入转 BGR（mono 相机直接输出 gray）
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()
    h, w = vis.shape[:2]
    info_y = 40

    for marker_id, data in aruco_result.items():
        corners = data['filtered_corners'].astype(int)
        rvec = data['rvec_m2c']
        tvec = data['tvec_m2c']
        euler = data['euler_m2c_zyx']
        reproj = data['reproj_err']
        method = data.get('method', '')
        marker_length = data.get('marker_length', 100.0)

        # 颜色：HOLD=红，WARN=橙，正常=青
        if 'HOLD' in method:
            box_color = (0, 0, 255)
        elif '!' in method:
            box_color = (0, 165, 255)
        else:
            box_color = (255, 255, 0)

        cv2.polylines(vis, [corners], True, box_color, LINE_THICKNESS)
        for j, (cx, cy) in enumerate(corners):
            cv2.circle(vis, (cx, cy), CORNER_RADIUS, (0, 255, 0), -1)
            cv2.putText(vis, str(j + 1), (cx + 6, cy - 6), FONT, FONT_SCALE_SMALL,
                        (0, 255, 0), LINE_THICKNESS)
            cv2.putText(vis, f"({cx},{cy})", (cx + 12, cy + 28), FONT, 0.7,
                        (0, 200, 0), 2)

        center = (int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1])))
        size_cm = marker_length / 10
        cv2.putText(vis, f"ID:{marker_id}({size_cm:.1f}cm)", (center[0] - 40, center[1]),
                    FONT, FONT_SCALE_MID, box_color, LINE_THICKNESS)
        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, marker_length / 2, LINE_THICKNESS)

        t = tvec.flatten()
        cv2.putText(vis, f"[ArUco ID{marker_id} {size_cm:.1f}cm] {method} reproj={reproj:.2f}px",
                    (20, info_y), FONT, FONT_SCALE_SMALL, (200, 200, 200), LINE_THICKNESS)
        cv2.putText(vis, f"[M->C] t(mm): X:{t[0]:.2f} Y:{t[1]:.2f} Z:{t[2]:.2f}",
                    (20, info_y + 30), FONT, FONT_SCALE_SMALL, (255, 0, 0), LINE_THICKNESS)
        cv2.putText(vis, f"[M->C] Euler(ZYX): {euler[0]:.1f} {euler[1]:.1f} {euler[2]:.1f} deg",
                    (20, info_y + 60), FONT, FONT_SCALE_SMALL, (255, 0, 0), LINE_THICKNESS)
        info_y += 100

    # 机械臂位姿信息
    if robot_pose is not None:
        cv2.putText(vis, f"[TCP] t(mm): X:{robot_pose[0]:.2f} Y:{robot_pose[1]:.2f} Z:{robot_pose[2]:.2f}",
                    (20, info_y), FONT, FONT_SCALE_SMALL, (0, 255, 255), LINE_THICKNESS)
        cv2.putText(vis, f"[TCP] Euler(XYZ): {robot_pose[3]:.1f} {robot_pose[4]:.1f} {robot_pose[5]:.1f} deg",
                    (20, info_y + 30), FONT, FONT_SCALE_SMALL, (0, 255, 255), LINE_THICKNESS)
    else:
        cv2.putText(vis, "[TCP] NOT CONNECTED",
                    (20, info_y), FONT, FONT_SCALE_SMALL, (0, 0, 255), LINE_THICKNESS)

    # 状态栏
    kalman_str = "ON" if use_kalman else "OFF"
    conn_str = "Connected" if robot_connected else "Disconnected"
    total_saved = sum(saved_counts.values()) if saved_counts else 0
    cv2.putText(
        vis,
        f"Kalman:{kalman_str} | ArUco:{len(aruco_result)} | Robot:{conn_str} | Saved:{total_saved} | Imgs:{saved_images}",
        (20, h - 50), FONT, FONT_SCALE_MID, (255, 255, 255), LINE_THICKNESS,
    )
    cv2.putText(vis, "s=Save | c=Clear | r=Reconnect | p=Print | q=Quit",
                (20, h - 20), FONT, FONT_SCALE_MID, (200, 200, 200), LINE_THICKNESS)

    return vis
