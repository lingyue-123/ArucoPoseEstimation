#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco 光照鲁棒性实验 v2 | ArUco Lighting Robustness Test v2

v2 改进: 光照鲁棒线程支持两级调整（增益优先 → 曝光兜底）。

实时监控 ArUco Marker-to-Camera ([M->C]) 位姿在外部光照变化下的表现。
仅连接相机，不连接机械臂。核心观察指标：[M->C] 平移/旋转 + ROI 亮度。

用法 (Usage):
    python scripts/aruco_lighting_robustness_v2.py --camera mecheye
    python scripts/aruco_lighting_robustness_v2.py --camera mecheye --marker-ids 0,1
    python scripts/aruco_lighting_robustness_v2.py --camera mecheye --raw

工作流 (Workflow):
    1. 启动后实时显示所有 marker 的 [M->C] 位姿
    2. 按 'r': 保存当前帧 [M->C] 为参考，显示误差 vs 参考
    3. 按 'c': 清除参考
    4. 按 's': 记录当前帧数据到终端（用于数据采集）
    5. 按 'q' / ESC: 退出
"""

import argparse
import logging
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from robovision.cameras import build_camera
from robovision.config.loader import get_config
from robovision.detection.aruco import (
    ArucoDetector, build_raw_aruco_detector, detect_raw_frame,
)
from robovision.servo.core import aruco_to_matrix, compute_pose_error_in_frame

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# ============================================================
# 常量 | Constants
# ============================================================

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.5            # 原 0.5 × 3
_FONT_THICK = 2              # 原 1   × 2
_FONT_SCALE_MARKER = 2.4     # draw_marker_overlay 内部文字 (原 0.8 × 3)
_FONT_SCALE_MARKER_ID = 3.0  # marker ID 标签 (原 1.0 × 3)
_FONT_SCALE_CORNER = 1.6     # 角点坐标标签 (原 0.7 × 2.3)
_CORNER_RADIUS = 8
_LINE_THICKNESS = 2

GAIN_LIMIT_DB = 12.0          # 光照鲁棒线程增益上限
LR_DEADBAND = 1               # 光照鲁棒线程亮度死区（±）
LR_GAIN_STEP = 0.4            # 光照鲁棒线程增益每步调整量（dB）
LR_EXP_LIMIT_US = 100000      # 曝光时间上限（μs）
LR_EXP_MIN_US = 100           # 曝光时间下限（μs）
LR_EXP_RATIO = 0.05           # 曝光每步调整比例（5%）


# ============================================================
# 本地辅助函数 | Local Helpers
# ============================================================

def put_text(img, text, y, color=(200, 200, 200)):
    """在图像上绘制带黑色描边的文本。"""
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2)
    cv2.putText(img, text, (20, y), _FONT, _FONT_SCALE, color, _FONT_THICK)


def _marker_roi_brightness(frame, target_data):
    """从 ArUco 检测结果提取 marker ROI 并返回中值亮度。失败返回 None。"""
    corners = target_data.get('filtered_corners')
    if corners is None:
        corners = target_data.get('raw_corners')
    if corners is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    x1 = max(0, int(corners[:, 0].min() - 20))
    y1 = max(0, int(corners[:, 1].min() - 20))
    x2 = min(w, int(corners[:, 0].max() + 20))
    y2 = min(h, int(corners[:, 1].max() + 20))
    if x2 <= x1 or y2 <= y1:
        return None
    return float(np.median(gray[y1:y2, x1:x2]))


def draw_marker_overlay(frame, aruco_result, intrinsics):
    """
    在图像上绘制 ArUco marker 框 + 坐标轴 + [M->C] 数据。
    去掉原 draw_aruco_result 中的 robot/TCP/保存计数等无关信息。
    """
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()
    info_y = 120

    for marker_id, data in aruco_result.items():
        corners = data['filtered_corners'].astype(int)
        rvec = data['rvec_m2c']
        tvec = data['tvec_m2c']
        euler = data['euler_m2c_zyx']
        reproj = data['reproj_err']
        method = data.get('method', '')
        marker_length = data.get('marker_length', 100.0)

        if 'HOLD' in method:
            box_color = (0, 0, 255)
        elif '!' in method:
            box_color = (0, 165, 255)
        else:
            box_color = (255, 255, 0)

        cv2.polylines(vis, [corners], True, box_color, _LINE_THICKNESS)
        center = (int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1])))
        for j, (cx, cy) in enumerate(corners):
            cv2.circle(vis, (cx, cy), _CORNER_RADIUS, (0, 255, 0), -1)
            # 角点编号：沿角点→中心方向的反方向偏移
            dx = cx - center[0]
            dy = cy - center[1]
            length = max(abs(dx), abs(dy), 1)
            dx_u, dy_u = int(dx / length * 40), int(dy / length * 40)
            cv2.putText(vis, str(j + 1), (cx + dx_u - 10, cy + dy_u - 10),
                        _FONT, _FONT_SCALE_CORNER, (0, 255, 0), _LINE_THICKNESS)
            # 坐标值：在编号外侧再偏移
            cv2.putText(vis, f"({cx},{cy})",
                        (cx + dx_u + 50, cy + dy_u + 20),
                        _FONT, _FONT_SCALE_CORNER, (0, 220, 0), _LINE_THICKNESS)

        size_cm = marker_length / 10
        cv2.putText(vis, f"ID:{marker_id}({size_cm:.1f}cm)", (center[0] - 80, center[1]),
                    _FONT, _FONT_SCALE_MARKER_ID, box_color, _LINE_THICKNESS)
        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, marker_length / 2, _LINE_THICKNESS)

        t = tvec.flatten()
        cv2.putText(vis, f"[M->C] ID{marker_id}  {method}  reproj={reproj:.2f}px",
                    (20, info_y), _FONT, _FONT_SCALE_MARKER, (200, 200, 200), _LINE_THICKNESS)
        cv2.putText(vis, f"  t(mm):  X={t[0]:.2f}  Y={t[1]:.2f}  Z={t[2]:.2f}",
                    (20, info_y + 80), _FONT, _FONT_SCALE_MARKER, (255, 0, 0), _LINE_THICKNESS)
        cv2.putText(vis, f"  Euler(ZYX):  rx={euler[0]:.1f}  ry={euler[1]:.1f}  rz={euler[2]:.1f} deg",
                    (20, info_y + 160), _FONT, _FONT_SCALE_MARKER, (255, 0, 0), _LINE_THICKNESS)
        info_y += 300

    return vis, info_y


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='ArUco 光照鲁棒性实验')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--marker-ids', type=str, default='0,1',
                        help='监测的 marker ID，逗号分隔（默认 0,1）')
    parser.add_argument('--raw', action='store_true',
                        help='使用 RAW 检测模式（固定 IPPE_SQUARE，不使用时序滤波）')
    parser.add_argument('--no-temporal-filter', action='store_true',
                        help='关闭卡尔曼角点滤波和位姿时序平滑（仅影响标准模式）')
    parser.add_argument('--lighting-robust', action='store_true',
                        help=f'启用光照鲁棒线程：增益优先→曝光兜底（死区±{LR_DEADBAND}，增益步长{LR_GAIN_STEP}dB，曝光步长{LR_EXP_RATIO*100:.0f}%）')
    parser.add_argument('--record-interval', type=float, default=1.0,
                        help='按 s 记录的最小间隔秒数（默认 1.0）')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 解析 marker ID 列表
    marker_ids = [int(x.strip()) for x in args.marker_ids.split(',') if x.strip()]

    # 加载全局配置
    cfg = get_config()
    marker_cfg = cfg.get_marker()
    detection_cfg = cfg.get_detection()

    # 从配置获取 valid_ids 和 marker_sizes（作为 fallback）
    all_valid_ids = set(marker_cfg.valid_ids)
    all_marker_sizes = marker_cfg.marker_sizes

    # --- 初始化相机 | Camera init ---
    camera = build_camera(args.camera, cfg)
    camera.open()
    intrinsics = camera.get_intrinsics()
    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs

    use_raw = args.raw
    use_temporal_filter = not args.no_temporal_filter
    if use_raw:
        raw_detector = build_raw_aruco_detector(marker_cfg.dictionary)
        valid_ids = all_valid_ids
        marker_sizes = all_marker_sizes
        detector = None
        logger.info("Detection mode: RAW (IPPE_SQUARE, no filter)")
    else:
        detector = ArucoDetector.from_config(intrinsics, marker_cfg, detection_cfg)
        detector.set_temporal_filter(use_temporal_filter)
        raw_detector = None
        logger.info("Detection mode: ArucoDetector (%s, multi-method PnP)",
                    "temporal filter ON" if use_temporal_filter else "temporal filter OFF")

    # --- 参考位姿存储 | Reference poses ---
    T_refs = {}    # {marker_id: 4x4 matrix}
    ref_bris = {}  # {marker_id: 参考亮度}  — 与 T_refs 同时记录/清除
    ref_set_ids = set()

    win = f"ArUco Lighting Robustness [{args.camera}] IDs={args.marker_ids}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    det_lock = threading.Lock()
    latest_det = {}
    latest_det_frame = None
    stop_event = threading.Event()
    last_record_time = 0.0

    def _detection_loop():
        nonlocal latest_det, latest_det_frame
        while not stop_event.is_set():
            ok_d, frm = camera.read_frame()
            if not ok_d or frm is None:
                time.sleep(0.005)
                continue
            frm = frm.copy()
            if use_raw:
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY) if frm.ndim == 3 else frm
                result = detect_raw_frame(gray, raw_detector, valid_ids, marker_sizes, K, dist)
            else:
                result = detector.detect(frm)
            with det_lock:
                latest_det = result
                latest_det_frame = frm

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()
    logger.info("Ready: r=SetRef c=ClearRef s=Record q=Quit (detection thread started)")

    # --- 光照鲁棒线程（增益优先 → 曝光兜底） ---
    if args.lighting_robust:
        def _lighting_robust_loop():
            while not stop_event.is_set():
                if not ref_bris:
                    time.sleep(1.0)
                    continue
                with det_lock:
                    cur_frame = latest_det_frame
                    cur_det = latest_det
                if cur_frame is None or not cur_det:
                    time.sleep(1.0)
                    continue

                deviations = []
                for mid, ref_bri in ref_bris.items():
                    if mid in cur_det:
                        cur_bri = _marker_roi_brightness(cur_frame, cur_det[mid])
                        if cur_bri is not None:
                            deviations.append(cur_bri - ref_bri)

                if not deviations:
                    time.sleep(0.15)
                    continue

                avg_dev = sum(deviations) / len(deviations)

                if abs(avg_dev) <= LR_DEADBAND:
                    time.sleep(0.15)
                    continue

                try:
                    cur_gain = camera.get_gain()
                    if cur_gain is None:
                        cur_gain = 0.0
                    cur_exp = camera.get_exposure_time()
                    if cur_exp is None:
                        cur_exp = 5000.0  # 默认 5ms

                    if avg_dev < 0:
                        # 偏暗：先增增益，增益到顶则增曝光
                        if cur_gain < GAIN_LIMIT_DB - LR_GAIN_STEP:
                            new_gain = min(cur_gain + LR_GAIN_STEP, GAIN_LIMIT_DB)
                            camera.set_gain(new_gain)
                            logger.debug("Lighting adj: dev=%.1f  gain %.1f→%.1f dB",
                                         avg_dev, cur_gain, new_gain)
                        else:
                            new_exp = min(cur_exp * (1.0 + LR_EXP_RATIO), LR_EXP_LIMIT_US)
                            if new_exp / cur_exp - 1.0 > 0.001:
                                camera.set_exposure_time(new_exp)
                                logger.debug("Lighting adj: dev=%.1f  exp %.0f→%.0f us",
                                             avg_dev, cur_exp, new_exp)
                    else:
                        # 偏亮：先减增益，增益到底则减曝光
                        if cur_gain > LR_GAIN_STEP:
                            new_gain = max(cur_gain - LR_GAIN_STEP, 0.0)
                            if cur_gain - new_gain > 0.001:
                                camera.set_gain(new_gain)
                                logger.debug("Lighting adj: dev=%.1f  gain %.1f→%.1f dB",
                                             avg_dev, cur_gain, new_gain)
                        else:
                            new_exp = max(cur_exp * (1.0 - LR_EXP_RATIO), LR_EXP_MIN_US)
                            if 1.0 - new_exp / cur_exp > 0.001:
                                camera.set_exposure_time(new_exp)
                                logger.debug("Lighting adj: dev=%.1f  exp %.0f→%.0f us",
                                             avg_dev, cur_exp, new_exp)
                except Exception:
                    pass

                time.sleep(0.15)

        lr_thread = threading.Thread(target=_lighting_robust_loop, daemon=True)
        lr_thread.start()
        logger.info("Lighting robustness thread started (gain-first then exp, deadband=%d)", LR_DEADBAND)

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            with det_lock:
                aruco_result = latest_det
                det_frame = latest_det_frame

            # 过滤出我们关心的 marker IDs
            observed = {mid: d for mid, d in aruco_result.items() if mid in marker_ids}

            # 绘制 marker 叠加层
            vis, overlay_end_y = draw_marker_overlay(frame, observed, intrinsics)
            h_disp, w_disp = vis.shape[:2]
            y = overlay_end_y + 20

            # 显示每个 marker 的 [M->C] 误差 vs 参考
            for mid in marker_ids:
                if mid not in observed:
                    put_text(vis, f"ID{mid}: not detected", y, (0, 0, 255))
                    y += 70
                    continue

                td = observed[mid]
                T_cur = aruco_to_matrix(td)

                # 显示当前 [M->C] 摘要
                t = td['tvec_m2c'].flatten()
                euler = td['euler_m2c_zyx']
                put_text(vis,
                    f"ID{mid} [M->C]: t=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) mm  "
                    f"Euler=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
                    y, (255, 100, 0))
                y += 65

                # 误差 vs 参考
                if T_refs.get(mid) is not None:
                    xyz, norm, rot = compute_pose_error_in_frame(T_cur, T_refs[mid])
                    if norm < 2.0:
                        err_color = (0, 255, 0)
                    elif norm < 10.0:
                        err_color = (0, 200, 255)
                    else:
                        err_color = (0, 0, 255)
                    put_text(vis,
                        f"  Err vs Ref: d=({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f}) mm  "
                        f"|trans|={norm:.2f} mm  rot={rot:.2f} deg",
                        y, err_color)
                    y += 65
                elif ref_set_ids:
                    put_text(vis, f"  (Ref set for IDs: {sorted(ref_set_ids)})", y, (140, 140, 140))
                    y += 60
                else:
                    put_text(vis, "  Press r to set reference", y, (140, 140, 140))
                    y += 60

            # ROI 亮度
            roi_bri_strs = []
            for mid in marker_ids:
                if mid in observed:
                    bri = _marker_roi_brightness(frame, observed[mid])
                    if bri is not None:
                        roi_bri_strs.append(f"ID{mid}={bri:.0f}")
            if roi_bri_strs:
                put_text(vis, f"ROI Bri: {', '.join(roi_bri_strs)}", y, (0, 255, 128))
            else:
                put_text(vis, "ROI Bri: None", y, (100, 100, 100))
            y += 65

            # 参考亮度对比
            if ref_bris:
                bri_ref_strs = []
                for mid in marker_ids:
                    if mid in observed and mid in ref_bris:
                        cur_bri = _marker_roi_brightness(frame, observed[mid])
                        if cur_bri is not None:
                            bri_ref_strs.append(
                                f"ID{mid}={ref_bris[mid]:.0f} d={cur_bri - ref_bris[mid]:+.0f}")
                if bri_ref_strs:
                    put_text(vis, f"Bri Ref: {', '.join(bri_ref_strs)}", y, (0, 200, 100))
                    y += 65

            # 曝光 / 增益
            try:
                exp = camera.get_exposure_time()
                gain_db = camera.get_gain()
                if exp is not None:
                    exp_str = f"Exposure: {exp/1000:.2f}ms  Gain: {gain_db:.1f}dB" if gain_db is not None else f"Exposure: {exp/1000:.2f}ms"
                    put_text(vis, exp_str, y, (180, 180, 180))
                    y += 65
            except Exception:
                pass

            # 底部状态栏
            kalman_str = "ON" if (not use_raw and use_temporal_filter) else "OFF"
            ref_str = f"Ref:{len(ref_set_ids)}/{len(marker_ids)}" if ref_set_ids else "Ref:NONE"
            mode_str = "RAW" if use_raw else "STD"
            seen = len(observed)
            lr_str = " LR" if args.lighting_robust else ""
            status = (f"LIGHTING TEST | Mode:{mode_str}{lr_str} | Kalman:{kalman_str} | "
                      f"ArUco:{seen}/{len(marker_ids)} | {ref_str} | "
                      f"r=Ref c=Clear s=Record q=Quit")
            cv2.putText(vis, status, (10, h_disp - 25), _FONT, _FONT_SCALE,
                        (140, 140, 140), _LINE_THICKNESS)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('r'):
                # 保存当前检测到的所有 marker 的 [M->C] 为参考
                if not observed:
                    logger.warning("No marker detected, cannot save reference")
                    continue
                T_refs.clear()
                ref_bris.clear()
                ref_set_ids.clear()
                for mid in observed:
                    T_refs[mid] = aruco_to_matrix(observed[mid]).copy()
                    ref_set_ids.add(mid)
                    t = observed[mid]['tvec_m2c'].flatten()
                    bri = _marker_roi_brightness(frame, observed[mid])
                    if bri is not None:
                        ref_bris[mid] = bri
                        logger.info("Ref saved ID%d: [M->C] t=(%.2f, %.2f, %.2f) mm  bri=%.0f", mid, *t, bri)
                    else:
                        logger.info("Ref saved ID%d: [M->C] t=(%.2f, %.2f, %.2f) mm  bri=None", mid, *t)
                logger.info("Reference set for %d marker(s): %s", len(T_refs), sorted(T_refs.keys()))

            elif key == ord('c'):
                T_refs.clear()
                ref_bris.clear()
                ref_set_ids.clear()
                logger.info("Reference cleared")

            elif key == ord('s'):
                # 记录当前帧数据到终端
                now = time.time()
                if now - last_record_time < args.record_interval:
                    logger.info("Record too fast, min interval %.1fs", args.record_interval)
                    continue
                last_record_time = now
                if not observed:
                    logger.info("[RECORD] No marker detected")
                    continue
                for mid in marker_ids:
                    if mid in observed:
                        td = observed[mid]
                        t = td['tvec_m2c'].flatten()
                        euler = td['euler_m2c_zyx']
                        reproj = td['reproj_err']
                        bri = _marker_roi_brightness(frame, td)
                        err_str = ""
                        if T_refs.get(mid) is not None:
                            T_cur = aruco_to_matrix(td)
                            _, norm, rot = compute_pose_error_in_frame(T_cur, T_refs[mid])
                            err_str = f"  err: trans={norm:.2f} mm  rot={rot:.2f} deg"
                        bri_str = f"  bri={bri:.0f}" if bri is not None else "  bri=None"
                        bri_ref_str = ""
                        if bri is not None and mid in ref_bris:
                            bri_ref_str = f"  bri_d={bri - ref_bris[mid]:+.0f}"
                        logger.info(
                            "[RECORD] ID%d  [M->C] t=(%.2f, %.2f, %.2f) mm  "
                            "Euler=(%.1f, %.1f, %.1f) deg  reproj=%.2fpx%s%s%s",
                            mid, t[0], t[1], t[2], euler[0], euler[1], euler[2],
                            reproj, bri_str, bri_ref_str, err_str,
                        )
                    else:
                        logger.info("[RECORD] ID%d: not detected", mid)

    finally:
        stop_event.set()
        det_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()


if __name__ == '__main__':
    main()
