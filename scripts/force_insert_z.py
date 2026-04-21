#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沿工具坐标系 Z 轴力控移动 5cm —— 最小可运行脚本

与 force_insert_smoke.py 的区别见文件末尾注释。

运行：
    /home/nvidia/data/conda/envs/transfuser/bin/python scripts/force_insert_z.py

前提：
    [ ] 充电枪 TCP 已在 APP 中配置
    [ ] 力传感器已启用、负载已配置
    [ ] Z+ 方向已通过 smoke 的 Step 2 确认是插入方向
    [ ] 运动路径无障碍，急停在手边

注意：jkrc linear_move 的 INCR 模式（move_mode=1）相对哪个坐标系，
SDK 文档没有明确说明。为保证沿工具 Z 轴移动的可靠性，本脚本采用
读当前 TCP → offset_pose_along_tool_axis 计算绝对目标 → ABS 模式
（move_mode=0）的方式运动。
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robovision.geometry.transforms import offset_pose_along_tool_axis
from robovision.robot.force_control import (
    AXIS_FX,
    AXIS_FY,
    AXIS_FZ,
    AXIS_MX,
    AXIS_MY,
    AXIS_MZ,
    FT_FRAME_TOOL,
    apply_ft_ctrl_config,
    prepare_force_control,
)

# ── 配置 ──────────────────────────────────────────────────────────────────
ROBOT_IP        = "192.168.1.106"
SDK_PATH        = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "third_party", "jaka-python-sdk"
))

DZ_MM           = 50.0    # 沿工具 Z 轴移动距离 (mm)
SPEED_MMPS      = 5.0     # 运动速度 (mm/s)
TARGET_FORCE_N  = 2.0     # 保留参数兼容；默认不在本脚本中启用 Fz 恒力
DAMPING         = 80.0    # 阻尼系数（ft_user）
STEP_MM         = 0.5     # 单步前进距离 (mm)
FORCE_THRESHOLD_N = 10.0  # 碰底停止阈值 (N)
LOG_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
TORQUE_SENSOR_READ_TYPE = 3
# ─────────────────────────────────────────────────────────────────────────


def _ensure_sdk(path):
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if path not in ld:
        os.environ["LD_LIBRARY_PATH"] = path + ":" + ld
        os.execv(sys.executable, [sys.executable] + sys.argv)
    if path not in sys.path:
        sys.path.insert(0, path)


def _build_parser():
    parser = argparse.ArgumentParser(description="沿工具坐标系 Z 轴小步推进，达到力阈值后停止")
    parser.add_argument("--dz-mm", type=float, default=DZ_MM, help="沿工具 Z 轴移动距离，单位 mm")
    parser.add_argument("--speed-mmps", type=float, default=SPEED_MMPS, help="直线运动速度，单位 mm/s")
    parser.add_argument("--step-mm", type=float, default=STEP_MM, help="单步推进距离，单位 mm")
    parser.add_argument("--force-threshold-n", type=float, default=FORCE_THRESHOLD_N, help="Z 轴碰底停止阈值，单位 N")
    parser.add_argument(
        "--compliance",
        choices=("on", "off"),
        default="off",
        help="是否初始化力控；注意本脚本默认不启用 Fz 恒力，只保留位置步进 + 力阈值停",
    )
    parser.add_argument("--target-force-n", type=float, default=TARGET_FORCE_N, help="兼容保留参数；当前脚本默认不使用 Fz 恒力")
    parser.add_argument("--damping", type=float, default=DAMPING, help="柔顺阻尼系数 ft_user")
    parser.add_argument("--log-file", default="", help="CSV 日志输出路径；默认自动生成到 logs/ 下")
    return parser


LOG_FIELDS = [
    "timestamp_s",
    "step_index",
    "moved_mm",
    "current_step_mm",
    "fx_n",
    "fy_n",
    "fz_n",
    "tx_nm",
    "ty_nm",
    "tz_nm",
    "tcp_x_mm",
    "tcp_y_mm",
    "tcp_z_mm",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
    "threshold_hit",
    "move_error",
]


def build_default_log_path():
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"force_insert_z_{time.strftime('%Y%m%d_%H%M%S')}.csv")


def append_log_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    needs_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def read_force_vector(robot):
    try:
        ret = robot.get_torque_sensor_data(TORQUE_SENSOR_READ_TYPE)
    except TypeError:
        ret = robot.get_torque_sensor_data()
    code = ret[0] if isinstance(ret, tuple) else ret
    if code != 0 or len(ret) < 2:
        raise RuntimeError(f"读取力传感器失败: {ret}")
    payload = ret[1]
    values = None
    if isinstance(payload, (list, tuple)) and len(payload) >= 6 and all(isinstance(v, (int, float)) for v in payload[:6]):
        values = payload
    elif (
        isinstance(payload, (list, tuple))
        and len(payload) >= 3
        and isinstance(payload[2], (list, tuple))
        and len(payload[2]) >= 6
    ):
        values = payload[2]
    if values is None:
        raise RuntimeError(f"无法解析力传感器返回格式: {ret}")
    return {
        "fx_n": float(values[0]),
        "fy_n": float(values[1]),
        "fz_n": float(values[2]),
        "tx_nm": float(values[3]),
        "ty_nm": float(values[4]),
        "tz_nm": float(values[5]),
    }


def move_until_force_threshold(robot, total_distance_mm, step_mm, speed_mmps, force_threshold_n, log_file=None):
    moved_mm = 0.0
    step_mm = abs(float(step_mm))
    total_distance_mm = float(total_distance_mm)
    direction = 1.0 if total_distance_mm >= 0 else -1.0
    max_travel_mm = abs(total_distance_mm)
    last_fz_n = 0.0
    step_index = 0

    while moved_mm < max_travel_mm:
        step_index += 1
        current_step_mm = min(step_mm, max_travel_mm - moved_mm)
        ret = robot.get_tcp_position()
        code = ret[0] if isinstance(ret, tuple) else ret
        if code != 0:
            raise RuntimeError(f"读取 TCP 失败: {ret}")
        tcp_raw = ret[1]
        tcp_deg = [
            tcp_raw[0], tcp_raw[1], tcp_raw[2],
            float(np.degrees(tcp_raw[3])),
            float(np.degrees(tcp_raw[4])),
            float(np.degrees(tcp_raw[5])),
        ]
        target_deg = offset_pose_along_tool_axis(tcp_deg, direction * current_step_mm, axis='z')
        target_rad = [
            target_deg[0], target_deg[1], target_deg[2],
            float(np.radians(target_deg[3])),
            float(np.radians(target_deg[4])),
            float(np.radians(target_deg[5])),
        ]
        ret = robot.linear_move(target_rad, 0, True, speed_mmps)
        code = ret[0] if isinstance(ret, tuple) else ret
        if code != 0:
            if log_file:
                append_log_row(
                    log_file,
                    {
                        "timestamp_s": f"{time.time():.6f}",
                        "step_index": step_index,
                        "moved_mm": direction * moved_mm,
                        "current_step_mm": direction * current_step_mm,
                        "move_error": str(ret),
                    },
                )
            raise RuntimeError(f"单步 linear_move 失败: {ret}")
        moved_mm += current_step_mm
        force = read_force_vector(robot)
        last_fz_n = force["fz_n"]
        current_pose = robot.get_tcp_position()
        pose_code = current_pose[0] if isinstance(current_pose, tuple) else current_pose
        if pose_code != 0:
            raise RuntimeError(f"读取运动后 TCP 失败: {current_pose}")
        tcp_after = current_pose[1]
        threshold_hit = abs(last_fz_n) >= float(force_threshold_n)
        if log_file:
            append_log_row(
                log_file,
                {
                    "timestamp_s": f"{time.time():.6f}",
                    "step_index": step_index,
                    "moved_mm": direction * moved_mm,
                    "current_step_mm": direction * current_step_mm,
                    **force,
                    "tcp_x_mm": float(tcp_after[0]),
                    "tcp_y_mm": float(tcp_after[1]),
                    "tcp_z_mm": float(tcp_after[2]),
                    "tcp_rx_rad": float(tcp_after[3]),
                    "tcp_ry_rad": float(tcp_after[4]),
                    "tcp_rz_rad": float(tcp_after[5]),
                    "threshold_hit": threshold_hit,
                    "move_error": "",
                },
            )
        print(f"  已前进 {direction * moved_mm:+.2f} mm, 当前 fz={last_fz_n:+.2f} N")
        if threshold_hit:
            return {
                "stopped_on_force": True,
                "moved_mm": moved_mm,
                "last_fz_n": last_fz_n,
            }

    return {
        "stopped_on_force": False,
        "moved_mm": moved_mm,
        "last_fz_n": last_fz_n,
    }


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _ensure_sdk(SDK_PATH)
    import jkrc  # noqa: E402

    def check(ret, label):
        code = ret[0] if isinstance(ret, tuple) else ret
        if code != 0:
            print(f"[FAIL] {label}: {ret}")
            sys.exit(2)
        print(f"[ OK ] {label}")

    robot = jkrc.RC(ROBOT_IP)
    log_file = args.log_file or build_default_log_path()

    try:
        # 1. 连接
        check(robot.login(),        "login")
        check(robot.power_on(),     "power_on")
        check(robot.enable_robot(), "enable_robot")

        if args.compliance == "on":
            # 2. 力控初始化
            check(prepare_force_control(robot, frame=FT_FRAME_TOOL), "prepare_force_control")

            # 3. 当前脚本不启用 Fz 恒力，避免与 Z 向 linear_move 组合时报错。
            #    所有轴保持 off，仅使用力传感器读数做阈值停。
            for axis in [AXIS_FX, AXIS_FY, AXIS_FZ, AXIS_MX, AXIS_MY, AXIS_MZ]:
                check(
                    apply_ft_ctrl_config(robot, axis, 0, args.damping, 0.0, 0.0, 0.0),
                    f"ft_ctrl_config axis={axis} off",
                )

            # 4. 进入力控模式，仅用于读取/对齐传感器链路，不在 Z 上施加恒力目标。
            check(robot.set_ft_ctrl_mode(1), "set_ft_ctrl_mode(1) 开力控链路")
            time.sleep(0.2)
        else:
            print("柔顺模式: OFF")

        # 5. 沿工具 Z 轴小步推进，力超阈值即停
        ret = robot.get_tcp_position()
        if ret[0] != 0:
            print(f"[FAIL] get_tcp_position: {ret}")
            sys.exit(2)
        tcp_raw = ret[1]  # [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
        print(f"当前位姿(raw): {tcp_raw}")
        print(f"日志文件: {log_file}")
        print(f"→ 沿工具 Z 轴最多移动 {args.dz_mm} mm, 单步 {args.step_mm} mm @ {args.speed_mmps} mm/s ...")
        result = move_until_force_threshold(
            robot,
            total_distance_mm=args.dz_mm,
            step_mm=args.step_mm,
            speed_mmps=args.speed_mmps,
            force_threshold_n=args.force_threshold_n,
            log_file=log_file,
        )
        ret = robot.get_tcp_position()
        print(f"到达位姿(raw): {ret[1]}")
        if result["stopped_on_force"]:
            print(f"[ OK ] force threshold stop: fz={result['last_fz_n']:+.2f} N, moved={result['moved_mm']:.2f} mm")
        else:
            print(f"[WARN] 未触发力阈值，已走满 {result['moved_mm']:.2f} mm")
    except Exception as exc:
        print(f"[FAIL] runtime exception: {exc}")
        try:
            abort_ret = robot.motion_abort()
            print(f"[SAFE] motion_abort -> {abort_ret}")
        except Exception as abort_exc:
            print(f"[SAFE-FAIL] motion_abort: {abort_exc}")
        raise
    finally:
        if args.compliance == "on":
            try:
                ret = robot.set_ft_ctrl_mode(0)
                print(f"[SAFE] set_ft_ctrl_mode(0) -> {ret}")
            except Exception as exc:
                print(f"[SAFE-FAIL] set_ft_ctrl_mode(0): {exc}")
        try:
            ret = robot.logout()
            print(f"[SAFE] logout -> {ret}")
        except Exception as exc:
            print(f"[SAFE-FAIL] logout: {exc}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# 与 force_insert_smoke.py 的区别
# ═══════════════════════════════════════════════════════════════════════════
#
# 维度              force_insert_smoke.py              force_insert_z.py（本脚本）
# ──────────────── ──────────────────────────────── ──────────────────────────────
# 目的             验证 API / 探明字段顺序 / 确认方向  直接执行力控移动，假设上述已知
# 交互性           每步需按 ENTER 确认              全自动，无交互
# 步骤数           5 个 Step，覆盖完整生命周期        单一流程，连接→力控→移动→断开
# 运动量           Step2: Z+30mm 纯位置验证方向       Z+50mm 力控运动
#                 Step4: Z+5mm @ 2mm/s 静态手推测试
# API 验证         set 后立即 get 读回比对字段顺序     不验证，直接用
# 旋转柔顺         可选 Rx/Ry 柔顺（USE_ROT_COMPLIANCE）  仅 Fz，其余刚性
# 软限位           设置 30N / 5N·m                  未设置（依赖 APP 里的配置）
# 用途定位         首次调试 / 机制不确定时跑           smoke 通过后的日常集成调用
# 运动模式         INCR (move_mode=1)               offset_pose + ABS (move_mode=0)
#                                                  原因: jkrc INCR 模式参考系不明确
