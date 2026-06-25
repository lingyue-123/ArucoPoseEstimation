#!/usr/bin/env python3
"""
夹爪 Modbus RTU 寄存器采集脚本
用法: python3 sniff_gripper.py [采集次数] [csv文件]
      python3 sniff_gripper.py                 # 默认50ms周期, 无限循环, 自动生成csv/xlsx/png
      python3 sniff_gripper.py 200             # 50ms周期, 采集200轮后退出
      python3 sniff_gripper.py 200 data.csv    # 50ms周期, 采集200轮并写入data.csv
"""

import argparse
import csv
import logging
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from gripper import GripperController

BEIJING_TZ = timezone(timedelta(hours=8))

# ── 默认采集寄存器定义 ──
REGISTER_DEFS = [
    {"addr": 0x0202, "key": "0x0202", "name": "位置反馈", "width": 16, "signed": False, "unit": "", "plot": True},
    {"addr": 0x0203, "key": "0x0203", "name": "速度反馈", "width": 16, "signed": False, "unit": "", "plot": True},
    {"addr": 0x0204, "key": "0x0204", "name": "电流反馈", "width": 16, "signed": False, "unit": "", "plot": True},
    {"addr": 0x0600, "key": "0x0600", "name": "母线电压", "width": 16, "signed": False, "unit": "", "plot": True},
    {"addr": 0x1300, "key": "0x1300", "name": "控制器1当前电流", "width": 16, "signed": False, "unit": "", "plot": True},
    {"addr": 0x1302, "key": "0x1302_0x1303", "name": "控制器1当前位置", "width": 32, "signed": True, "unit": "", "plot": True},
    {"addr": 0x1308, "key": "0x1308", "name": "夹爪编码器读数", "width": 16, "signed": False, "unit": "", "plot": True},
]

# 状态码对照表，保留给兼容寄存器解析使用
STATUS_INIT = {0: "未初始化", 1: "已初始化", 2: "初始化中"}
STATUS_GRIP = {0: "运动中", 1: "到位", 2: "夹住物体", 3: "物体掉落"}
STATUS_DIR = {0: "张开", 1: "闭合"}
STATUS_BAUD = {0: "115200", 1: "57600", 2: "38400", 3: "19200", 4: "9600", 5: "4800"}
STATUS_STOP = {0: "1bit", 1: "2bit"}
STATUS_PARITY = {0: "无校验", 1: "奇校验", 2: "偶校验"}


def ts():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def beijing_time():
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_int_auto_base(text):
    return int(str(text), 0)


def default_prefix():
    return f"gripper_registers_{datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')}"


def setup_hex_logging(hex_log, console_hex=True):
    log_fmt = logging.Formatter('%(asctime)s.%(msecs)03d  %(message)s', datefmt='%H:%M:%S')

    handlers = []
    if console_hex:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(log_fmt)
        handlers.append(console_handler)

    file_handler = logging.FileHandler(hex_log, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_fmt)
    handlers.append(file_handler)

    for logger_name in ("pymodbus.transaction", "pymodbus.framer"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = False


def col_prefix(reg):
    return f"{reg['key']}_{reg['name']}"


def fmt_val(addr: int, val):
    """根据寄存器地址返回解释后的值字符串"""
    if val is None:
        return "读取失败"
    s = f"{val}"
    if addr == 0x0200:
        s += f" ({STATUS_INIT.get(val, '?')})"
    elif addr == 0x0201:
        s += f" ({STATUS_GRIP.get(val, '?')})"
    elif addr == 0x0301:
        s += f" ({STATUS_DIR.get(val, '?')})"
    elif addr == 0x0303:
        s += f" ({STATUS_BAUD.get(val, '?')})"
    elif addr == 0x0304:
        s += f" ({STATUS_STOP.get(val, '?')})"
    elif addr == 0x0305:
        s += f" ({STATUS_PARITY.get(val, '?')})"
    elif addr == 0x0315:
        s += f" ({val / 10.0}℃)"
    return s


def combine_32bit(registers, byteorder='big', wordorder='little'):
    if wordorder == 'little':
        combined = (registers[1] << 16) | registers[0]
    else:
        combined = (registers[0] << 16) | registers[1]

    if byteorder == 'big':
        packed = struct.pack('>I', combined)
        return struct.unpack('>i', packed)[0]

    packed = struct.pack('<I', combined)
    return struct.unpack('<i', packed)[0]


def read_register_value(gripper, reg):
    """读取寄存器，返回(raw, value)。value第一版默认等于raw。"""
    width = reg.get("width", 16)
    scale = reg.get("scale", 1)
    offset = reg.get("offset", 0)

    if width == 32:
        registers = gripper.read_register(reg["addr"], count=2)
        if registers is None:
            return None, None
        raw = combine_32bit(
            registers,
            byteorder=reg.get("byteorder", "big"),
            wordorder=reg.get("wordorder", "little"),
        )
        return f"{registers[0]},{registers[1]}", raw * scale + offset

    raw = gripper.read_16bit_value(reg["addr"], signed=reg.get("signed", False))
    if raw is None:
        return None, None
    return raw, raw * scale + offset


def parse_registers(registers_text):
    if not registers_text:
        return REGISTER_DEFS

    defs = []
    names = {reg["addr"]: reg for reg in REGISTER_DEFS if reg.get("width") == 16}
    for item in registers_text.split(','):
        addr = parse_int_auto_base(item.strip())
        if addr in names:
            defs.append(dict(names[addr]))
        else:
            defs.append({
                "addr": addr,
                "key": f"0x{addr:04X}",
                "name": f"寄存器0x{addr:04X}",
                "width": 16,
                "signed": False,
                "unit": "",
                "plot": True,
            })
    return defs


def build_fieldnames(registers):
    fieldnames = ["round", "北京时间", "timestamp_iso", "elapsed_ms"]
    for reg in registers:
        prefix = col_prefix(reg)
        fieldnames.append(f"{prefix}_raw")
        fieldnames.append(f"{prefix}_value")
    return fieldnames


def write_control(gripper, args, events, start_time):
    now_elapsed = lambda: (time.monotonic() - start_time) * 1000

    if args.position is not None:
        success = gripper.write_register(args.position_register, args.position)
        event = {
            "北京时间": beijing_time(),
            "elapsed_ms": f"{now_elapsed():.3f}",
            "action": "position",
            "register": f"0x{args.position_register:04X}",
            "value": args.position,
            "success": success,
        }
        events.append(event)
        print(f"{ts()}  写位置控制 0x{args.position_register:04X} = {args.position} {'成功' if success else '失败'}")

    if args.force is not None:
        success = gripper.write_register(args.force_register, args.force)
        event = {
            "北京时间": beijing_time(),
            "elapsed_ms": f"{now_elapsed():.3f}",
            "action": "force",
            "register": f"0x{args.force_register:04X}",
            "value": args.force,
            "success": success,
        }
        events.append(event)
        print(f"{ts()}  写力控制   0x{args.force_register:04X} = {args.force} {'成功' if success else '失败'}")


def export_xlsx(rows, registers, events, meta, xlsx_path):
    try:
        from openpyxl import Workbook
        from openpyxl.chart import LineChart, Reference
    except ImportError:
        print("未安装 openpyxl，跳过Excel导出。可执行: pip install openpyxl")
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    fieldnames = build_fieldnames(registers)
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(name, "") for name in fieldnames])
    ws.freeze_panes = "A2"

    meta_ws = wb.create_sheet("meta")
    meta_ws.append(["key", "value"])
    for key, value in meta.items():
        meta_ws.append([key, value])
    meta_ws.append(["registers", ", ".join(f"{reg['key']} {reg['name']}" for reg in registers)])

    events_ws = wb.create_sheet("events")
    event_fields = ["北京时间", "elapsed_ms", "action", "register", "value", "success"]
    events_ws.append(event_fields)
    for event in events:
        events_ws.append([event.get(name, "") for name in event_fields])

    charts_ws = wb.create_sheet("charts")
    chart_row = 1
    if rows:
        for reg in registers:
            prefix = col_prefix(reg)
            value_col_name = f"{prefix}_value"
            if value_col_name not in fieldnames:
                continue
            col_idx = fieldnames.index(value_col_name) + 1
            chart = LineChart()
            chart.title = f"{reg['name']} ({reg['key']})"
            chart.y_axis.title = reg.get("unit", "value") or "value"
            chart.x_axis.title = "北京时间"
            data = Reference(ws, min_col=col_idx, min_row=1, max_row=len(rows) + 1)
            cats = Reference(ws, min_col=2, min_row=2, max_row=len(rows) + 1)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.height = 8
            chart.width = 22
            charts_ws.add_chart(chart, f"A{chart_row}")
            chart_row += 16

    wb.save(xlsx_path)
    print(f"Excel文件: {xlsx_path}")
    return True


def export_plot(rows, registers, plot_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过曲线绘制。可执行: pip install matplotlib")
        return False

    plot_regs = [reg for reg in registers if reg.get("plot", True)]
    if not rows or not plot_regs:
        print("没有可绘制的数据，跳过曲线绘制")
        return False

    x = [row.get("elapsed_ms", "") for row in rows]
    fig, axes = plt.subplots(len(plot_regs), 1, figsize=(12, max(3, 2.6 * len(plot_regs))), sharex=True)
    if len(plot_regs) == 1:
        axes = [axes]

    for ax, reg in zip(axes, plot_regs):
        value_key = f"{col_prefix(reg)}_value"
        y = []
        x_valid = []
        for x_value, row in zip(x, rows):
            value = row.get(value_key, "")
            if value == "" or value is None:
                continue
            x_valid.append(float(x_value))
            y.append(float(value))
        ax.plot(x_valid, y, marker='.', linewidth=1)
        ax.set_title(f"{reg['name']} ({reg['key']})")
        ax.set_ylabel(reg.get("unit", "") or "value")
        ax.grid(True)

    axes[-1].set_xlabel("elapsed_ms")
    start_bt = rows[0].get("北京时间", "")
    end_bt = rows[-1].get("北京时间", "")
    fig.suptitle(f"夹爪寄存器曲线 北京时间 {start_bt} ~ {end_bt}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"曲线图片: {plot_path}")
    return True


def parse_args():
    prefix = default_prefix()
    parser = argparse.ArgumentParser(description="夹爪寄存器50ms采集、控制写入、CSV/Excel/曲线导出")
    parser.add_argument("legacy_samples", nargs="?", type=int, help="兼容旧用法：采集次数，0表示无限")
    parser.add_argument("legacy_csv", nargs="?", help="兼容旧用法：CSV文件")
    parser.add_argument("--samples", type=int, default=None, help="采集次数，0表示无限")
    parser.add_argument("--interval-ms", type=int, default=50, help="采集周期ms，默认50")
    parser.add_argument("--csv", default=None, help="CSV输出路径")
    parser.add_argument("--xlsx", nargs="?", const="auto", default="auto", help="Excel输出路径；不带值时自动命名")
    parser.add_argument("--plot", nargs="?", const="auto", default="auto", help="曲线PNG输出路径；不带值时自动命名")
    parser.add_argument("--no-xlsx", action="store_true", help="不生成Excel")
    parser.add_argument("--no-plot", action="store_true", help="不生成曲线PNG")
    parser.add_argument("--fsync", action="store_true", help="每轮CSV写入后执行fsync，可靠性更高但可能影响50ms周期")
    parser.add_argument("--port", default="/dev/ttysWK3", help="串口设备")
    parser.add_argument("--baudrate", type=int, default=115200, help="波特率")
    parser.add_argument("--slave-id", type=int, default=4, help="Modbus从站ID")
    parser.add_argument("--registers", help="覆盖默认采集寄存器，逗号分隔，如 0x0202,0x0203")
    parser.add_argument("--position", type=int, help="位置控制目标值，默认写入0x1003")
    parser.add_argument("--position-register", type=parse_int_auto_base, default=0x1003, help="位置控制寄存器，默认0x1003")
    parser.add_argument("--force", type=int, help="力控制目标值，默认写入0x1001")
    parser.add_argument("--force-register", type=parse_int_auto_base, default=0x1001, help="力控制寄存器，默认0x1001")
    parser.add_argument("--write-timing", choices=("before", "each", "none"), default="before", help="控制写入时机，默认before")
    parser.add_argument("--hex-log", default=f"{prefix}.hex.log", help="pymodbus交互报文日志路径")
    parser.add_argument("--no-console-hex", action="store_true", help="不在控制台输出pymodbus hex日志")
    args = parser.parse_args()

    if args.samples is None:
        args.samples = args.legacy_samples if args.legacy_samples is not None else 0
    if args.csv is None:
        args.csv = args.legacy_csv if args.legacy_csv else f"{prefix}.csv"
    if args.xlsx == "auto":
        args.xlsx = f"{prefix}.xlsx"
    if args.plot == "auto":
        args.plot = f"{prefix}.png"

    if args.position is not None and not 0 <= args.position <= 1000:
        parser.error("--position 范围应为 0-1000")
    if args.force is not None and not 0 <= args.force <= 1000:
        parser.error("--force 范围应为 0-1000，请按设备手册确认实际力值范围")
    if args.interval_ms <= 0:
        parser.error("--interval-ms 必须大于0")

    return args


def main():
    args = parse_args()
    registers = parse_registers(args.registers)
    interval_s = args.interval_ms / 1000.0

    setup_hex_logging(args.hex_log, console_hex=not args.no_console_hex)

    print(f"=== 夹爪寄存器采集 周期={args.interval_ms}ms 次数={'无限' if args.samples==0 else args.samples} "
          f"串口={args.port} 从站ID={args.slave_id} csv={args.csv} ===")
    print("采集寄存器: " + ", ".join(f"{reg['key']}({reg['name']})" for reg in registers))

    gripper = GripperController(port=args.port, baudrate=args.baudrate, device_id=args.slave_id)
    if not gripper.connect():
        print("连接夹爪失败")
        return

    fieldnames = build_fieldnames(registers)
    rows = []
    events = []
    round_num = 0
    start_time = time.monotonic()
    meta = {
        "start_beijing_time": beijing_time(),
        "port": args.port,
        "baudrate": args.baudrate,
        "slave_id": args.slave_id,
        "interval_ms": args.interval_ms,
        "samples": args.samples,
        "csv": args.csv,
        "xlsx": "" if args.no_xlsx else args.xlsx,
        "plot": "" if args.no_plot else args.plot,
        "hex_log": args.hex_log,
        "position_register": f"0x{args.position_register:04X}",
        "position": "" if args.position is None else args.position,
        "force_register": f"0x{args.force_register:04X}",
        "force": "" if args.force is None else args.force,
        "write_timing": args.write_timing,
    }

    try:
        if args.write_timing == "before":
            write_control(gripper, args, events, start_time)

        with open(args.csv, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while args.samples == 0 or round_num < args.samples:
                loop_start = time.monotonic()
                round_num += 1

                if args.write_timing == "each":
                    write_control(gripper, args, events, start_time)

                row = {
                    "round": round_num,
                    "北京时间": beijing_time(),
                    "timestamp_iso": datetime.now(BEIJING_TZ).isoformat(timespec="milliseconds"),
                    "elapsed_ms": f"{(loop_start - start_time) * 1000:.3f}",
                }
                for reg in registers:
                    raw, value = read_register_value(gripper, reg)
                    prefix = col_prefix(reg)
                    row[f"{prefix}_raw"] = "" if raw is None else raw
                    row[f"{prefix}_value"] = "" if value is None else value
                    print(f"{ts()}  {reg['key']:<11} {reg['name']:<18} = {fmt_val(reg['addr'], value)}")

                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                if args.fsync:
                    os.fsync(csv_file.fileno())

                print(f"--- round {round_num} end ---")
                if args.samples == 0 or round_num < args.samples:
                    sleep_s = interval_s - (time.monotonic() - loop_start)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        print(f"警告: 本轮采集耗时超过{args.interval_ms}ms ({(time.monotonic() - loop_start) * 1000:.1f}ms)")

    except KeyboardInterrupt:
        print(f"\n中断，共 {round_num} 轮")
    finally:
        gripper.disconnect()
        meta["end_beijing_time"] = beijing_time()
        meta["rounds"] = round_num

        if not args.no_xlsx and args.xlsx:
            export_xlsx(rows, registers, events, meta, args.xlsx)
        if not args.no_plot and args.plot:
            export_plot(rows, registers, args.plot)

        print(f"csv文件: {args.csv}")
        print(f"hex日志: {args.hex_log}")


if __name__ == "__main__":
    main()
