#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time
import threading
import argparse
import argcomplete
import asyncio

class GripperController:
    """基于 Modbus RTU 的夹爪控制器（纯 serial + CRC，无任何第三方库依赖）"""

    # 寄存器地址映射
    REG_INIT         = 0x0100
    REG_FORCE        = 0x0101
    REG_POSITION     = 0x0103
    REG_SPEED        = 0x0104
    REG_INIT_STATUS  = 0x0200
    REG_GRIP_STATUS  = 0x0201
    REG_POS_FEEDBACK = 0x0202

    INIT_STATUS_MSG = {0: "未初始化", 1: "初始化成功", 2: "初始化中"}
    GRIP_STATUS_MSG = {0: "运动中", 1: "到达位置", 2: "夹住物体", 3: "物体掉落"}

    def __init__(self, port='/dev/ttysWK3', baudrate=115200, slave_id=4, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.timeout = timeout
        self.ser = None
        self.connected = False
        self._serial_lock = threading.Lock()
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        self._latest_grip_status = None

    @property
    def latest_grip_status(self):
        """心跳线程最近一次读取的夹爪状态（0=运动中, 1=到达位置, 2=夹住物体, 3=物体掉落），None=尚未读取"""
        return self._latest_grip_status

    @staticmethod
    def _crc16(data: bytes) -> int:
        """Modbus RTU CRC-16"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _ensure_connected(self):
        """自动连接（如尚未连接则打开串口但不启动心跳）"""
        if self.connected and self.ser is not None and self.ser.is_open:
            return True
        return self.connect()

    def connect(self):
        """打开串口并启动心跳线程"""
        try:
            if self.ser is not None and self.ser.is_open:
                self.connected = True
            else:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                self.connected = True
            self._start_heartbeat()
            print(" 夹爪连接成功")
            return True
        except Exception as e:
            print(f" 串口打开失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """停止心跳并关闭串口"""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print("夹爪已断开")

    def _start_heartbeat(self):
        """启动心跳线程（如尚未运行）"""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """心跳循环：每 5ms 读取一次 0x0201 寄存器（夹持状态）"""
        while not self._heartbeat_stop.is_set():
            try:
                with self._serial_lock:
                    value = self._read_register_no_lock(self.REG_GRIP_STATUS)
                    if value is not None:
                        self._latest_grip_status = value
            except Exception:
                pass
            time.sleep(0.005)

    def _send_frame(self, req_data: bytes) -> bytes:
        """发送请求并接收响应（自动添加CRC，并验证响应CRC）——线程安全"""
        self._ensure_connected()
        if not self.ser:
            raise RuntimeError("串口未连接")
        with self._serial_lock:
            crc = self._crc16(req_data)
            frame = req_data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            time.sleep(0.05)
            resp = self.ser.read(8)
            if len(resp) == 0:
                raise TimeoutError("无响应")
            if len(resp) >= 2:
                recv_crc = resp[-2] | (resp[-1] << 8)
                calc_crc = self._crc16(resp[:-2])
                if recv_crc != calc_crc:
                    raise ValueError(f"CRC校验失败")
            return resp

    # ---------- 写操作（功能码06）----------
    def _write_register(self, reg_addr, value):
        """底层写入，返回成功/失败，并打印详细日志"""
        req = bytes([
            self.slave_id, 0x06,
            (reg_addr >> 8) & 0xFF, reg_addr & 0xFF,
            (value >> 8) & 0xFF, value & 0xFF
        ])
        try:
            resp = self._send_frame(req)
            if resp[:6] == req[:6]:
                print(f"写入寄存器成功: 地址={hex(reg_addr)} (0x{reg_addr:04X}), 值={value} (0x{value:04X})")
                return True
            else:
                print(f"写入寄存器失败: 地址={hex(reg_addr)}, 值={value}, 响应异常: {resp.hex()}")
                return False
        except Exception as e:
            print(f"写入寄存器异常: 地址={hex(reg_addr)}, 值={value}, 错误={e}")
            return False

    def home(self):
        """回零位（写1）"""
        return self._write_register(self.REG_INIT, 1)

    def recalibrate(self):
        """重新标定（写0xA5）"""
        return self._write_register(self.REG_INIT, 0xA5)

    def set_force(self, percent):
        if not 20 <= percent <= 100:
            print(f"力值 {percent} 超出范围 20-100")
            return False
        return self._write_register(self.REG_FORCE, percent)

    def set_position(self, permille):
        return self._write_register(self.REG_POSITION, permille)
    
    async def async_set_position(self, permille):
        return await asyncio.get_event_loop().run_in_executor(None, self.set_position, permille)

    def set_speed(self, percent):
        if not 1 <= percent <= 100:
            print(f"速度 {percent} 超出范围 1-100")
            return False
        return self._write_register(self.REG_SPEED, percent)

    # ---------- 读操作（功能码03）----------
    def _read_register_no_lock(self, reg_addr):
        """读取单个保持寄存器（不加锁，由调用者保证串行访问）"""
        req = bytes([
            self.slave_id, 0x03,
            (reg_addr >> 8) & 0xFF, reg_addr & 0xFF,
            0x00, 0x01
        ])
        resp = self._send_frame_internal(req)
        if len(resp) >= 5 and resp[1] == 0x03:
            value = (resp[3] << 8) | resp[4]
            return value
        else:
            print(f"读取响应格式错误: {resp.hex()}")
            return None

    def _send_frame_internal(self, req_data: bytes) -> bytes:
        """内部发送（不自动连接，不加锁，由调用者处理）"""
        if not self.ser:
            raise RuntimeError("串口未连接")
        crc = self._crc16(req_data)
        frame = req_data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        time.sleep(0.05)
        resp = self.ser.read(8)
        if len(resp) == 0:
            raise TimeoutError("无响应")
        if len(resp) >= 2:
            recv_crc = resp[-2] | (resp[-1] << 8)
            calc_crc = self._crc16(resp[:-2])
            if recv_crc != calc_crc:
                raise ValueError(f"CRC校验失败")
        return resp

    def _read_register(self, reg_addr):
        """读取单个保持寄存器（公开加锁接口）"""
        with self._serial_lock:
            return self._read_register_no_lock(reg_addr)

    def get_initialization_status(self):
        return self._read_register(self.REG_INIT_STATUS)

    def get_grip_status(self):
        return self._read_register(self.REG_GRIP_STATUS)

    def get_current_position(self):
        return self._read_register(self.REG_POS_FEEDBACK)

    def get_set_force(self):
        return self._read_register(self.REG_FORCE)

    def get_set_position(self):
        return self._read_register(self.REG_POSITION)

    def get_set_speed(self):
        return self._read_register(self.REG_SPEED)

    # ---------- 辅助方法 ----------
    def get_initialization_status_str(self):
        s = self.get_initialization_status()
        return self.INIT_STATUS_MSG.get(s, f"未知({s})") if s is not None else "读取失败"

    def get_grip_status_str(self):
        s = self.get_grip_status()
        return self.GRIP_STATUS_MSG.get(s, f"未知({s})") if s is not None else "读取失败"

    def print_all_status(self):
        print("\n========== 夹爪状态 ==========")
        print(f"初始化状态: {self.get_initialization_status()} - {self.get_initialization_status_str()}")
        print(f"夹持状态:   {self.get_grip_status()} - {self.get_grip_status_str()}")
        pos = self.get_current_position()
        print(f"实时位置:   {pos}‰" if pos is not None else "实时位置: 读取失败")
        print(f"设定力值:   {self.get_set_force()}%")
        print(f"设定位置:   {self.get_set_position()}‰")
        print(f"设定速度:   {self.get_set_speed()}%")
        print("===============================\n")

    def open(self, speed: int = 20, force: int = 50) -> bool:
        self.set_speed(speed)
        self.set_force(force)
        return self.set_position(0)

    def open_cover(self, speed: int = 20, force: int = 50) -> bool:
        self.set_speed(speed)
        self.set_force(force)
        return self.set_position(32)

    def close(self, speed: int = 20, force: int = 50) -> bool:
        self.set_speed(speed)
        self.set_force(force)
        return self.set_position(45)


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["home", "open", "open_cover", "close", "status", "calib"],
                        help="要执行的操作")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    gripper = GripperController(port='/dev/ttysWK3', baudrate=115200, slave_id=4)
    if gripper.connect():
        if args.command == "home":
            gripper.home()

        if args.command == "open":
            gripper.open()

        if args.command == "open_cover":
            gripper.open_cover()

        if args.command == "close":
            gripper.close()

        if args.command == "status":
            for _ in range(10):
                print(f"心跳状态: {gripper.latest_grip_status}")
                time.sleep(0.01)
            gripper.print_all_status()

        if args.command == "calib":
            gripper.recalibrate()

    gripper.disconnect()
