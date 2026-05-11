#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time
import struct
import sys
import argparse
import argcomplete


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

    def connect(self):
        """打开串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.connected = True
            print(" 夹爪连接成功")
            return True
        except Exception as e:
            print(f" 串口打开失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print("夹爪已断开")

    def _send_frame(self, req_data: bytes) -> bytes:
        """发送请求并接收响应（自动添加CRC，并验证响应CRC）"""
        if not self.ser:
            raise RuntimeError("串口未连接")
        # 添加CRC
        crc = self._crc16(req_data)
        frame = req_data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        self.ser.write(frame)
        time.sleep(0.05)  # 等待设备响应
        resp = self.ser.read(8)  # 正常响应8字节
        if len(resp) == 0:
            raise TimeoutError("无响应")
        # 验证CRC（可选）
        if len(resp) >= 2:
            recv_crc = resp[-2] | (resp[-1] << 8)
            calc_crc = self._crc16(resp[:-2])
            if recv_crc != calc_crc:
                raise ValueError(f"CRC校验失败")
        return resp

    # ---------- 写操作（功能码06）----------
    # def _write_register(self, reg_addr, value):
    #     req = bytes([
    #         self.slave_id, 0x06,
    #         (reg_addr >> 8) & 0xFF, reg_addr & 0xFF,
    #         (value >> 8) & 0xFF, value & 0xFF
    #     ])
    #     resp = self._send_frame(req)
    #     # 检查响应是否正确（应原样返回前6字节）
    #     if resp[:6] == req[:6]:
    #         return True
    #     else:
    #         print(f"写入响应异常: {resp.hex()}")
    #         return False
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
        # if not 0 <= permille <= 1000:
        #     print(f"位置 {permille} 超出范围 0-1000")
        #     return False
        return self._write_register(self.REG_POSITION, permille)

    def set_speed(self, percent):
        if not 1 <= percent <= 100:
            print(f"速度 {percent} 超出范围 1-100")
            return False
        return self._write_register(self.REG_SPEED, percent)

    # ---------- 读操作（功能码03）----------
    def _read_register(self, reg_addr):
        """读取单个保持寄存器"""
        req = bytes([
            self.slave_id, 0x03,
            (reg_addr >> 8) & 0xFF, reg_addr & 0xFF,
            0x00, 0x01
        ])
        resp = self._send_frame(req)
        if len(resp) >= 5 and resp[1] == 0x03:
            # 响应格式： slave,03,02, 数据高8,数据低8, CRC0,CRC1
            value = (resp[3] << 8) | resp[4]
            return value
        else:
            print(f"读取响应格式错误: {resp.hex()}")
            return None

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
    argcomplete.autocomplete(parser)   # 启用自动补全
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
            gripper.print_all_status()

        if args.command == "calib":
            gripper.recalibrate()

    gripper.disconnect()

