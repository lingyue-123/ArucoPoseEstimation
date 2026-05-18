from pymodbus.client import ModbusSerialClient
import struct
import time
import sys

# 自定义 Endian 类
class Endian:
    BIG = 'big'
    LITTLE = 'little'

class MotorController:
    """电机控制器类,封装Modbus RTU通信接口"""
    
    def __init__(self, port='/dev/ttysWK3', baudrate=115200, device_id=1):
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
        self.device_id = device_id
        self.connected = False
    
    def connect(self):
        if self.client.connect():
            self.connected = True
            print("连接成功")
            return True
        else:
            self.connected = False
            print("连接失败,请检查串口设备名和权限")
            return False
    
    def disconnect(self):
        if self.connected:
            self.client.close()
            self.connected = False
            print("已断开连接")
    
    def _pack_32bit_int(self, value, byteorder='big', wordorder='little'):
        """
        将32位整数打包为两个16位寄存器的值（已修正字节序+字序逻辑）
        Args:
            value: 32位整数值
            byteorder: 字节序 ('big' 或 'little')
            wordorder: 字序（高低16位顺序）
        Returns:
            list: [寄存器1, 寄存器2]
        """
        # 1. 按指定字节序打包 32位 int → 4字节
        if byteorder == Endian.BIG:
            packed = struct.pack('>i', value)
            unpack_fmt = '>H'  # 解包16位也用大端
        else:
            packed = struct.pack('<i', value)
            unpack_fmt = '<H'  # 解包16位也用小端

        # 2. 拆分成两个16位寄存器
        word1 = struct.unpack(unpack_fmt, packed[0:2])[0]
        word2 = struct.unpack(unpack_fmt, packed[2:4])[0]

        # 3. 按字序返回（wordorder=little 表示低字在前，最常用）
        if wordorder == Endian.LITTLE:
            return [word2, word1]
        else:
            return [word1, word2]
    
    def set_target_position(self, position, register_addr=5):
        if not self.connected:
            print("未连接,请先调用connect()")
            return False
        
        # 绝大多数伺服/步进驱动：字节大端，字小端（低字在前）
        payload = self._pack_32bit_int(position, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        
        try:
            response = self.client.write_registers(register_addr, payload, device_id=self.device_id)
            if response.isError():
                print(f"设置位置失败: {response}")
                return False
            else:
                print(f"设置位置成功: {position}")
                return True
        except Exception as e:
            print(f"设置位置异常: {e}")
            return False

    def set_move_speed(self, speed, register_addr=7):
        if not self.connected:
            print("未连接,请先调用connect()")
            return False
        
        payload = self._pack_32bit_int(speed, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        
        try:
            response = self.client.write_registers(register_addr, payload, device_id=self.device_id)
            if response.isError():
                print(f"设置运行速度失败: {response}")
                return False
            else:
                print(f"设置运行速度成功: {speed}")
                return True
        except Exception as e:
            print(f"设置运行速度异常: {e}")
            return False

    def set_return_speed(self, return_speed, register_addr=9):
        if not self.connected:
            print("未连接,请先调用connect()")
            return False
        
        payload = self._pack_32bit_int(return_speed, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        
        try:
            response = self.client.write_registers(register_addr, payload, device_id=self.device_id)
            if response.isError():
                print(f"设置回归速度失败: {response}")
                return False
            else:
                print(f"设置回归速度成功: {return_speed}")
                return True
        except Exception as e:
            print(f"设置回归速度异常: {e}")
            return False

    def pause_motor(self, register_addr=33):
        if not self.connected:
            print("未连接,请先调用connect()")
            return False
        
        try:
            response = self.client.write_register(register_addr, 0, device_id=self.device_id)
            if response.isError():
                print(f"暂停电机失败: {response}")
                return False
            else:
                print("电机已暂停")
                return True
        except Exception as e:
            print(f"暂停电机异常: {e}")
            return False

    def homing_motor(self, register_addr=34):
        if not self.connected:
            print("未连接,请先调用connect()")
            return False
        
        try:
            response = self.client.write_register(register_addr, 1, device_id=self.device_id)
            if response.isError():
                print(f"电机回原位失败: {response}")
                return False
            else:
                print("电机开始回原位")
                return True
        except Exception as e:
            print(f"电机回原位异常: {e}")
            return False


# ==================== 使用示例 ====================
if __name__ == "__main__":
    motor = MotorController(port='/dev/ttysWK3', baudrate=115200, device_id=1)
    if len(sys.argv)>0:
        if motor.connect():
            # motor.set_move_speed(14000) # 运行速度
            # time.sleep(5)
            if sys.argv[1] == "home":
                motor.homing_motor()
            # time.sleep(5)
            if sys.argv[1] == "move":
                motor.set_target_position(14000) # 目标位置
                time.sleep(1)
            # motor.homing_motor() 
            motor.disconnect()
