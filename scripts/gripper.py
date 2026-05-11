import struct
import time
from pymodbus.client import ModbusSerialClient


class GripperController:
    """夹爪控制器类，封装Modbus RTU通信接口，支持03和06功能码"""
    
    def __init__(self, port='/dev/ttysWK3', baudrate=115200, device_id=4):
        """
        初始化夹爪控制器
        
        Args:
            port: 串口设备名
            baudrate: 波特率
            device_id: Modbus从站ID
        """
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
        self.device_id = 4
        self.connected = False
    
    def connect(self):
        """连接夹爪"""
        if self.client.connect():
            self.connected = True
            print("夹爪连接成功")
            return True
        else:
            self.connected = False
            print("夹爪连接失败，请检查串口设备名和权限")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.client.close()
            self.connected = False
            print("夹爪已断开连接")
    
    # ==================== 03功能码：读取保持寄存器 ====================
    
    def read_register(self, register_addr, count=1):
        """
        读取单个或多个保持寄存器（03功能码）
        
        Args:
            register_addr: 寄存器地址
            count: 读取的寄存器数量
            
        Returns:
            list: 寄存器值列表，失败返回None
        """
        if not self.connected:
            print("未连接，请先调用connect()")
            return None
        
        try:
            response = self.client.read_holding_registers(
                register_addr, count, device_id=self.device_id
            )
            
            if response.isError():
                print(f"读取寄存器失败: {response}")
                return None
            else:
                return response.registers
                
        except Exception as e:
            print(f"读取寄存器异常: {e}")
            return None
    
    def read_32bit_value(self, register_addr, byteorder='big', wordorder='little'):
        """
        读取32位数值（占两个寄存器，03功能码）
        
        Args:
            register_addr: 起始寄存器地址
            byteorder: 字节序 ('big' 或 'little')
            wordorder: 字序（高低16位顺序）
            
        Returns:
            int: 32位整数值，失败返回None
        """
        registers = self.read_register(register_addr, count=2)
        if registers is None:
            return None
        
        # 根据字序重组32位数据
        if wordorder == 'little':
            # 低字在前：[低16位, 高16位]
            combined = (registers[1] << 16) | registers[0]
        else:
            # 高字在前：[高16位, 低16位]
            combined = (registers[0] << 16) | registers[1]
        
        # 根据字节序打包和解包为有符号整数
        if byteorder == 'big':
            packed = struct.pack('>I', combined)
        else:
            packed = struct.pack('<I', combined)
        
        # 解包为有符号整数
        return struct.unpack('>i', packed)[0] if byteorder == 'big' else struct.unpack('<i', packed)[0]
    
    def read_16bit_value(self, register_addr, signed=False):
        """
        读取16位数值（03功能码）
        
        Args:
            register_addr: 寄存器地址
            signed: 是否为有符号整数
            
        Returns:
            int: 16位整数值，失败返回None
        """
        registers = self.read_register(register_addr, count=1)
        if registers is None:
            return None
        
        value = registers[0]
        
        if signed and value > 0x7FFF:
            value = value - 0x10000
            
        return value
    
    # ==================== 06功能码：写入单个寄存器 ====================
    
    def write_register(self, register_addr, value):
        """
        写入单个保持寄存器（06功能码）
        
        Args:
            register_addr: 寄存器地址
            value: 写入的值（0-65535）
            
        Returns:
            bool: 是否成功
        """
        if not self.connected:
            print("未连接，请先调用connect()")
            return False
        
        # 检查值范围
        if value < 0 or value > 0xFFFF:
            print(f"值 {value} 超出16位范围 (0-65535)")
            return False
        
        try:
            print("id:"+str(self.device_id))
            response = self.client.write_register(
                register_addr, value, device_id=self.device_id
            )
            
            if response.isError():
                print(f"写入寄存器失败: {response}")
                return False
            else:
                print(f"写入寄存器成功: 地址={register_addr}, 值={value}")
                return True
                
        except Exception as e:
            print(f"写入寄存器异常: {e}")
            return False
    
    def write_32bit_value(self, register_addr, value, byteorder='big', wordorder='little'):
        """
        写入32位数值（占两个寄存器，06功能码需要分两次写入）
        
        Args:
            register_addr: 起始寄存器地址
            value: 32位整数值
            byteorder: 字节序 ('big' 或 'little')
            wordorder: 字序（高低16位顺序）
            
        Returns:
            bool: 是否成功
        """
        if not self.connected:
            print("未连接，请先调用connect()")
            return False
        
        # 打包32位整数为4字节
        if byteorder == 'big':
            packed = struct.pack('>i', value)
        else:
            packed = struct.pack('<i', value)
        
        # 拆分为两个16位值
        word1 = struct.unpack('>H', packed[0:2])[0]
        word2 = struct.unpack('>H', packed[2:4])[0]
        
        # 根据字序排列
        if wordorder == 'little':
            # 低字在前：先写低16位，再写高16位
            low_word = word2
            high_word = word1
        else:
            # 高字在前：先写高16位，再写低16位
            low_word = word1
            high_word = word2
        
        # 分两次写入
        if not self.write_register(register_addr, low_word):
            return False
        if not self.write_register(register_addr + 1, high_word):
            return False
        
        print(f"写入32位值成功: {value}")
        return True
    
    # ==================== 夹爪专用控制方法 ====================
    
    def set_position(self, position):
        """
        设置夹爪位置（0-1000）
        
        Args:
            position: 目标位置，0=完全闭合，1000=完全张开
            
        Returns:
            bool: 是否成功
        """
        if position < 0 or position > 1000:
            print(f"位置 {position} 超出范围 (0-1000)")
            return False
        
        return self.write_register(0x0300, position)
    
    def set_speed(self, speed):
        """
        设置夹爪运动速度（0-100）
        
        Args:
            speed: 速度值，0=最小速度，100=最大速度
            
        Returns:
            bool: 是否成功
        """
        if speed < 0 or speed > 100:
            print(f"速度 {speed} 超出范围 (0-100)")
            return False
        
        return self.write_register(0x0104, speed)
    
    def set_force(self, force):
        """
        设置夹爪夹持力（0-1000）
        
        Args:
            force: 力度值，0=最小力度，1000=最大力度
            
        Returns:
            bool: 是否成功
        """
        if force < 20 or force > 100:
            print(f"力度 {force} 超出范围 (20-100)")
            return False
        
        return self.write_register(0x0101, force)
    
    def set_position(self, position):
        """
        设置夹爪位置（0-1000）
        
        Args:
            force: 力度值，0=最小力度，1000=最大力度
            
        Returns:
            bool: 是否成功
        """
        if position< 0 or position > 1000:
            print(f"位置 {position} 超出范围 (20-100)")
            return False
        
        return self.write_register(0x0103, position)

    
    def set_mode(self, mode):
        """
        设置夹爪工作模式
        
        Args:
            mode: 0=位置模式，1=速度模式，2=力控模式
            
        Returns:
            bool: 是否成功
        """
        if mode not in [0, 1, 2]:
            print(f"模式 {mode} 无效，可选 0=位置模式, 1=速度模式, 2=力控模式")
            return False
        
        return self.write_register(0x0303, mode)
    
    def enable_gripper(self, enable=True):
        """
        使能/禁用夹爪
        
        Args:
            enable: True=使能，False=禁用
            
        Returns:
            bool: 是否成功
        """
        value = 1 if enable else 0
        return self.write_register(0x0100, value)
    
    def start_motion(self):
        """
        启动夹爪运动
        
        Returns:
            bool: 是否成功
        """
        return self.write_register(0x0305, 1)
    
    def stop_motion(self):
        """
        停止夹爪运动
        
        Returns:
            bool: 是否成功
        """
        return self.write_register(0x0305, 0)
    
    # ==================== 读取状态信息（03功能码） ====================
    
    def get_current_position(self):
        """
        获取当前夹爪位置
        
        Returns:
            int: 当前位置（0-1000），失败返回None
        """
        return self.read_16bit_value(0x0202)
    
    def get_current_speed(self):
        """
        获取当前速度
        
        Returns:
            int: 当前速度，失败返回None
        """
        return self.read_16bit_value(0x0311, signed=True)
    
    def get_current_force(self):
        """
        获取当前夹持力
        
        Returns:
            int: 当前力度，失败返回None
        """
        return self.read_16bit_value(0x0312)
    
    def get_status(self):
        """
        获取夹爪状态
        
        Returns:
            dict: 状态信息字典，失败返回None
        """
        status = self.read_16bit_value(0x0200)
        if status is None:
            return None
        
        return {
            'raw_value': status,
            'unenabled': bool(status & 0x0000),
            'is_enabled': bool(status & 0x0001),
        }
    
    def get_error_code(self):
        """
        获取错误代码
        
        Returns:
            int: 错误代码，0表示无错误，失败返回None
        """
        return self.read_16bit_value(0x0314)
    
    def get_temperature(self):
        """
        获取夹爪温度
        
        Returns:
            float: 温度值（摄氏度），失败返回None
        """
        temp_raw = self.read_16bit_value(0x0315)
        if temp_raw is None:
            return None
        return temp_raw / 10.0  # 假设温度分辨率为0.1°C
    
    # ==================== 便捷控制方法 ====================
    
    def grip(self, force=500):
        """
        夹紧操作
        
        Args:
            force: 夹持力（0-1000）
            
        Returns:
            bool: 是否成功
        """
        if not self.set_position(0):  # 0表示完全闭合
            return False
        if not self.set_force(force):
            return False
        return self.start_motion()
    
    def release(self):
        """
        松开操作
        
        Returns:
            bool: 是否成功
        """
        if not self.set_position(1000):  # 1000表示完全张开
            return False
        return self.start_motion()
    
    def home(self):
        """
        回零操作
        
        Returns:
            bool: 是否成功
        """
        return self.write_register(0x0100, 1)
    
    def clear_error(self):
        """
        清除错误
        
        Returns:
            bool: 是否成功
        """
        return self.write_register(0x0307, 1)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建夹爪控制器实例
    gripper = GripperController(port='/dev/ttysWK3', baudrate=115200, device_id=4)
    
    if gripper.connect():
        print("\n=== 夹爪控制示例 ===")
        
        # 1. 使能夹爪
        print("\n1. 使能夹爪")
        gripper.enable_gripper(True)
        time.sleep(0.5)
        
        # 2. 设置参数
        print("\n2. 设置参数")
        gripper.set_speed(50)   # 设置速度
        gripper.set_position(950)
        gripper.set_force(40)   # 设置力度
        

        gripper.enable_gripper(False)
        
        gripper.disconnect()