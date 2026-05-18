import serial
import time
import sys

# --- 常量定义 (基于原 C++ 头文件 SMS_STS.h) ---
SMS_STS_ACC = 41
INST_WRITE = 0x03

# --- 协议实现类 ---
class SMS_STSPython:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.HEADER = [0xFF, 0xFF] 

    def begin(self):
        """初始化串口"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            if self.ser.is_open:
                pass 
            else:
                self.ser.open()
            return True
        except Exception as e:
            print(f"Failed to init sms/sts motor! Error: {e}")
            return False

    def end(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _calc_checksum(self, data):
        """计算校验和"""
        return (~sum(data)) & 0xFF

    def _write_packet(self, id, instruction, params):
        """发送数据包"""
        if not self.ser: return
        
        length = len(params) + 2
        packet = self.HEADER + [id, length, instruction] + params
        
        checksum = self._calc_checksum(packet[2:])
        packet.append(checksum)
        
        self.ser.write(bytearray(packet))

    def WritePosEx(self, id, position, speed, acc=0):
        """写入位置、速度和加速度"""
        if not self.ser: return -1

        # 参数打包: ACC(1) + Pos(2) + Time(2) + Speed(2)
        params = [
            acc,                            # ACC
            position & 0xFF, (position >> 8) & 0xFF, # Position Low/High
            0, 0,                           # Time (0 = Speed control mode)
            speed & 0xFF, (speed >> 8) & 0xFF    # Speed Low/High
        ]

        # 写入地址从 SMS_STS_ACC (41) 开始
        write_params = [SMS_STS_ACC] + params
        
        self._write_packet(id, INST_WRITE, write_params)
        return 0

# --- 主程序逻辑 ---
def main():
    # --- 1. 配置区域 ---
    # 串口路径写死在这里
    PORT_NAME = "/dev/ttysWK1"
    
    # 默认参数 (如果在命令行没传参，就用这个位置)
    DEFAULT_POS = 2048 
    # -----------------

    # --- 2. 解析命令行参数 ---
    target_pos = DEFAULT_POS
    
    # 如果命令行传了参数 (例如: python script.py 1000)
    if len(sys.argv) > 1:
        try:
            target_pos = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid position value '{sys.argv[1]}'. Must be an integer.")
            return

    print(f"Serial: {PORT_NAME}")
    print(f"Target Position: {target_pos}")

    # --- 3. 初始化串口 ---
    sms_sts = SMS_STSPython(PORT_NAME, 115200)

    if not sms_sts.begin():
        return

    try:
        # --- 4. 运动循环 ---
        # 这里演示只运动一次，如果你需要它一直动，可以放在 while True 里
        while True:
            # 发送运动指令
            # ID=1, Pos=target_pos, Speed=2400, Acc=50
            sms_sts.WritePosEx(1, target_pos, 2400, 50)
            print(f"Command sent: Move to {target_pos}")
            
            # 等待运动完成 (根据你的速度和距离，这个时间可能需要调整)
            time.sleep(2) 
            
            # 如果只想动一次就退出，可以在这里 break 或者 return
            # break 
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        sms_sts.end()

if __name__ == "__main__":
    main()