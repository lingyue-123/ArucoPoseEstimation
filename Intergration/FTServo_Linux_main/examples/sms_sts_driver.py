import serial
import time
import threading
import logging
import sys

# --- 常量定义 ---
SMS_STS_ACC = 41
INST_WRITE = 0x03
DEFAULT_BAUDRATE = 115200

class SMSSTSController:
    """
    舵机控制接口类
    功能：封装串口通信，提供线程安全的运动控制方法
    """
    def __init__(self, port_name, baudrate=DEFAULT_BAUDRATE):
        self.port_name = port_name
        self.baudrate = baudrate
        self.ser = None
        self.is_connected = False
        # 线程锁：防止多个模块同时调用导致串口数据打架
        self.lock = threading.Lock() 
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SMSSTS")

    def connect(self):
        """建立串口连接"""
        try:
            if self.ser and self.ser.is_open:
                return True
                
            self.ser = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            self.is_connected = True
            self.logger.info(f"成功连接到端口: {self.port_name}")
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """关闭串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            self.logger.info("串口已关闭")

    def _calc_checksum(self, data):
        return (~sum(data)) & 0xFF

    def _write_packet(self, id, instruction, params):
        """底层数据包发送 (内部使用)"""
        if not self.is_connected:
            return -1

        # 使用锁确保原子操作
        with self.lock:
            try:
                length = len(params) + 2
                packet = [0xFF, 0xFF, id, length, instruction] + params
                
                checksum = self._calc_checksum(packet[2:])
                packet.append(checksum)
                
                self.ser.write(bytearray(packet))
                return 0
            except Exception as e:
                self.logger.error(f"写入错误: {e}")
                return -1

    def _move_to(self, servo_id, position, speed=2400, acc=50, wait_time=2.0):
        """
        内部通用移动方法
        """
        if not self.is_connected:
            self.logger.warning("未连接，正在尝试重连...")
            if not self.connect():
                return False

        # 参数打包: ACC(1) + Pos(2) + Time(2) + Speed(2)
        params = [
            acc, 
            position & 0xFF, (position >> 8) & 0xFF, 
            0, 0, # Time (0 = Speed control mode)
            speed & 0xFF, (speed >> 8) & 0xFF
        ]
        
        write_params = [SMS_STS_ACC] + params
        
        if self._write_packet(servo_id, INST_WRITE, write_params) == 0:
            self.logger.debug(f"指令已发: ID={servo_id}, Pos={position}")
            # 等待运动完成
            time.sleep(wait_time)
            return True
        return False

    # ==========================================
    # 对外暴露的业务接口 API
    # ==========================================

    def press_trigger(self, servo_id=1):
        """
        业务接口：执行“按下”动作 (移动到 2048)
        通常用于机械臂取枪时的触发或按压
        """
        self.logger.info(f">>> 执行 [按下] 动作 -> 目标位置: 1650")
        # 假设速度2400，等待2秒足够到位
        return self._move_to(servo_id, position=1650, speed=1500, wait_time=2.0)

    def reset_position(self, servo_id=1):
        """
        业务接口：执行“回原点”动作 (移动到 2048)
        通常在取完枪后复位
        """
        self.logger.info(f">>> 执行 [回原点] 动作 -> 目标位置: 2300")
        return self._move_to(servo_id, position=2300, speed=2400, wait_time=2.0)

def main():
    servo = SMSSTSController("/dev/ttysWK1")
    servo.connect()
    if len(sys.argv) > 0:
        if sys.argv[1] == "press":
            servo.press_trigger()
            servo.disconnect()
        elif sys.argv[1] == "home":
            servo.reset_position()
            servo.disconnect()
        else:
            print("choose 'press' or 'home'")
    else:
        print("choose 'press' or 'home'")

if __name__ == "__main__":
    main()
