import struct
import time
import threading
import logging
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotInterface")

# --- 寄存器地址定义 ---
REG_STATUS = 1
REG_COORD_SELECT = 1000
REG_TRIGGER_LINEAR = 1001
REG_TRIGGER_JOINT = 1002
REG_SPEED_SELECT = 1004
REG_TARGET_POSE_START = 15000
REG_CURRENT_POSE_START = 12288

class CartesianPose:
    """位姿数据类"""
    def __init__(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def __repr__(self):
        return f"Pose(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, rx={self.rx:.3f}, ry={self.ry:.3f}, rz={self.rz:.3f})"

class RobotInterface:
    def __init__(self, ip="192.168.1.133", port=502, unit_id=1):
        self.ip = ip
        self.port = port
        self.unit_id = unit_id  # Modbus从站ID
        self.client = None
        self.lock = threading.Lock()

    def connect(self):
        """建立 Modbus TCP 连接 """
        self.client = ModbusTcpClient(
            host=self.ip,
            port=self.port,
            timeout=5.0  # 超时配置
        )
        try:
            connection = self.client.connect()
            if not connection:
                logger.error(f"无法连接到机械臂 {self.ip}:{self.port}")
                return False
            logger.info(f"[Init] 成功连接至 {self.ip}:{self.port} (unit_id={self.unit_id})")
            return True
        except Exception as e:
            logger.error(f"连接异常: {e}")
            return False

    def _float_to_regs(self, value):
        """将 float32 转换为两个 16 位寄存器 (大端序)"""
        packed = struct.pack('>f', value)
        high, low = struct.unpack('>HH', packed)
        return [high, low]

    def _regs_to_float(self, high, low):
        """将两个 16 位寄存器转换回 float32"""
        packed = struct.pack('>HH', high, low)
        return struct.unpack('>f', packed)[0]

    def _write_pose_registers(self, pose: CartesianPose):
        """内部方法：写入 12 个寄存器 (6个float)"""
        regs = []
        for val in [pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz]:
            regs.extend(self._float_to_regs(val))
        
        with self.lock:
            try:
                result = self.client.write_registers(
                    address=REG_TARGET_POSE_START,
                    values=regs
                )
                if result.isError():
                    logger.error(f"写入位姿寄存器失败: {result}")
                    return False
                return True
            except ModbusException as e:
                logger.error(f"写入寄存器异常: {e}")
                return False

    def wait_until_idle(self, reg_addr, timeout=10.0):
        """轮询等待目标寄存器归零"""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            with self.lock:
                try:
                    result = self.client.read_holding_registers(
                        address=reg_addr,
                        count=1
                    )
                    if result.isError():
                        logger.warning(f"读取寄存器{reg_addr}失败: {result}")
                        return -1
                    
                    if result.registers[0] == 0:
                        return 0
                except ModbusException as e:
                    logger.error(f"轮询寄存器异常: {e}")
                    return -1
            time.sleep(0.01)
        return -2  # 超时

    def arm_get_status(self):
        """读取状态寄存器 (0:就绪, 1: busy)"""
        with self.lock:
            try:
                result = self.client.read_holding_registers(
                    address=REG_STATUS,
                    count=1
                )
                if result.isError():
                    logger.error(f"读取状态寄存器失败: {result}")
                    return -1
                return result.registers[0]
            except ModbusException as e:
                logger.error(f"读取状态异常: {e}")
                return -1

    def arm_get_coord_sys(self):
        """读取当前坐标系"""
        with self.lock:
            try:
                result = self.client.read_holding_registers(
                    address=REG_COORD_SELECT,
                    count=1
                )
                if result.isError():
                    logger.error("[GetCoord] 读取坐标系失败")
                    return -1
                return result.registers[0]
            except ModbusException as e:
                logger.error(f"读取坐标系异常: {e}")
                return -1

    def arm_set_coord_sys(self, coord):
        """设置坐标系"""
        with self.lock:
            try:
                result = self.client.write_register(
                    address=REG_COORD_SELECT,
                    value=coord
                )
                if result.isError():
                    logger.error(f"设置坐标系失败: {result}")
                    return -1
                return 0
            except ModbusException as e:
                logger.error(f"设置坐标系异常: {e}")
                return -1

    def arm_get_current_pose(self):
        """读取当前机械臂位姿"""
        with self.lock:
            try:
                result = self.client.read_holding_registers(
                    address=REG_CURRENT_POSE_START,
                    count=12  # 6个float，每个占2个寄存器，共12个
                )
                if result.isError():
                    logger.error(f"读取位姿失败: {result}")
                    return None
                
                r = result.registers
                # 解析6个float值（x/y/z/rx/ry/rz）
                return CartesianPose(
                    x=self._regs_to_float(r[0], r[1]),
                    y=self._regs_to_float(r[2], r[3]),
                    z=self._regs_to_float(r[4], r[5]),
                    rx=self._regs_to_float(r[6], r[7]),
                    ry=self._regs_to_float(r[8], r[9]),
                    rz=self._regs_to_float(r[10], r[11])
                )
            except ModbusException as e:
                logger.error(f"读取当前位姿异常: {e}")
                return None
            except IndexError as e:
                logger.error(f"位姿数据解析失败: {e}")
                return None

    def arm_move_linear(self, target: CartesianPose):
        """触发直线运动"""
        if not self._write_pose_registers(target):
            return -1
        
        with self.lock:
            try:
                result = self.client.write_register(
                    address=REG_TRIGGER_LINEAR,
                    value=1
                )
                if result.isError():
                    logger.error("[MoveLinear] 触发失败")
                    return -1
                logger.info("[MoveLinear] 指令下发成功")
                return 0
            except ModbusException as e:
                logger.error(f"直线运动触发异常: {e}")
                return -1

    def arm_move_joint(self, target: CartesianPose):
        """触发关节运动"""
        if not self._write_pose_registers(target):
            return -1
        
        logger.info("等待触发寄存器就绪...")
        if self.wait_until_idle(REG_TRIGGER_JOINT) != 0:
            return -1
            
        with self.lock:
            try:
                result = self.client.write_register(
                    address=REG_TRIGGER_JOINT,
                    value=1
                )
                if result.isError():
                    logger.error("[MoveJoint] 触发失败")
                    return -1
                logger.info("[MoveJoint] 指令下发成功")
                return 0
            except ModbusException as e:
                logger.error(f"关节运动触发异常: {e}")
                return -1

    def arm_set_coord_speed(self, speed):
        """设置坐标系运动速度 (0~100%)"""
        with self.lock:
            try:
                result = self.client.write_register(
                    address=REG_SPEED_SELECT,
                    value=speed
                )
                if result.isError():
                    logger.error(f"设置速度失败: {result}")
                    return -1
                logger.info(f"[SetSpeed] 速度已设置为 {speed}%")
                return 0
            except ModbusException as e:
                logger.error(f"设置速度异常: {e}")
                return -1

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("连接已关闭")
