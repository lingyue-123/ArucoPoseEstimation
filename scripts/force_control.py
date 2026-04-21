import time
from typing import List, Tuple

# -------------------------- 模拟机械臂 SDK 接口（实际项目中替换为真实 SDK） --------------------------
class MockRobotSDK:
    """模拟机械臂 SDK 的核心接口（仅用于演示逻辑）"""
    ERR_SUCC = 0
    ABS = 0
    TRUE = True

    def login_in(self, ip: str) -> int:
        print(f"[SDK] 机械臂登录成功: {ip}")
        return self.ERR_SUCC

    def power_on(self) -> int:
        print("[SDK] 机械臂上电成功")
        return self.ERR_SUCC

    def enable_robot(self) -> int:
        print("[SDK] 机械臂使能成功")
        return self.ERR_SUCC

    def set_collision_level(self, level: int) -> int:
        print(f"[SDK] 碰撞等级设置成功: {level} (对应 {(level)*25}N)")
        return self.ERR_SUCC

    def set_torque_sensor_mode(self, mode: int) -> int:
        status = "开启" if mode == 1 else "关闭"
        print(f"[SDK] 力传感器{status}成功")
        return self.ERR_SUCC

    def set_ft_ctrl_frame(self, frame: int) -> int:
        frame_name = "工具坐标系" if frame == 0 else "世界坐标系"
        print(f"[SDK] 力控坐标系设置为: {frame_name}")
        return self.ERR_SUCC

    def set_compliant_speed_limit(self, vel: float, angular_vel: float) -> int:
        print(f"[SDK] 力控速度限制设置成功: 线速度={vel}mm/s, 角速度={angular_vel}rad/s")
        return self.ERR_SUCC

    def zero_end_sensor(self) -> int:
        print("[SDK] 末端力传感器校零成功")
        return self.ERR_SUCC

    def set_admit_ctrl_config(self, axis: int, enable: int, damping: int, force: float, *args) -> int:
        axis_names = ["X", "Y", "Z", "Mx", "My", "Mz"]
        status = "开启" if enable == 1 else "关闭"
        print(f"[SDK] {axis_names[axis]}轴恒力参数设置成功: {status}, 阻尼={damping}, 恒力={force}N")
        return self.ERR_SUCC

    def set_compliant_type(self, type_: int, init: int) -> int:
        if type_ == 1 and init == 0:
            print("[SDK] 第一步: 开启恒力柔顺模式")
        elif type_ == 0 and init == 1:
            print("[SDK] 第二步: 完成力控初始化")
        elif type_ == 0 and init == 0:
            print("[SDK] 关闭恒力柔顺模式")
        return self.ERR_SUCC

    def get_robot_status(self, status: dict) -> int:
        # 模拟实际力数据（实际项目中由传感器返回）
        # 这里模拟 Z 轴力在正常范围内波动
        status["torq_sensor_monitor_data"] = {
            "actTorque": [0, 0, 120]  # X, Y, Z 轴实际力 (单位: N)
        }
        return self.ERR_SUCC

    def motion_abort(self) -> int:
        print("[SDK] 机械臂运动已中止")
        return self.ERR_SUCC

    def is_in_pos(self, in_pos: List[bool]) -> int:
        in_pos[0] = True  # 模拟机械臂已停止
        return self.ERR_SUCC

    def is_in_collision(self, in_collision: List[bool]) -> int:
        in_collision[0] = False  # 模拟未触发碰撞
        return self.ERR_SUCC

    def collision_recover(self) -> int:
        print("[SDK] 已从碰撞保护中恢复")
        return self.ERR_SUCC


# -------------------------- 力控接口实现（对应 C++ robot_driver_interface.cpp） --------------------------
class RobotInterface:
    def __init__(self, robot_ip: str):
        self.robot = MockRobotSDK()
        self.robot_ip = robot_ip
        self.force_params = {
            "CollisionLevel": 5,
            "ForceControlSleepTime": 40,
            "ForceControlVelocity": 10,
            "ForceControlAngleVelocity": 0.1,
            "DampingForceThreshold": [150, 150, 170, 150, 150, 150],
            "ChargeGunConstantForce": 150,
            "PullOutGunConstantForce": -110,
            "ChargeGunActForceMax": -150,
            "PullOutGunActForceMax": 168,
        }

    def init_robot(self) -> int:
        """初始化机械臂（登录、上电、使能、设置碰撞等级）"""
        print("\n========== 机械臂初始化开始 ==========")
        if self.robot.login_in(self.robot_ip) != self.robot.ERR_SUCC:
            return -1
        time.sleep(1)
        if self.robot.power_on() != self.robot.ERR_SUCC:
            return -1
        time.sleep(1)
        if self.robot.enable_robot() != self.robot.ERR_SUCC:
            return -1
        time.sleep(1)
        if self.robot.set_collision_level(self.force_params["CollisionLevel"]) != self.robot.ERR_SUCC:
            return -1
        print("========== 机械臂初始化完成 ==========\n")
        return 0

    def force_control(self, charge_flag: bool) -> int:
        """
        开启力控并配置参数
        :param charge_flag: True=充电模式(插枪), False=拔枪模式
        """
        print("\n========== 力控初始化开始 ==========")
        # 1. 开启力传感器
        if self.robot.set_torque_sensor_mode(1) != self.robot.ERR_SUCC:
            return -1
        # 2. 设置力控坐标系（工具坐标系）
        if self.robot.set_ft_ctrl_frame(0) != self.robot.ERR_SUCC:
            return -1
        # 3. 设置力控速度限制
        if self.robot.set_compliant_speed_limit(
            self.force_params["ForceControlVelocity"],
            self.force_params["ForceControlAngleVelocity"]
        ) != self.robot.ERR_SUCC:
            return -1
        # 4. 力传感器校零
        self.robot.zero_end_sensor()
        time.sleep(0.5)
        # 5. 配置 6 轴恒力柔顺参数
        damping = self.force_params["DampingForceThreshold"]
        # X, Y, Mx, My, Mz 轴（仅开启阻尼，无恒力）
        for axis in [0, 1, 3, 4, 5]:
            if self.robot.set_admit_ctrl_config(axis, 1, damping[axis], 0, 0, 0) != self.robot.ERR_SUCC:
                return -1
        # Z 轴（根据模式设置恒力）
        z_force = (
            self.force_params["ChargeGunConstantForce"]
            if charge_flag
            else self.force_params["PullOutGunConstantForce"]
        )
        if self.robot.set_admit_ctrl_config(2, 1, damping[2], z_force, 0, 0) != self.robot.ERR_SUCC:
            return -1
        # 6. 分两步开启恒力柔顺控制（兼容控制器版本）
        if self.robot.set_compliant_type(1, 0) != self.robot.ERR_SUCC:
            return -1
        time.sleep(1)
        if self.robot.set_compliant_type(0, 1) != self.robot.ERR_SUCC:
            return -1
        time.sleep(0.2)
        print("========== 力控初始化完成，机械臂开始柔性动作 ==========\n")
        return 0

    def status_monitor(self, force_status: List[bool]) -> None:
        """
        实时监控力控状态
        :param force_status: 引用传递的标志位，force_status[0] = True 表示触发异常
        """
        status = {"torq_sensor_monitor_data": {"actTorque": [0, 0, 0]}}
        self.robot.get_robot_status(status)
        z_force = status["torq_sensor_monitor_data"]["actTorque"][2]
        print(f"[监控] 当前 Z 轴实际力: {z_force} N")

        # 检查力是否超出阈值
        if (
            z_force < self.force_params["ChargeGunActForceMax"]
            or z_force > self.force_params["PullOutGunActForceMax"]
        ):
            print(f"[警告] Z 轴力超出阈值！触发安全保护")
            self.robot.motion_abort()
            force_status[0] = True

            # 确认机械臂停止后关闭力控
            in_pos = [False]
            self.robot.is_in_pos(in_pos)
            if in_pos[0]:
                in_collision = [False]
                self.robot.is_in_collision(in_collision)
                if in_collision[0]:
                    self.robot.collision_recover()
                self.force_close()

    def force_close(self) -> None:
        """关闭力控"""
        print("\n========== 关闭力控 ==========")
        self.robot.set_compliant_type(0, 0)
        self.robot.set_torque_sensor_mode(0)
        print("========== 力控已关闭 ==========\n")


# -------------------------- 主流程（对应 C++ linear_move_gunInsertPort） --------------------------
def linear_move_gun_insert_port(robot_interface: RobotInterface) -> bool:
    """力控插枪主流程"""
    print("\n>>>>>>>>>> 开始执行力控插枪流程 <<<<<<<<<<")
    force_status = [False]
    count_time = 0

    # 1. 模拟直线运动到充电口附近（实际项目中替换为真实运动代码）
    print("\n[步骤 1] 直线运动到充电口附近...")
    time.sleep(2)  # 模拟运动耗时
    print("[步骤 1] 运动到充电口成功\n")

    # 2. 开启力控（充电模式）
    ret_force_ctr = robot_interface.force_control(charge_flag=True)
    if ret_force_ctr != 0:
        print("[错误] 力控初始化失败！")
        return False

    # 3. 实时监控循环
    print(f"[步骤 2] 进入力控监控循环（最长等待 {robot_interface.force_params['ForceControlSleepTime']} 秒）")
    while count_time < robot_interface.force_params["ForceControlSleepTime"] and not force_status[0]:
        count_time += 1
        robot_interface.status_monitor(force_status)
        time.sleep(1)

    # 4. 关闭力控
    if ret_force_ctr == 0:
        robot_interface.force_close()

    print(">>>>>>>>>> 力控插枪流程执行完毕 <<<<<<<<<<\n")
    return True


if __name__ == "__main__":
    # 初始化机械臂接口（替换为实际机械臂 IP）
    robot_ip = "192.168.1.106"
    robot_interface = RobotInterface(robot_ip)

    # 执行机械臂初始化
    if robot_interface.init_robot() != 0:
        print("[错误] 机械臂初始化失败！")
        exit(1)

    # 执行力控插枪流程
    success = linear_move_gun_insert_port(robot_interface)
    if success:
        print("[结果] 插枪流程成功完成！")
    else:
        print("[结果] 插枪流程失败！")
