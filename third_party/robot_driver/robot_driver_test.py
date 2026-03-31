from robot_driver_interface import RobotInterface, CartesianPose
import time


def main():
    # 1. 初始化机器人接口（替换为你的机器人实际IP）
    robot = RobotInterface(ip="192.168.1.133", port=502, unit_id=1)
    
    # 2. 建立连接
    if not robot.connect():
        print("连接失败")
        return
    
    try:


        current_pose = robot.arm_get_current_pose()
        if current_pose:
            print(f"当前位姿：{current_pose}")
        else:
            print("获取位姿失败")
        
        coord = robot.arm_get_coord_sys()
        if coord != -1:
            print(f"当前坐标系：{coord}")
        else:
            print("读取坐标系失败")
        
        status = robot.arm_get_status()
        if status != -1:
            status_desc = "就绪" if status == 0 else "忙碌" if status == 1 else f"未知状态({status})"
            print(f"机械臂状态：{status} ({status_desc})")
        else:
            print("读取状态失败")

        ## 关节运动
        # target_pose = CartesianPose(-104.47, 32.37, -38.56, 173.43, 26.59, 99.85)
        # move_result = robot.arm_move_joint(target_pose)
        # if move_result == 0:
        #     print(f"关节运动指令已下发")
        # else:
        #     print("关节运动触发失败")


        ## 直线运动 不切换坐标系默认为世界坐标系
        target_pose = CartesianPose(x=179.488, y=-606.283, z=60, rx=-126.776, ry=-70.741, rz=-122.100) # 世界坐标系位姿
        move_result = robot.arm_move_linear(target_pose)
        if move_result == 0:
            print(f"直线运动指令已下发，目标位姿：{target_pose}")
        else:
            print("直线运动触发失败")
        

        ## 直线运动 设置坐标系后触发 0:世界坐标系 1：相机坐标系 2：法兰坐标系
        # target_coord = 1  # 目标坐标系编号 相机坐标系
        # set_result = robot.arm_set_coord_sys(target_coord)
        # if set_result == 0:
        #     print(f"成功设置坐标系为：{target_coord}")
        # else:
        #     print("设置坐标系失败")
        
        # # target_pose = CartesianPose(x=338.17, y=-814.18, z=95.81, rx=169.45, ry=-70.47, rz=-55.91) # 相机坐标系位姿
        # move_result = robot.arm_move_linear(target_pose)
        # if move_result == 0:
        #     print(f"直线运动指令已下发，目标位姿：{target_pose}")
        # else:
        #     print("直线运动触发失败")

        
    finally:
        while robot.arm_get_status():
            time.sleep(0.1)
        robot.close()
        print("连接已关闭")

if __name__ == "__main__":
    main()
