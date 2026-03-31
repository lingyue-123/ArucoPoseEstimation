import numpy as np

def get_robot_pose(index):
    """
    模拟getRobotPose函数，返回4x4变换矩阵。
    在实际应用中，这里应该调用机器人接口获取当前位姿。
    这里返回一个示例矩阵（单位矩阵，表示基坐标系）。
    """
    if index == 0:
        # 示例：返回单位矩阵，表示机器人当前在基坐标系原点，无旋转
        return np.eye(4)
    else:
        # 其他坐标系的模拟
        return np.eye(4)  # 简化处理

def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角 (ZYX顺序，与C++代码一致)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6  # 奇异情况
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])

def matrix_to_cartesian(matrix):
    """
    将4x4变换矩阵转换为CartesianPose格式。
    返回字典：{'tran': {'x': float, 'y': float, 'z': float}, 'rpy': {'rx': float, 'ry': float, 'rz': float}}
    """
    # 提取平移向量
    tran = matrix[:3, 3]
    
    # 提取旋转矩阵
    rot_matrix = matrix[:3, :3]
    
    # 将旋转矩阵转换为欧拉角 (ZYX顺序，与C++代码一致)
    rpy = rotation_matrix_to_euler(rot_matrix)
    
    return {
        'tran': {'x': tran[0], 'y': tran[1], 'z': tran[2]},
        'rpy': {'rx': rpy[0], 'ry': rpy[1], 'rz': rpy[2]}
    }

def statMechCtrl_callback_simulation():
    """
    模拟statMechCtrl_callback中的沿z轴移动逻辑。
    """
    # 假设已经获取了aruco_final_pose，这里直接模拟移动到目标后进行z轴移动
    
    # 获取zmove参数（示例值）
    zmove = 50.0  # 假设zmove为50mm，实际应从参数服务器获取
    
    # 构造lastZTrans矩阵（沿z轴平移zmove）
    lastZTrans = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, zmove],
        [0, 0, 0, 1]
    ])
    
    # 获取当前机器人位姿（假设在基坐标系0）
    current_pose_matrix = get_robot_pose(0)
    
    # 计算新的位姿：setObjPose = getRobotPose(0) * lastZTrans
    setObjPose = current_pose_matrix @ lastZTrans
    
    # 转换为CartesianPose格式
    falan_pose = matrix_to_cartesian(setObjPose)
    
    # 输出结果
    print("=== 沿z轴移动后的目标位姿 ===")
    print(f"Trans (x, y, z): [{falan_pose['tran']['x']:.3f}, {falan_pose['tran']['y']:.3f}, {falan_pose['tran']['z']:.3f}]")
    print(f"RPY (rx, ry, rz): [{np.degrees(falan_pose['rpy']['rx']):.3f}, {np.degrees(falan_pose['rpy']['ry']):.3f}, {np.degrees(falan_pose['rpy']['rz']):.3f}]")
    print("=============================")
    
    return falan_pose

if __name__ == "__main__":
    # 运行模拟
    result = statMechCtrl_callback_simulation()
    print("模拟完成，结果：", result)