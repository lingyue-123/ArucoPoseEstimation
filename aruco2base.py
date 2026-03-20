import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(suppress=True)

def euler_xyz_to_rotmat(rx, ry, rz, degrees=True):
    """
    将XYZ欧拉角转为3×3旋转矩阵
    :param rx/ry/rz: 欧拉角（默认角度制，degrees=False为弧度制）
    :return: 3×3旋转矩阵
    """
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=degrees)
    return rot.as_matrix()

def homogeneous_matrix(rotmat, t):
    """
    由旋转矩阵+平移向量构建4×4齐次矩阵
    :param rotmat: 3×3旋转矩阵
    :param t: 平移向量 [x,y,z]
    :return: 4×4齐次矩阵
    """
    T = np.eye(4)
    T[:3, :3] = rotmat
    T[:3, 3] = t
    return T

def get_eye2base_matrix(hand_xyzrpy, eye2hand, degrees=True):
    """
    计算相机（眼）在base下的4×4齐次矩阵
    :param hand_xyzrpy: 手在base下的位姿 [x,y,z,rx,ry,rz]
    :param eye2hand: 眼到手的4×4齐次矩阵（手眼标定结果）
    :param degrees: 欧拉角是否为角度制（默认True）
    :return: 眼在base下的4×4齐次矩阵
    """
    # 1. 解析手的位姿
    x, y, z, rx, ry, rz = hand_xyzrpy
    
    # 2. 构建手在base下的齐次矩阵
    R_hand_base = euler_xyz_to_rotmat(rx, ry, rz, degrees)
    hand2base = homogeneous_matrix(R_hand_base, [x, y, z])
    
    # 3. 计算眼在base下的齐次矩阵
    eye2base = hand2base @ eye2hand # 等价写法：np.dot(hand2base, eye2hand)
    
    return eye2base

def get_ar2base_matrix(ar2cam_xyzrpy, cam2base, degrees=True):
    """
    计算AR在base下的4×4齐次矩阵
    :param ar2cam_xyzrpy: AR在相机下的位姿 [x,y,z,rx,ry,rz]
    :param cam2base: 相机在base下的4×4齐次矩阵
    :param degrees: 欧拉角是否为角度制（默认True）
    :return: AR在base下的4×4齐次矩阵
    """
    # 1. 解析AR在相机下的位姿
    x, y, z, rx, ry, rz = ar2cam_xyzrpy
    
    # 2. 构建AR在相机下的齐次矩阵
    R_ar_cam = euler_xyz_to_rotmat(rx, ry, rz, degrees)
    ar2cam = homogeneous_matrix(R_ar_cam, [x, y, z])
    
    # 3. 计算AR在base下的齐次矩阵
    ar2base = cam2base @ ar2cam # 等价写法：np.dot(cam2base, ar2cam)
    
    return ar2base

if __name__ == "__main__":
    # ====================== 配置参数 ======================
    # 1. 手在base下的位姿 [x,y,z,rx,ry,rz]
    hand_xyzrpy = [-520.401,-694.835,-13.092,-138.1509,-68.7359,-176.8607]
    
    # 2. 手到眼的4×4齐次矩阵（手眼标定结果）
    eye2hand_matrix = np.array([
        [0.91375753, 0.40452578, -0.03749755, 45.15799675],
        [-0.40282893, 0.91414316, 0.04550981, -89.33185268],
        [0.05268802, -0.02647983, 0.99825988, 170.96361783],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    
    # 3. AR在相机下的位姿 [x,y,z,rx,ry,rz]
    ar2cam_xyzrpy = [-26.635,-9.251,358.782,-174.9394,5.6557,2.5923]
    
    # ====================== 计算流程 ======================
    # 1. 计算相机在base下的矩阵
    eye2base_matrix = get_eye2base_matrix(hand_xyzrpy, eye2hand_matrix)
    print("=== 相机（眼）在base下的齐次矩阵 ===")
    print(np.round(eye2base_matrix, 4))
    print()
    
    # 2. 计算AR在base下的矩阵
    ar2base_matrix = get_ar2base_matrix(ar2cam_xyzrpy, eye2base_matrix)
    print("=== AR在base下的最终转移矩阵 ===")
    print(np.round(ar2base_matrix, 4))
    
    # 可选：输出AR在base下的欧拉角和平移向量
    ar2base_rot = R.from_matrix(ar2base_matrix[:3, :3])
    ar2base_euler = ar2base_rot.as_euler('xyz', degrees=True)
    ar2base_trans = ar2base_matrix[:3, 3]
    print()
    print("=== AR在base下的位姿信息 ===")
    print(f"平移向量 [x, y, z]: {np.round(ar2base_trans, 4)}")
    print(f"欧拉角 [rx, ry, rz] (角度制): {np.round(ar2base_euler, 4)}")
