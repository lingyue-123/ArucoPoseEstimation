import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def cam_from_pix(u, v, depth_mm, K, dist_coeffs):
    '''
    使用 OpenCV 函数将像素坐标和深度转换为相机坐标系下的3D点
    '''
    # 构建点集（OpenCV 要求 Nx1x2 格式）
    points = np.array([[[u, v]]], dtype=np.float32)

    # 去畸变，得到归一化平面坐标
    undistorted_points = cv2.undistortPoints(points, K, dist_coeffs)

    # 提取归一化坐标
    x_norm = undistorted_points[0, 0, 0]
    y_norm = undistorted_points[0, 0, 1]

    undistorted_points = cv2.undistortPoints(points, K, dist_coeffs)

    # 转换为相机坐标系下的3D点
    X_cam = x_norm * depth_mm
    Y_cam = y_norm * depth_mm
    Z_cam = depth_mm

    return np.array([X_cam, Y_cam, Z_cam])


def fit_plane_and_pose(pts):
    '''
    拟合平面并构造局部坐标系（6D位姿）
    输入:
        pts - Nx3 numpy array (相机坐标系下的点)
    返回:
        (R_plane2cam, t_plane2cam) - 平面局部坐标系到相机坐标系的变换
        normal - 平面法向量
        centroid - 平面中心点面中心点
    '''
    # 计算质心
    centroid = np.mean(pts, axis=0)

    # PCA拟合平面（最小特征值对应的特征向量为法向量）
    centered_pts = pts - centroid
    cov = np.cov(centered_pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # 最小特征值对应的特征向量为法线方向
    normal_idx = np.argmin(eigvals)
    normal = eigvecs[:, normal_idx]
    normal /= np.linalg.norm(normal)

    # 确保法线方向指向相机（Z轴正方向大致指向相机前方）
    camera_z = np.array([0, 0, 1])
    if np.dot(normal, camera_z) < 0:
        normal = -normal

    # 构造局部坐标系的X轴和Y轴
    ref = np.array([0, 1, 0])

    # 如果参考向量与法线平行，则换一个参考向量
    if abs(np.dot(ref, normal)) > 0.9999:
        ref = np.array([1, 0, 0])

    # X轴 = ref × normal
    x_axis = np.cross(ref, normal)
    x_axis /= np.linalg.norm(x_axis)

    # Y轴 = normal × X轴
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 构造旋转矩阵（局部坐标系到相机坐标系）
    R_plane2cam = np.column_stack((x_axis, y_axis, normal))

    # 平移向量（原点在相机坐标系下的坐标）
    t_plane2cam = centroid.reshape(3, 1)
    return R_plane2cam, t_plane2cam, normal, centroid


def transform_pose_to_base(R_plane2cam, t_plane2cam, R_cam2base, t_cam2base):
    '''
    将平面位姿从相机坐标系转换到基坐标系
    '''
    R_plane2base = R_cam2base @ R_plane2cam
    t_plane2base = R_cam2base @ t_plane2cam + t_cam2base
    return R_plane2base, t_plane2base

def rotate_plane_in_plane(R_plane2base, rotation_angle_deg):
    '''
    让平面绕其法线方向（局部Z轴）旋转指定角度
    输入:
        R_plane2base - 平面的旋转矩阵（基坐标系）
        rotation_angle_deg - 旋转角度（度）
    返回:
        R_rotated - 旋转后的旋转矩阵
    '''
    # 将旋转矩阵转换为欧拉角（使用XYZ顺序）
    euler_xyz = Rot.from_matrix(R_plane2base).as_euler("xyz", degrees=True)
    
    # 获取当前的Rx, Ry, Rz
    rx, ry, rz = euler_xyz
    
    # 在平面内旋转：相当于绕局部Z轴（法线方向）旋转
    # 由于欧拉角XYZ顺序中，Rz是绕世界Z轴旋转，需要调整
    # 这里采用更直接的方法：构建绕法线方向的旋转矩阵
    
    # 提取平面的法线方向（旋转矩阵的第三列）
    normal = R_plane2base[:, 2]
    
    # 构建绕法线方向旋转指定角度的旋转矩阵
    rotation_angle_rad = np.radians(rotation_angle_deg)
    
    # 使用罗德里格斯公式
    rot_vec = normal * rotation_angle_rad
    R_rotation = Rot.from_rotvec(rot_vec).as_matrix()
    
    # 应用旋转
    R_rotated = R_rotation @ R_plane2base
    
    # print(f"平面内旋转角度: {rotation_angle_deg}°")
    # print(f"旋转轴（法线方向）: {normal}")
    
    return R_rotated

def get_3D_pose(pts_pixel, depths, l_offset, h_offset, d_offset, euler_offset):
    # 相机内参
    K = camera_matrix = np.array([[611.696,	0,	643.269],
                                  [0, 611.671, 408.698],
                                  [0,	0,	1]])

    # 畸变系数
    dist_coeffs = np.array([-0.0231334, 0.0283417, 0.00021241, 0.000386759, -0.0102324])

    # 相机到基座的外参
    R_cam2base = np.array([[ 0.97515238,  0.05695814,  0.21408786],
                           [-0.03443397,  0.99360797, -0.10750585],
                           [-0.21884273,  0.09746269,  0.97088047]])

    t_cam2base = np.array([[ -11.33253739],
                           [ -80.82107591],
                           [-184.95612831]])

    # 步骤1：将像素坐标转换为相机坐标系下的3D点
    pts_cam = []
    for (u, v), depth in zip(pts_pixel, depths):
        pt_cam = cam_from_pix(u, v, depth, K, dist_coeffs)
        pts_cam.append(pt_cam)

    # 步骤2：拟合平面并计算相机坐标系下的6D位姿
    R_plane2cam, t_plane2cam, normal_cam, centroid_cam = fit_plane_and_pose(pts_cam)
    t_plane2cam[0] += l_offset

    # 步骤3：转换到基坐标系
    R_plane2base, t_plane2base = transform_pose_to_base(R_plane2cam, t_plane2cam, R_cam2base, t_cam2base)
    t_plane2base = t_plane2base[0]
    print(t_plane2base)
    t_plane2base[2] += h_offset
    
    # === 新增：在基坐标系下沿XY平面方向移动550mm ===
    displacement = d_offset  # mm
    
    # 获取当前中心点在基坐标系下的位置（3x1矩阵转换为1D数组）
    current_pos = t_plane2base.flatten()  # [x, y, z]
    
    # 计算当前中心点在XY平面的方向向量（从原点指向投影点）
    xy_direction = np.array([current_pos[0], current_pos[1], 0.0])
    xy_distance = np.linalg.norm(xy_direction)
    
    if xy_distance > 1e-6:  # 避免除零
        # 归一化方向向量
        xy_direction_normalized = xy_direction / xy_distance
        
        # 沿XY平面方向移动550mm
        displacement_vector = xy_direction_normalized * displacement
        
        # 移动后的新位置
        new_pos = current_pos + displacement_vector
        
        # 更新平移向量
        t_plane2base_shifted = new_pos.reshape(3, 1)
    
    # === 新增：让平面在平面上旋转90度 ===
    rotation_angle = euler_offset  # 旋转90度
    R_plane2base_rotated = rotate_plane_in_plane(R_plane2base, rotation_angle)
    
    # 使用旋转后的旋转矩阵
    R_plane2base_final = R_plane2base_rotated
    
    # 步骤4：转换为欧拉角
    euler_xyz = Rot.from_matrix(R_plane2base_final).as_euler("xyz", degrees=True)
    
    # 计算原始欧拉角用于对比
    euler_xyz_original = Rot.from_matrix(R_plane2base).as_euler("xyz", degrees=True)
    
    # print("\n=== 位姿信息 ===")
    # print(f"原始位置 (x, y, z): [{t_plane2base[0,0]:.3f}, {t_plane2base[1,0]:.3f}, {t_plane2base[2,0]:.3f}] mm")
    # print(f"原始欧拉角 (Rx, Ry, Rz): [{euler_xyz_original[0]:.3f}, {euler_xyz_original[1]:.3f}, {euler_xyz_original[2]:.3f}]°")
    # print(f"\n旋转{rotation_angle}°后的欧拉角 (Rx, Ry, Rz): [{euler_xyz[0]:.3f}, {euler_xyz[1]:.3f}, {euler_xyz[2]:.3f}]°")
    cover_pose = []
    for i in range(len(t_plane2base_shifted)):
        cover_pose.append(t_plane2base_shifted[i][0])
    for i in range(len(euler_xyz)):
        cover_pose.append(euler_xyz[i])

    return cover_pose