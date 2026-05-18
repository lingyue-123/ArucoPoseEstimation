import cv2
import numpy as np
import glob
import os
from natsort import natsorted
import math
from scipy.spatial.transform import Rotation as Rot


class CalibrationBoardPoseEstimator:
    '''
    标定板位姿估计类
    '''

    def __init__(self, camera_matrix, dist_coeffs, pattern_size=(11, 8), square_size=25.0):
        '''
        初始化标定板位姿估计器
        参数:
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 相机畸变系数 (1x5 或 1x4)
            pattern_size: 棋盘格内角点数 (宽, 高)，例如 (9,6)
            square_size: 棋盘格格子边长，单位 mm
        '''
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pattern_size = pattern_size
        self.square_size = square_size

        # 生成标定板上的3D点（所有图像共用）
        self.object_points = self._generate_object_points()

        # 用于角点检测的迭代终止条件
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def _generate_object_points(self):
        """生成标定板上的3D点（Z=0平面）"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def compute_reprojection_error(self, object_points, image_points, rvec, tvec):
        '''
        计算重投影误差
        参数:
            object_points: 3D物体点
            image_points: 2D图像点
            rvec: 旋转向量
            tvec: 平移向量
        返回:
            mean_error: 平均重投影误差（像素）
            reprojected_points: 重投影点
        '''
        # 将3D点投影到图像平面
        reprojected_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        # 计算每个点的误差
        reprojected_points = reprojected_points.reshape(-1, 2)
        errors = np.linalg.norm(image_points - reprojected_points, axis=1)
        
        # 计算统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        return mean_error, std_error, max_error, min_error, reprojected_points

    def detect_board_pose(self, image, compute_error=True):
        '''
        从单张图像中检测标定板并计算其位姿
        参数:
            image: 输入图像 (BGR格式)
            compute_error: 是否计算重投影误差
        返回:
            success: 是否成功检测
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3x1)
            corners: 检测到的角点坐标
            reprojection_error: 重投影误差统计信息（元组）
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        if not ret:
            return False, None, None, None, None

        # 亚像素精细化角点坐标
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
        corners_reshaped = corners2.reshape(-1, 2)

        # 使用solvePnP求解标定板位姿
        ret, rvec, tvec = cv2.solvePnP(self.object_points, corners_reshaped, 
                                       self.camera_matrix, self.dist_coeffs)
        if not ret:
            return False, None, None, None, None

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 计算重投影误差
        reprojection_error = None
        if compute_error:
            mean_error, std_error, max_error, min_error, reprojected = self.compute_reprojection_error(
                self.object_points, corners_reshaped, rvec, tvec
            )
            reprojection_error = {
                'mean': mean_error,
                'std': std_error,
                'max': max_error,
                'min': min_error,
                'reprojected_points': reprojected,
                'detected_points': corners_reshaped
            }

        return True, R, tvec, corners2, reprojection_error


def load_robot_poses_from_list(pose_list):
    '''
    从列表加载机械臂位姿数据（修正版）
    参数:
        pose_list: 包含多个位姿的列表，每个位姿是 [x, y, z, rx, ry, rz]
                   (x,y,z: 机械臂末端在基座标系下的位置，单位 mm)
                   (rx,ry,rz: 机械臂末端在基座标系下的欧拉角，单位 度，XYZ顺序)
    返回:
        R_end_to_base_list: 旋转矩阵列表 (3x3)，表示从末端坐标系到基座标系的旋转
        t_end_to_base_list: 平移向量列表 (3x1)，表示末端坐标系原点在基座标系中的位置
    '''
    R_end_to_base_list = []
    t_end_to_base_list = []

    for i, pose in enumerate(pose_list):
        x, y, z = pose[0:3]
        rx, ry, rz = pose[3:6]
        
        # 平移向量：末端在基座标系下的位置
        t = np.array([[x], [y], [z]], dtype=np.float64)
        
        # 旋转矩阵：末端在基座标系下的姿态
        # 使用XYZ欧拉角顺序（与机械臂通常的输出一致）
        R = Rot.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        
        R_end_to_base_list.append(R)
        t_end_to_base_list.append(t)

    return R_end_to_base_list, t_end_to_base_list


def verify_robot_pose_conversion(pose_list):
    '''
    验证机械臂位姿转换是否正确
    打印原始位姿和转换后的矩阵，便于人工检查
    '''
    print("=" * 60)
    print("验证机械臂位姿转换")
    print("=" * 60)
    
    for i, pose in enumerate(pose_list[:3]):  # 只检查前3个
        x, y, z = pose[0:3]
        rx, ry, rz = pose[3:6]
        
        print(f"\n位姿 {i+1}:")
        print(f"  位置 (mm): x={x:.3f}, y={y:.3f}, z={z:.3f}")
        print(f"  欧拉角 (度): rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}")
        
        # 计算旋转矩阵
        R = Rot.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        print(f"  旋转矩阵 R_end_to_base:")
        print(f"    [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]")
        print(f"    [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]")
        print(f"    [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]")
        
        # 验证旋转矩阵是正交的（应该是单位正交矩阵）
        det = np.linalg.det(R)
        print(f"  行列式: {det:.6f} (应为 ±1.0)")
        
        # 验证旋转矩阵的逆等于转置
        R_inv = np.linalg.inv(R)
        R_T = R.T
        print(f"  逆矩阵与转置的差异: {np.max(np.abs(R_inv - R_T)):.2e}")


def visualize_reprojection(image, reprojection_error, save_path=None):
    '''
    可视化重投影结果
    参数:
        image: 原始图像
        reprojection_error: 重投影误差信息字典
        save_path: 保存路径（如果为None则显示）
    '''
    if reprojection_error is None:
        return
    
    # 创建彩色图像用于可视化
    vis_img = image.copy()
    
    # 绘制检测到的角点（绿色）
    detected_pts = reprojection_error['detected_points']
    for pt in detected_pts:
        cv2.circle(vis_img, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
    
    # 绘制重投影点（红色）
    reprojected_pts = reprojection_error['reprojected_points']
    for pt in reprojected_pts:
        cv2.circle(vis_img, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
    
    # 绘制误差连线（黄色）
    for det_pt, reproj_pt in zip(detected_pts, reprojected_pts):
        cv2.line(vis_img, tuple(det_pt.astype(int)), 
                tuple(reproj_pt.astype(int)), (0, 255, 255), 1)
    
    # 添加误差信息文本
    info_text = [
        f"Mean Error: {reprojection_error['mean']:.3f} px",
        f"Std Error: {reprojection_error['std']:.3f} px",
        f"Max Error: {reprojection_error['max']:.3f} px"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(vis_img, text, (10, 30 + i * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"    可视化图像已保存: {save_path}")
    else:
        cv2.imshow('Reprojection Visualization', vis_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # ========== 1. 设置相机内参 ==========
    camera_matrix = np.array([[611.696, 0, 643.269],
                              [0, 611.671, 408.698],
                              [0, 0, 1]], dtype=np.float64)

    # 畸变系数
    dist_coeffs = np.array([-0.0231334, 0.0283417, 0.00021241, 0.000386759, -0.0102324], dtype=np.float64)

    # ========== 2. 设置标定板参数 ==========
    pattern_size = (11, 8)  # 棋盘格内角点数（宽，高）
    square_size = 25.0      # 格子边长，单位 mm

    # ========== 3. 设置重投影误差阈值 ==========
    MAX_REPROJECTION_ERROR = 1.5  # 最大允许的平均重投影误差（像素）
    MAX_STD_ERROR = 1.0           # 最大允许的标准差（像素）
    MAX_POINT_ERROR = 3.0         # 最大允许的单个点误差（像素）

    # ========== 4. 设置图像路径 ==========
    image_dir = "./ex_417"  # 存放标定图像的文件夹
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    # 创建可视化结果保存目录
    viz_dir = "./reprojection_viz"
    os.makedirs(viz_dir, exist_ok=True)

    # 收集所有图像文件
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = natsorted(image_paths)
    print(f"找到 {len(image_paths)} 张图像")

    # ========== 5. 创建位姿估计器 ==========
    pose_estimator = CalibrationBoardPoseEstimator(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size,
        square_size=square_size)

    # ========== 6. 存储所有图像的位姿结果（通过重投影误差筛选） ==========
    R_target2cam_list = []   # 旋转矩阵列表
    t_target2cam_list = []   # 平移向量列表
    valid_images = []        # 成功检测且误差合格的图像路径
    valid_indices = []       # 有效图像的索引
    rejected_images = []     # 被拒绝的图像及原因
    all_errors = []          # 所有成功检测的图像的误差信息

    for i, img_path in enumerate(image_paths):
        print(f"\n处理图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ 警告：无法读取图像，跳过")
            rejected_images.append((img_path, "无法读取图像"))
            continue

        # 检测标定板位姿并计算重投影误差
        success, R, t, corners, reproj_error = pose_estimator.detect_board_pose(img, compute_error=True)

        if not success:
            print(f"  ❌ 标定板检测失败，跳过")
            rejected_images.append((img_path, "标定板检测失败"))
            continue

        # 计算误差统计
        mean_error = reproj_error['mean']
        std_error = reproj_error['std']
        max_error = reproj_error['max']
        
        # 判断图像是否可用于手眼标定
        is_valid = (mean_error <= MAX_REPROJECTION_ERROR and 
                   std_error <= MAX_STD_ERROR and 
                   max_error <= MAX_POINT_ERROR)
        
        if is_valid:
            print(f"  ✅ 检测成功 - 平均误差: {mean_error:.3f}px, 标准差: {std_error:.3f}px, 最大误差: {max_error:.3f}px")
            print(f"     图像可用于手眼标定")
            
            # 存储结果
            R_target2cam_list.append(R)
            t_target2cam_list.append(t)
            valid_images.append(img_path)
            valid_indices.append(i)
            all_errors.append(reproj_error)
            
            # 可选：保存可视化图像
            viz_path = os.path.join(viz_dir, f"valid_{i+1:03d}_{os.path.basename(img_path)}")
            visualize_reprojection(img, reproj_error, save_path=viz_path)
        else:
            reject_reasons = []
            if mean_error > MAX_REPROJECTION_ERROR:
                reject_reasons.append(f"平均误差过高({mean_error:.3f} > {MAX_REPROJECTION_ERROR})")
            if std_error > MAX_STD_ERROR:
                reject_reasons.append(f"标准差过大({std_error:.3f} > {MAX_STD_ERROR})")
            if max_error > MAX_POINT_ERROR:
                reject_reasons.append(f"最大误差过大({max_error:.3f} > {MAX_POINT_ERROR})")
            
            print(f"  ❌ 检测成功但重投影误差不合格 - {', '.join(reject_reasons)}")
            print(f"     平均误差: {mean_error:.3f}px, 标准差: {std_error:.3f}px, 最大误差: {max_error:.3f}px")
            print(f"     图像不适合用于手眼标定，已丢弃")
            
            rejected_images.append((img_path, ", ".join(reject_reasons)))
            
            # 可选：保存被拒绝图像的可视化（用于调试）
            viz_path = os.path.join(viz_dir, f"rejected_{i+1:03d}_{os.path.basename(img_path)}")
            visualize_reprojection(img, reproj_error, save_path=viz_path)

    # ========== 7. 输出详细统计信息 ==========
    print("\n" + "=" * 70)
    print("处理完成！")
    print(f"总图像数: {len(image_paths)}")
    print(f"成功检测: {len(valid_images) + len(rejected_images)}/{len(image_paths)} 张图像")
    print(f"  - 有效图像（误差合格）: {len(valid_images)} 张")
    print(f"  - 无效图像（误差超限）: {len(rejected_images)} 张")
    
    if len(valid_images) < 5:
        print(f"\n⚠️ 警告：有效图像数量不足（{len(valid_images)}/5），手眼标定结果可能不准确！")
    
    # 输出被拒绝的图像列表
    if rejected_images:
        print("\n被拒绝的图像列表：")
        for img_path, reason in rejected_images:
            print(f"  - {os.path.basename(img_path)}: {reason}")
    
    # 输出误差统计
    if all_errors:
        errors_mean = [e['mean'] for e in all_errors]
        errors_std = [e['std'] for e in all_errors]
        errors_max = [e['max'] for e in all_errors]
        
        print("\n有效图像的误差统计：")
        print(f"  平均误差: {np.mean(errors_mean):.3f} ± {np.std(errors_mean):.3f} px")
        print(f"  标准差: {np.mean(errors_std):.3f} ± {np.std(errors_std):.3f} px")
        print(f"  最大误差: {np.mean(errors_max):.3f} ± {np.std(errors_max):.3f} px")
        print(f"  可视化图像已保存至: {viz_dir}")
    
    # ========== 8. 机械臂位姿数据 ==========
    # 注意：这里应该与图像顺序对应
    end_pose = [
        [190.122973, -535.226942, 125.527981, 156.745490, -78.082289, -58.330155],
        [382.647933, -505.202421, 109.582415, 156.735788, -78.079192, -58.321151],
        [-138.192211, -586.441301, 152.706986, 156.750615, -78.079168, -58.334033],
        [173.154770, -561.710331, 249.060161, 156.736607, -78.077408, -58.320798],
        [-241.101406, -624.874170, 274.989218, 31.697928, -81.079001, 67.479481],
        [-241.090721, -624.771663, 274.989502, -98.042803, -77.752799, -173.243464],
        [-178.881840, -620.451700, 89.855097, -111.937743, -77.755310, -173.242765],
        [517.766611, -683.821848, 296.631774, 175.563321, -72.177883, -88.355032],
        [517.962932, -683.959329, 296.557395, 140.975430, -87.892007, -22.955146],
        [624.244556, -663.016697, -214.082843, 83.739578, -52.017316, 11.890384],
        [609.741950, -543.948602, -222.133760, 83.737590, -52.018487, 11.891428],
        [556.347820, -541.608249, -110.343083, 46.667443, -84.598241, 39.580658],
        [241.490457, -681.527467, 52.832948, 90.282223, -82.979847, 1.512484],
        [649.134591, -668.871301, 2.720495, 20.305316, -69.387702, 72.694714],
        [614.600482, -670.287804, 7.056826, 129.509660, -80.473203, -37.201948],
        [179.362731, -667.770836, -154.277406, 129.523395, -80.476798, -37.217532],
        [243.350933, -716.383753, 324.907600, 129.521907, -80.478570, -37.216666],
        [-264.664781, -715.704717, 157.749507, 129.546048, -80.475841, -37.239403],
        [-192.711162, -803.551658, 358.433436, 162.276920, -65.221876, -71.119305],
        [759.855292, -760.850117, 254.856736, 113.630394, -63.457045, -20.020562],
        [618.833195, -724.337047, -227.073824, 60.218405, -77.586287, 60.071835],
        [729.414524, -751.953522, -259.624388, 94.534632, -74.024000, -18.74497],
        [-511.114743, -964.720172, -200.665125, -131.762140, -64.909608, -132.747146],
        [-483.348309, -1086.672777, 25.140930, 171.515496, -76.487026, -72.473048]
    ]
    
    # 验证机械臂位姿转换
    verify_robot_pose_conversion(end_pose)
    
    # ========== 9. 执行手眼标定（仅使用有效图像） ==========
    if len(valid_images) >= 3:
        print("\n" + "=" * 70)
        print("开始手眼标定...")
        print("=" * 70)
        
        # 只使用有效图像对应的机械臂位姿
        valid_end_pose = [end_pose[idx] for idx in valid_indices]
        
        print(f"\n使用 {len(valid_end_pose)} 组数据进行手眼标定")
        print(f"有效图像索引: {valid_indices}")
        
        # 使用修正后的转换函数
        R_end_to_base_list, t_end_to_base_list = load_robot_poses_from_list(valid_end_pose)
        
        # 验证数据一致性
        print(f"\n数据一致性检查:")
        print(f"  相机位姿数量: {len(R_target2cam_list)}")
        print(f"  机械臂位姿数量: {len(R_end_to_base_list)}")
        
        if len(R_target2cam_list) != len(R_end_to_base_list):
            print(f"  ❌ 数据数量不匹配！")
            exit(1)
        else:
            print(f"  ✅ 数据数量匹配")
        
        # 尝试不同的手眼标定方法
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai (经典方法)"),
            (cv2.CALIB_HAND_EYE_PARK, "Park (基于四元数)"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff (SVD方法)")
        ]
        
        results = []
        
        for method, method_name in methods:
            print(f"\n{'='*50}")
            print(f"使用 {method_name} 方法:")
            print(f"{'='*50}")
            
            try:
                R_cam_to_base, t_cam_to_base = cv2.calibrateHandEye(
                    R_gripper2base=R_end_to_base_list,
                    t_gripper2base=t_end_to_base_list,
                    R_target2cam=R_target2cam_list,
                    t_target2cam=t_target2cam_list,
                    method=method,
                )
                
                print(f"\n手眼标定结果 (相机->基座):")
                print(f"\n旋转矩阵 R_cam_to_base:")
                print(f"  [{R_cam_to_base[0,0]:.6f}, {R_cam_to_base[0,1]:.6f}, {R_cam_to_base[0,2]:.6f}]")
                print(f"  [{R_cam_to_base[1,0]:.6f}, {R_cam_to_base[1,1]:.6f}, {R_cam_to_base[1,2]:.6f}]")
                print(f"  [{R_cam_to_base[2,0]:.6f}, {R_cam_to_base[2,1]:.6f}, {R_cam_to_base[2,2]:.6f}]")
                
                print(f"\n平移向量 t_cam_to_base (mm):")
                print(f"  [{t_cam_to_base[0,0]:.3f}, {t_cam_to_base[1,0]:.3f}, {t_cam_to_base[2,0]:.3f}]")
                
                # 将旋转矩阵转换为欧拉角
                r = Rot.from_matrix(R_cam_to_base)
                euler_angles = r.as_euler('xyz', degrees=True)
                print(f"\n欧拉角 (xyz, 度):")
                print(f"  rx: {euler_angles[0]:.3f}°")
                print(f"  ry: {euler_angles[1]:.3f}°")
                print(f"  rz: {euler_angles[2]:.3f}°")
                
                # 计算平移向量的模长
                translation_norm = np.linalg.norm(t_cam_to_base)
                print(f"\n平移向量模长: {translation_norm:.3f} mm")
                
                results.append({
                    'method': method_name,
                    'R': R_cam_to_base,
                    't': t_cam_to_base,
                    'euler': euler_angles,
                    'norm': translation_norm
                })
                
            except Exception as e:
                print(f"\n❌ {method_name} 方法失败: {e}")
        
        # 比较不同方法的结果
        if len(results) >= 2:
            print("\n" + "=" * 70)
            print("不同方法结果对比:")
            print("=" * 70)
            
            for i in range(len(results)):
                for j in range(i+1, len(results)):
                    # 计算旋转矩阵差异
                    R_diff = results[i]['R'] @ results[j]['R'].T
                    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)) * 180 / np.pi
                    
                    # 计算平移向量差异
                    t_diff = np.linalg.norm(results[i]['t'] - results[j]['t'])
                    
                    print(f"\n{results[i]['method']} vs {results[j]['method']}:")
                    print(f"  旋转差异角度: {angle_diff:.3f}°")
                    print(f"  平移差异: {t_diff:.3f} mm")
        
        # 保存标定结果
        if results:
            print("\n" + "=" * 70)
            print("保存标定结果...")
            print("=" * 70)
            
            # 使用第一种方法的结果作为最终结果
            final_R = results[0]['R']
            final_t = results[0]['t']
            
            # 保存到文件
            with open('hand_eye_calibration_result.txt', 'w') as f:
                f.write("手眼标定结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"使用有效图像数量: {len(valid_images)}\n")
                f.write(f"平均重投影误差: {np.mean(errors_mean):.3f} px\n\n")
                
                f.write("旋转矩阵 R_cam_to_base:\n")
                f.write(f"{final_R[0,0]:.6f} {final_R[0,1]:.6f} {final_R[0,2]:.6f}\n")
                f.write(f"{final_R[1,0]:.6f} {final_R[1,1]:.6f} {final_R[1,2]:.6f}\n")
                f.write(f"{final_R[2,0]:.6f} {final_R[2,1]:.6f} {final_R[2,2]:.6f}\n\n")
                
                f.write("平移向量 t_cam_to_base (mm):\n")
                f.write(f"{final_t[0,0]:.3f} {final_t[1,0]:.3f} {final_t[2,0]:.3f}\n\n")
                
                r = Rot.from_matrix(final_R)
                euler = r.as_euler('xyz', degrees=True)
                f.write("欧拉角 (xyz, 度):\n")
                f.write(f"{euler[0]:.3f} {euler[1]:.3f} {euler[2]:.3f}\n")
            
            print("结果已保存到: hand_eye_calibration_result.txt")
            
    else:
        print(f"\n❌ 有效图像数量不足 ({len(valid_images)}/3)，无法进行手眼标定！")
        print("\n请检查：")
        print("  1. 重投影误差阈值是否过于严格")
        print("  2. 图像质量和标定板检测是否准确")
        print("  3. 相机内参和畸变系数是否准确")
        print("  4. 标定板是否清晰可见")