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
        """生成标定板上的3D点（Z=0平面）"""        # R = R.T
        # t = -R @ t
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
    从列表加载机械臂位姿数据
    参数:
        pose_list: 包含多个位姿的列表，每个位姿是 [x, y, z, rx, ry, rz]
    返回:
        R_end_to_base_list: 旋转矩阵列表
        t_end_to_base_list: 平移向量列表
    '''
    R_end_to_base_list = []
    t_end_to_base_list = []

    for i, pose in enumerate(pose_list):
        x, y, z = pose[0:3]
        rx, ry, rz = pose[3:6]
        t = np.array([[x], [y], [z]], dtype=np.float32)
        R = Rot.from_euler("ZYX", [rz, ry, rx], degrees=True).as_matrix()
        R = R.T
        t = -R @ t
        t_end_to_base_list.append(t)
        R_end_to_base_list.append(R)

    return R_end_to_base_list, t_end_to_base_list


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
    else:
        cv2.imshow('Reprojection Visualization', vis_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # ========== 1. 设置相机内参 ==========
    camera_matrix = np.array([[610.031, 0,       642.691], 
                              [0,       609.985, 369.464], 
                              [0,       0,       1]])

    # 畸变系数
    dist_coeffs = np.array([-0.0229736, 0.0262991, 0.000405298, 8.76571e-05, -0.00801325])

    # ========== 2. 设置标定板参数 ==========
    pattern_size = (11, 8)  # 棋盘格内角点数（宽，高）
    square_size = 25.0      # 格子边长，单位 mm

    # ========== 3. 设置重投影误差阈值 ==========
    # 根据实际需求调整，通常1-2像素是较好的结果
    MAX_REPROJECTION_ERROR = 2.0  # 最大允许的平均重投影误差（像素）
    MAX_STD_ERROR = 1.0           # 最大允许的标准差（像素）
    MAX_POINT_ERROR = 5.0         # 最大允许的单个点误差（像素）

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
    
    # ========== 8. 执行手眼标定（仅使用有效图像） ==========
    if len(valid_images) >= 3:  # 至少需要3对点进行手眼标定
        print("\n开始手眼标定...")
        
        # end_pose = [
        #     [190.122973, -535.226942, 125.527981, 156.745490, -78.082289, -58.330155],
        #     [382.647933, -505.202421, 109.582415, 156.735788, -78.079192,-58.321151],
        #     [-138.192211, -586.441301, 152.706986, 156.750615, -78.079168, -58.334033],
        #     [173.154770, -561.710331, 249.060161, 156.736607, -78.077408, -58.320798],
        #     [-241.101406, -624.874170, 274.989218, 31.697928, -81.079001, 67.479481],
        #     [-241.090721, -624.771663, 274.989502, -98.042803, -77.752799, -173.243464],
        #     [-178.881840, -620.451700, 89.855097, -111.937743, -77.755310, -173.242765],
        #     [517.766611, -683.821848, 296.631774, 175.563321, -72.177883, -88.355032],
        #     [517.962932, -683.959329, 296.557395, 140.975430, -87.892007, -22.955146],
        #     [624.244556, -663.016697, -214.082843, 83.739578, -52.017316, 11.890384],
        #     [609.741950, -543.948602, -222.133760, 83.737590, -52.018487, 11.891428],
        #     [556.347820, -541.608249, -110.343083, 46.667443, -84.598241, 39.580658],
        #     [241.490457, -681.527467, 52.832948, 90.282223, -82.979847, 1.512484],
        #     [649.134591, -668.871301, 2.720495, 20.305316, -69.387702, 72.694714],
        #     [614.600482, -670.287804, 7.056826, 129.509660, -80.473203, -37.201948],
        #     [179.362731, -667.770836, -154.277406, 129.523395, -80.476798, -37.217532],
        #     [243.350933, -716.383753, 324.907600, 129.521907, -80.478570, -37.216666],
        #     [-264.664781, -715.704717, 157.749507, 129.546048, -80.475841, -37.239403],
        #     [-192.711162, -803.551658, 358.433436, 162.276920, -65.221876, -71.119305],
        #     [759.855292, -760.850117, 254.856736, 113.630394, -63.457045, -20.020562],
        #     [618.833195, -724.337047, -227.073824, 60.218405, -77.586287, 60.071835],
        #     [729.414524, -751.953522, -259.624388, 94.534632, -74.024000, -18.74497],
        #     [-511.114743, -964.720172, -200.665125, -131.762140, -64.909608, -132.747146],
        #     # [-483.348309, -1086.672777, 25.140930, 171.515496, -76.487026, -72.473048]
        # ]

        # end_pose = [[190.122973, -535.226942, 125.52798100000001, 175.23664527767852, 100.93703213685207, -82.04094273109011], [382.647933, -505.202421, 109.582415, 175.2335081010941, 100.93906167632164, -82.04202682355283], [-138.192211, -586.441301, 152.706986, 175.2363542482543, 100.9403153333502, -82.03986145349934], [173.15477, -561.710331, 249.060161, 175.2329349161182, 100.9407630202808, -82.04098126743169], [-241.101406, -624.87417, 274.989218, 175.28496501941603, 82.41826944588558, -80.50999809848243], [-241.090721, -624.771663, 274.989502, -167.86968015382115, 91.7007714426626, -91.10553978442192], [-178.88184, -620.4517, 89.855097, -168.61793764397757, 94.54462417795442, -104.72737743837423], [517.766611, -683.821848, 296.631774, 178.57536747702756, 107.7669286182391, -93.01439247652971], [517.962932, -683.959329, 296.557395, 178.67233310249702, 91.63750269260768, -61.998690376506175], [624.244556, -663.016697, -214.082843, 142.1834321524228, 86.15196006813268, -83.05150037162332], [609.74195, -543.948602, -222.13376, 142.18470708457892, 86.15084206511818, -83.05210889306221], [556.34782, -541.608249, -110.343083, 176.0653665841664, 86.2960508794254, -93.6246253537059], [241.490457, -681.527467, 52.832948, 172.97993131407497, 90.03449272029384, -88.20740872866594], [649.134591, -668.871301, 2.720495, 172.56366376508936, 70.72117993861747, -85.73522135076551], [614.600482, -670.287804, 7.056826, 172.6224621689483, 96.04434265271978, -88.08232864743373], [179.362731, -667.770836, -154.277406, 172.62670885285237, 96.04383724021704, -88.08391985589721], [243.350933, -716.383753, 324.9076, 172.62793481870304, 96.04252794803462, -88.0843922858943], [-264.664781, -715.704717, 157.749507, 172.62834181729374, 96.04734819269821, -88.08327809181961], [-192.711162, -803.551658, 358.433436, 172.00121080570483, 113.52906743763815, -90.51084538610164], [759.855292, -760.850117, 254.85673600000004, 155.409577918861, 100.31854278429007, -88.64487777153815], [618.833195, -724.337047, -227.073824, 169.1843401887349, 83.87061004919984, -59.12896790425461], [729.414524, -751.953522, -259.624388, 164.07146839775461, 91.24688384437322, -104.38478990006479], [-511.1147429999999, -964.720172, -200.665125, -160.7477244045957, 106.40543901604622, -81.70820595636268]]
        end_pose = [[190.122973, -535.226942, 125.52798100000001, 85.23664527767849, 100.93703213685207, 7.9590572689099], [382.647933, -505.202421, 109.582415, 85.23350810109407, 100.93906167632164, 7.957973176447163], [-138.192211, -586.441301, 152.706986, 85.23635424825427, 100.9403153333502, 7.960138546500684], [173.15477, -561.710331, 249.060161, 85.23293491611821, 100.9407630202808, 7.95901873256831], [-241.101406, -624.87417, 274.989218, 85.28496501941603, 82.41826944588558, 9.490001901517575], [-241.090721, -624.771663, 274.989502, 102.13031984617886, 91.7007714426626, -1.1055397844219073], [-178.88184, -620.4517, 89.855097, 101.3820623560224, 94.54462417795442, -14.727377438374234], [517.766611, -683.821848, 296.631774, 88.57536747702754, 107.7669286182391, -3.014392476529713], [517.962932, -683.959329, 296.557395, 88.672333102497, 91.63750269260768, 28.00130962349382], [624.244556, -663.016697, -214.082843, 52.18343215242279, 86.15196006813268, 6.948499628376693], [609.74195, -543.948602, -222.13376, 52.184707084578896, 86.15084206511818, 6.947891106937784], [556.34782, -541.608249, -110.343083, 86.06536658416641, 86.2960508794254, -3.6246253537059077], [241.490457, -681.527467, 52.832948, 82.97993131407497, 90.03449272029384, 1.7925912713340488], [649.134591, -668.871301, 2.720495, 82.56366376508937, 70.72117993861747, 4.264778649234479], [614.600482, -670.287804, 7.056826, 82.62246216894827, 96.04434265271978, 1.917671352566271], [179.362731, -667.770836, -154.277406, 82.62670885285237, 96.04383724021704, 1.9160801441027941], [243.350933, -716.383753, 324.9076, 82.62793481870301, 96.04252794803462, 1.9156077141057104], [-264.664781, -715.704717, 157.749507, 82.62834181729372, 96.04734819269821, 1.9167219081803992], [-192.711162, -803.551658, 358.433436, 82.00121080570484, 113.52906743763815, -0.5108453861016179], [759.855292, -760.850117, 254.85673600000004, 65.409577918861, 100.31854278429007, 1.3551222284618674], [618.833195, -724.337047, -227.073824, 79.1843401887349, 83.87061004919984, 30.8710320957454], [729.414524, -751.953522, -259.624388, 74.07146839775463, 91.24688384437322, -14.384789900064801], [-511.1147429999999, -964.720172, -200.665125, 109.2522755954043, 106.40543901604622, 8.291794043637324]]
        # 只使用有效图像对应的机械臂位姿
        valid_end_pose = [end_pose[i] for i in range(len(valid_images))]
        
        R_end_to_base_list, t_end_to_base_list = load_robot_poses_from_list(valid_end_pose)
        
        try:
            R_cam_to_base, t_cam_to_base = cv2.calibrateHandEye(
                R_gripper2base=R_end_to_base_list,
                t_gripper2base=t_end_to_base_list,
                R_target2cam=R_target2cam_list,
                t_target2cam=t_target2cam_list,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )
            
            print("\n手眼标定结果：")
            print("旋转矩阵 R_cam_to_base:")
            print(R_cam_to_base)
            print("\n平移向量 t_cam_to_base (mm):")
            print(t_cam_to_base)
            
            # 将旋转矩阵转换为欧拉角（便于理解）
            r = Rot.from_matrix(R_cam_to_base)
            euler_angles = r.as_euler('xyz', degrees=True)
            print(f"\n欧拉角 (xyz, 度): {euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}")
            
        except Exception as e:
            print(f"\n❌ 手眼标定失败: {e}")
    else:
        print(f"\n❌ 有效图像数量不足 ({len(valid_images)}/3)，无法进行手眼标定！")
        print("请检查：")
        print("  1. 重投影误差阈值是否过于严格")
        print("  2. 图像质量和标定板检测是否准确")
        print("  3. 相机内参和畸变系数是否准确")
