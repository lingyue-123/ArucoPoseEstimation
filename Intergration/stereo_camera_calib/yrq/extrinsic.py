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

    def detect_board_pose(self, image):
        '''
        从单张图像中检测标定板并计算其位姿
        参数:
            image: 输入图像 (BGR格式)
        返回:
            success: 是否成功检测
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3x1)
            corners: 检测到的角点坐标
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        if not ret:
            return False, None, None, None

        # 亚像素精细化角点坐标
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)

        # 使用solvePnP求解标定板位姿
        # 注意：solvePnP 输出的 rvec, tvec 是标定板→相机的变换
        ret, rvec, tvec = cv2.solvePnP(self.object_points, corners2, self.camera_matrix, self.dist_coeffs)
        if not ret:
            return False, None, None, None

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        return True, R, tvec, corners2


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
        R = Rot.from_euler("yzx", [rx, ry, rz], degrees=True).as_matrix()

        # R = R.T
        # t = -R @ t
        t_end_to_base_list.append(t)
        R_end_to_base_list.append(R)

    return R_end_to_base_list, t_end_to_base_list


if __name__ == "__main__":
    # ========== 1. 设置相机内参 ==========
    # camera_matrix = np.array([[611.696,	0,	643.269],
    #                           [0, 611.671, 408.698],
    #                           [0,	0,	1]])

    # # 畸变系数
    # dist_coeffs = np.array([-0.0231334, 0.0283417, 0.00021241, 0.000386759, -0.0102324])
    camera_matrix = np.array([[610.031, 0,       642.691], 
                              [0,       609.985, 369.464], 
                              [0,       0,       1]])

    # 畸变系数
    dist_coeffs = np.array([-0.0229736, 0.0262991, 0.000405298, 8.76571e-05, -0.00801325])


    # ========== 2. 设置标定板参数 ==========
    pattern_size = (11, 8)  # 棋盘格内角点数（宽，高）
    square_size = 25.0      # 格子边长，单位 mm

    # ========== 3. 设置图像路径 ==========
    image_dir = "./ex_417"  # 存放标定图像的文件夹
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    # 收集所有图像文件
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = natsorted(image_paths)
    print(f"找到 {len(image_paths)} 张图像")

    # ========== 4. 创建位姿估计器 ==========
    pose_estimator = CalibrationBoardPoseEstimator(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size,
        square_size=square_size)

    # ========== 5. 存储所有图像的位姿结果 ==========
    R_target2cam_list = []   # 旋转矩阵列表
    t_target2cam_list = []   # 平移向量列表
    valid_images = []        # 成功检测的图像路径

    for i, img_path in enumerate(image_paths):
        print(f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告：无法读取图像，跳过")
            continue

        # 检测标定板位姿
        success, R, t, corners = pose_estimator.detect_board_pose(img)

        if success:
            # 存储结果
            R_target2cam_list.append(R)
            t_target2cam_list.append(t)
            valid_images.append(img_path)

    # ========== 6. 输出结果统计 ==========
    print(" " + "=" * 50)
    print(f"处理完成！")
    print(f"成功检测: {len(valid_images)}/{len(image_paths)} 张图像")

    end_pose = [
        [190.122973, -535.226942, 125.527981, 156.745490, -78.082289, -58.330155],
        [382.647933, -505.202421, 109.582415, 156.735788, -78.079192,-58.321151],
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

    R_end_to_base_list, t_end_to_base_list = load_robot_poses_from_list(end_pose)

    R_cam_to_base, t_cam_to_base = cv2.calibrateHandEye(
        R_gripper2base=R_end_to_base_list,
        t_gripper2base=t_end_to_base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    print(R_cam_to_base)
    print(t_cam_to_base)