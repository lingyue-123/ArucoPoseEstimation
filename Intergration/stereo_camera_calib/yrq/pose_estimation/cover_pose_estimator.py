import numpy as np
import pyorbbecsdk as ob
from Intergration.stereo_camera_calib.yrq.pose_estimation.run_onnx import YOLOv5, filter_box
from Intergration.stereo_camera_calib.yrq.pose_estimation.pix_2_base_new import get_3D_pose_one, get_3D_pose_rpy
from Intergration.stereo_camera_calib.yrq.pose_estimation.utils import frame_to_bgr_image, get_inner_rectangle
from Intergration.stereo_camera_calib.yrq.pose_estimation.coutour_detect import detect_charging_port_cover_simple
from datetime import datetime
import cv2
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--x_off", type = float, default=105, help="x方向补偿")
    # parser.add_argument("--y_off", type = float, default=-70, help="y方向补偿")
    # parser.add_argument("--z_off", type = float, default=-315.72, help="z方向补偿")
    parser.add_argument("--x_off", type = float, default=105.681, help="x方向补偿")
    parser.add_argument("--y_off", type = float, default=-69.813, help="y方向补偿")
    parser.add_argument("--z_off", type = float, default=-326.996, help="z方向补偿")
    parser.add_argument("--rx_off", type = float, default=-0.199, help="rx方向补偿")
    parser.add_argument("--ry_off", type = float, default=-6.464, help="ry方向补偿")
    parser.add_argument("--rz_off", type = float, default=0.273, help="rz方向补偿")
    return parser.parse_args()

class CoverPoseEstimator:
    def __init__(self):
        self.onnx_path = "/home/nvidia/Downloads/HD/HD_0323/Intergration/stereo_camera_calib/yrq/pose_estimation/cover_depth_best.onnx"
        self.pipeline = None
        self.config = None
        self.model = None

        # 1. 模型初始化
        self.model_init()
        # 2. 相机初始化
        self.camera_init()

    def model_init(self):
        # 模型实例化
        self.model = YOLOv5(self.onnx_path)
        print(datetime.now().strftime("%H:%M:%S.%f"), "充电口盖检测模型初始化成功")

    def camera_init(self):
        # 1. 创建Pipeline
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()

        # 2. 配置彩色图像流
        color_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_video_stream_profile(1280, 800, ob.OBFormat.MJPG, 30)
        self.config.enable_stream(color_profile)
        print(f"已配置彩色流: {color_profile.get_width()}x{color_profile.get_height()}, "
              f"格式: {color_profile.get_format()}, {color_profile.get_fps()}fps")
        
        # 3. 配置深度图像流
        depth_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_video_stream_profile(1280, 800, ob.OBFormat.Y16, 30)
        self.config.enable_stream(depth_profile)
        print(f"已配置深度流: {depth_profile.get_width()}x{depth_profile.get_height()}, "
              f"格式: {depth_profile.get_format()}, {depth_profile.get_fps()}fps")
        
        # 4. 设置对齐模式
        self.config.set_align_mode(ob.OBAlignMode.SW_MODE)
        print("对齐模式设置为软件对齐")
        print(datetime.now().strftime("%H:%M:%S.%f"), "双目相机初始化配置成功")
    
    def pose_estimation(self):
        """双目相机拉流并估计充电口盖位姿"""
        args = args_parse()
        try:
            # 1. 启动流水线
            self.pipeline.start(self.config)
            print(datetime.now().strftime("%H:%M:%S.%f"), "双目相机启动成功，开始采集数据...")
            while True:
                # 2. 获取彩色图像和深度图像
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    print("双目相机等待帧超时")
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame is not None:
                    color_frame = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                    color_image = cv2.imdecode(color_frame, cv2.IMREAD_COLOR)

                if depth_frame is not None:
                    depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_image = depth_image.reshape((depth_frame.get_height(), depth_frame.get_width()))

                # 3. 执行轮廓检测
                if (color_frame is not None) and (depth_frame is not None):
                    height, width = color_image.shape[:2]
                    _left, _right, _top, _bottom = detect_charging_port_cover_simple(color_image)

                    if _left== width:
                        _left -= 1
                    if _right== width:
                        _right -= 1

                    pts_pixel = [(_left, _top), (_left, _bottom), (_right, _top), (_right, _bottom)]
                    cv2.rectangle(color_image, pts_pixel[0], pts_pixel[-1], (255, 255, 0), 2)

                    center_x = (_left + _right) / 2 
                    center_y = (_top + _bottom) / 2 

                    depth_center = self.get_depth(depth_image, center_x, center_y)
                    cv2.circle(color_image, (int(center_x), int(center_y)), radius=5, color=(0, 0, 255), thickness=-1)
                    print(f"[充电口盖中心深度值]: {depth_center}mm")
                    cover_pose_t = get_3D_pose_one(center_x, center_y, depth_center)

                    cover_pose = [0, 0, 0, args.rx_off, args.ry_off, args.rz_off]
                    if depth_center > 0:
                        cover_pose[:3] = [cover_pose_t[0][0], cover_pose_t[1][0], cover_pose_t[2][0]]
                        cover_pose[0] += args.x_off
                        cover_pose[1] += args.y_off
                        cover_pose[2] += args.z_off

                        # 无力控补偿
                        # cover_pose[0] += 67.998
                        # cover_pose[1] += -69.096
                        # cover_pose[2] += -336.030
                        print(f"双目相机估计[有效]")
                        return cover_pose
                    else:
                        print(f"双目相机估计[无效]")
                        return
                #     cv2.imshow("Color", color_image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        finally:
            # 释放所有资源
            self.release_resources()

    def pose_estimation_yolo(self):
        """双目相机拉流并估计充电口盖位姿"""
        args = args_parse()
        cover_pose = None
        try:
            # 1. 启动流水线
            self.pipeline.start(self.config)
            print(datetime.now().strftime("%H:%M:%S.%f"), "双目相机启动成功，开始采集数据...")
            cover_poses = []
            frame_counts = 0
            while True:
                # 2. 获取彩色图像和深度图像
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    print("双目相机等待帧超时")
                    continue
                # 3s左右未检测到充电口盖，返回
                # frame_counts += 1
                # if cover_poses == [] and frame_counts == 30:
                #     self.release_resources()
                #     return 

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame is not None:
                    # color_image = frame_to_bgr_image(color_frame)
                    color_frame = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                    color_image = cv2.imdecode(color_frame, cv2.IMREAD_COLOR)

                if depth_frame is not None:
                    depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_image = depth_image.reshape((depth_frame.get_height(), depth_frame.get_width()))

                # 3. 执行模型推理
                if (color_frame is not None) and (depth_frame is not None):
                    # print({datetime.now().strftime("%H:%M:%S.%f")}, "执行充电口盖检测模型推理")
                    output, _ = self.model.inference(color_image)
                    # print({datetime.now().strftime("%H:%M:%S.%f")}, "完成充电口盖检测模型推理")
                    
                    outbox = filter_box(output, 0.6, 0.5)
                    
                    height, width = color_image.shape[:2]
                    scale_h = height / 640
                    scale_w = width / 640

                    if outbox.shape[0] == 1:
                        outbox = np.squeeze(outbox)
                        left, top, right, bottom = outbox[:4].astype(np.int32)

                        _left, _right, _top, _bottom = detect_charging_port_cover_simple(color_image)

                        # _top = int(top * scale_h)
                        # _bottom = int(bottom * scale_h)
                        # _left = int(left * scale_w)
                        # _right = int(right * scale_w)
                        if _left== width:
                            _left -= 1
                        if _right== width:
                            _right -= 1

                        pts_pixel = [(_left, _top), (_left, _bottom), (_right, _top), (_right, _bottom)]
                        pts_pixel_m = get_inner_rectangle(pts_pixel, 0.4)
                        pts_pixel_s = get_inner_rectangle(pts_pixel, 0.2)
                        
                        cv2.rectangle(color_image, pts_pixel[0], pts_pixel[-1], (255, 255, 0), 2)
                        # cv2.rectangle(color_image, pts_pixel_m[0], pts_pixel_m[-1], (0, 255, 0), 1)
                        # cv2.rectangle(color_image, pts_pixel_s[0], pts_pixel_s[-1], (0, 0, 255), 1)

                        # center_x = (_left + _right) / 2 - 110
                        # center_y = (_top + _bottom) / 2 - 20
                        center_x = (_left + _right) / 2 
                        center_y = (_top + _bottom) / 2 
                        depth_center = self.get_depth(depth_image, center_x, center_y)
                        cv2.circle(color_image, (int(center_x), int(center_y)), radius=5, color=(0, 0, 255), thickness=-1)
                        print(f"[充电口盖中心深度值]: {depth_center}mm")
                        cover_pose_t = get_3D_pose_one(center_x, center_y, depth_center)

                        pts_pixel_all = pts_pixel_m + pts_pixel_s
                        pts_pixel_all.append((center_x, center_y))
                        depths = []
                        pts_pixels = []
                        for i in range(len(pts_pixel_all)):
                            depth = self.get_depth(depth_image, pts_pixel_all[i][0], pts_pixel_all[i][1])
                            if depth > 0 :
                                pts_pixels.append(pts_pixel_all[i])
                                depths.append(depth)
                        # print(datetime.now().strftime("%H:%M:%S.%f"), "完成充电口盖位姿估计")
                        # print(f"[深度值]: {depth}mm")

                        cover_pose = [0, 0, 0, 0, 0, 0]
                        if depth_center > 0:
                            cover_pose[:3] = [cover_pose_t[0][0], cover_pose_t[1][0], cover_pose_t[2][0]]
                            # cover_pose[0] += args.x_off
                            # cover_pose[1] += args.y_off
                            # cover_pose[2] += args.z_off
                            print(f"双目相机估计[有效]")
                            return cover_pose
                        else:
                            print(f"双目相机估计[无效]")
                            return

                            # try:
                            #     cover_pose_rpy = get_3D_pose_rpy(pts_pixels, depths)
                            #     cover_pose[:3] = [cover_pose_t[0][0], cover_pose_t[1][0], cover_pose_t[2][0]]
                            #     cover_pose[3:] = cover_pose_rpy[:]
                            #     cover_pose[-1] = 0
                            #     # cover_poses.append(cover_pose)
                            #     # print(f"双目相机有效估计次数： {len(cover_poses)}")
                            #     cover_pose[0] += 105
                            #     cover_pose[1] -= 70
                            #     cover_pose[2] -= 315.72
                            #     print(f"双目相机有效估计")
                            #     # print(f"cover_pose: {cover_pose}")
                            #     return cover_pose
                            # except:
                            #     print(f"无法计算rpy角")
                            #     continue
                        # if len(cover_poses) == 20:
                        #     final_pose = np.mean(np.array(cover_poses), axis=0).tolist()
                        #     self.release_resources()
                        #     return final_pose
                #     else:
                #         print(f"未检测到充电口盖")
                #     cv2.imshow("Color", color_image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        finally:
            # 释放所有资源
            self.release_resources()
            

    def get_depth(self,depth_img, x, y):
        """从深度图数组中获取 (x, y) 处的深度值"""
        h, w = depth_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"坐标 ({x}, {y}) 超出图像范围 (宽{w}, 高{h})")
        return depth_img[int(y), int(x)]  # 注意: numpy索引为 [行, 列]
    
    def release_resources(self):
        """释放所有资源"""
        # print(datetime.now().strftime("%H:%M:%S.%f"), "开始释放资源...")
        
        # 1. 停止并关闭相机 pipeline
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"关闭相机 pipeline 时出错: {e}")
        # print(datetime.now().strftime("%H:%M:%S.%f"), "双目相机资源释放完成")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.release_resources()

if __name__ == "__main__":
    cover_pose_estimator = CoverPoseEstimator()
    cover_pose = cover_pose_estimator.pose_estimation()
    # print(f"cover pose: {cover_pose}")