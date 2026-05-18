import cv2
import numpy as np
import pyorbbecsdk as ob
from Intergration.stereo_camera_calib.yrq.pose_estimation.run_onnx import YOLOv5, filter_box
from Intergration.stereo_camera_calib.yrq.pose_estimation.pix_2_base_new import get_3D_pose_one, get_3D_pose
# from pix_2_base_415 import get_3D_pose
from KEBA.actuator.cover_action_main import CoverActionFlow


import argparse
import json
import fcntl
import os
import time
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_offset", type = float, default=0, help="水平补偿")
    parser.add_argument("--h_offset", type = float, default=0, help="高度补偿")
    parser.add_argument("--d_offset", type = float, default=0, help="深度补偿")
    parser.add_argument("--euler_offset", type = float, default=90.0, help="角度补偿")
    parser.add_argument("--scale", type = float, default=0.4, help="复选框缩放比例")
    return parser.parse_args()

def find_profile(profiles, width, height, format, fps):
    """在 profiles 中查找匹配的配置"""
    if profiles is None:
        return None
    for i in range(len(profiles)):
        p = profiles.get_stream_profile_by_index(i)
        if (p.get_width() == width and p.get_height() == height and
                p.get_format() == format and p.get_fps() == fps):
            return p
    return None

def get_depth(depth_img, x, y):
    """从深度图数组中获取 (x, y) 处的深度值"""
    h, w = depth_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"坐标 ({x}, {y}) 超出图像范围 (宽{w}, 高{h})")
    return depth_img[int(y), int(x)]  # 注意: numpy索引为 [行, 列]

def get_inner_rectangle(pts_pixel, scale=0.8):
    """
    获取原矩形内部的小矩形，中点不变，边长为原矩形的scale倍
    
    Args:
        pts_pixel: 原矩形的四个角点 [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
        scale: 缩放比例，默认0.8
    
    Returns:
        inner_pts: 内部矩形的四个角点
    """
    # 提取矩形的边界
    xs = [pt[0] for pt in pts_pixel]
    ys = [pt[1] for pt in pts_pixel]
    
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    
    # 计算矩形的中心点
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    
    # 计算原矩形的宽度和高度
    width = right - left
    height = bottom - top
    
    # 计算新矩形的宽度和高度（原来的80%）
    new_width = width * scale
    new_height = height * scale
    
    # 计算新矩形的边界
    new_left = int(center_x - new_width / 2)
    new_right = int(center_x + new_width / 2)
    new_top = int(center_y - new_height / 2)
    new_bottom = int(center_y + new_height / 2)
    
    # 返回新矩形的四个角点
    inner_pts = [(new_left, new_top), (new_left, new_bottom), 
                 (new_right, new_top), (new_right, new_bottom)]
    
    return inner_pts
def get_cover_pose():
    args = args_parse()
    # 模型实例化
    onnx_path = "/home/nvidia/Downloads/HD/HD_0323/Intergration/stereo_camera_calib/yrq/pose_estimation/cover_depth_best.onnx"
    model = YOLOv5(onnx_path)

    # 1. 创建Pipeline
    pipeline = ob.Pipeline()
    config = ob.Config()

    try:
        # ----- 彩色传感器配置 -----
        color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = find_profile(color_profiles, 1280, 800, ob.OBFormat.MJPG, 15)
        config.enable_stream(color_profile)
        print(f"已启用彩色流: {color_profile.get_width()}x{color_profile.get_height()}, "
              f"格式: {color_profile.get_format()}, {color_profile.get_fps()}fps")

        # ----- 深度传感器配置 -----
        depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        depth_profile = find_profile(depth_profiles, 1280, 800, ob.OBFormat.Y16, 15)
        config.enable_stream(depth_profile)
        print(f"已启用深度流: {depth_profile.get_width()}x{depth_profile.get_height()}, "
              f"格式: {depth_profile.get_format()}, {depth_profile.get_fps()}fps")


        # 设置对齐模式
        if hasattr(ob, 'OBAlignMode') and hasattr(ob.OBAlignMode, 'SW_MODE'):
            config.set_align_mode(ob.OBAlignMode.SW_MODE)
            print("对齐模式设置为软件对齐")

    except Exception as e:
        print(f"配置流时出错: {e}")
        return

    # 3. 启动流水线
    try:
        pipeline.start(config)
        print("相机启动成功，开始采集数据...")
    except Exception as e:
        print(f"启动失败: {e}")
        return

    # pose_start = 0
    # counts = 0
    # poses = []
    try:
        while True:

            # with open('/data/yrq/pose_start.json', 'r', encoding='utf-8') as f:
            #     fcntl.flock(f, fcntl.LOCK_SH) 
            #     pose_start = json.load(f)
            #     fcntl.flock(f, fcntl.LOCK_UN)

            # if pose_start == 1:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                print("等待帧超时")
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # 处理彩色帧
            if color_frame is not None:
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                format_type = color_frame.get_format()
                height = color_frame.get_height()
                width = color_frame.get_width()

                # 根据格式解码
                if format_type == ob.OBFormat.MJPG:
                    color_img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                elif format_type == ob.OBFormat.YUYV:
                    # YUYV 转 BGR
                    yuyv = color_data.reshape((height, width, 2))
                    color_img = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
                elif format_type == ob.OBFormat.RGB:
                    color_img = color_data.reshape((height, width, 3))
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                elif format_type == ob.OBFormat.BGR:
                    color_img = color_data.reshape((height, width, 3))
                elif format_type == ob.OBFormat.UYVY:
                    # UYVY 转 BGR
                    uyvy = color_data.reshape((height, width, 2))
                    color_img = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
                elif format_type == ob.OBFormat.NV12:
                    # NV12 转 BGR
                    nv12 = color_data.reshape((height * 3 // 2, width))
                    color_img = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                else:
                    print(f"未知彩色格式: {format_type}，无法显示")
                    color_img = None

            # 处理深度帧
            if depth_frame is not None:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_img = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                
            if (color_frame is not None) and (depth_frame is not None):
                
                output, _ = model.inference(color_img)
                outbox = filter_box(output, 0.5, 0.5)
                
                height, width = color_img.shape[:2]
                scale_h = height / 640
                scale_w = width / 640

                if outbox.shape[0] == 1:
                    outbox = np.squeeze(outbox)
                    left, top, right, bottom = outbox[:4].astype(np.int32)

                    _top = int(top * scale_h)
                    _bottom = int(bottom * scale_h)
                    _left = int(left * scale_w)
                    _right = int(right * scale_w)
                    if _left== width:
                        _left -= 1
                    if _right== width:
                        _right -= 1
                
                
                    pts_pixel = [(_left, _top), (_left, _bottom), (_right, _top), (_right, _bottom)]
                    # print(pts_pixel)
                    # pts_pixel = get_inner_rectangle(pts_pixel, args.scale)
                    center_x = (_left + _right) / 2
                    center_y = (_top + _bottom) / 2
                    # print(center_x)
                    depth = get_depth(depth_img, center_x, center_y)
                    cover_pose = get_3D_pose_one(center_x, center_y, depth, args.l_offset, args.d_offset, args.h_offset)



                    # cv2.rectangle(color_img, (_left, _top), (_right, _bottom), (255, 0, 0), 2)
                    # cv2.rectangle(color_img, pts_pixel[0], pts_pixel[-1], (0, 255, 0), 2)
            

                    # depths = []
                    # for i in range(4):
                    #     depths.append(get_depth(depth_img, pts_pixel[i][0], pts_pixel[i][1]))
                    # for i in range(4):
                    #     if depths[i] == 0 and i < 2:
                    #         depths[i] = depths[i+2]
                    #     if depths[i] == 0 and i > 1:
                    #         depths[i] = depths[i-2]

                    # print(pts_pixel)
                    print("========================================")
                    print("深度值")
                    print(depth)
                    # cover_pose = get_3D_pose(pts_pixel, depths, args.l_offset, args.h_offset, args.d_offset, args.euler_offset)
                    print("\n6D位姿")
                    res_pose = [cover_pose[0][0], cover_pose[1][0], cover_pose[2][0], -173.11256336193946, -68.93333123664308, -87.13262339367421]
                    print(res_pose)
                    if depth > 0:
                        # poses.append(res_pose)
                        # counts += 1
                        # with open('/data/yrq/cover_pose.json', 'w', encoding='utf-8') as f:
                        #     fcntl.flock(f, fcntl.LOCK_EX)
                        #     json.dump(res_pose, f)
                        #     f.flush()
                        #     os.fsync(f.fileno())
                        #     fcntl.flock(f, fcntl.LOCK_UN)
                        return res_pose
                        break
                    # if counts == 3:
                    #     pose_init = np.mean(np.array(poses), axis=0).tolist()
                    #     flow = CoverActionFlow()
                    #     time.sleep(2)
                    #     flow.run_open_cover(pose_init)
                    #     flow.run(2)
                    #     flow.motor.disconnect()
                    #     # flow.arm.disconnect()
                    #     break
                        # poses.append(res_pose)

                    # cover_pose[4] -= 5
                    # if cover_pose[3] > 0:
                    #     cover_pose[3] = -cover_pose[3]

                    # print("========================================\n")
                    # counts += 1
                    # if counts > 1:
                    #     if (cover_pose[0] * cover_pose[1] * cover_pose[2] * cover_pose[3] * cover_pose[4] *cover_pose[5] != 0): 
                    #         poses.append(cover_pose)

                    # if (counts == 5):
                    #     mean_pose = np.mean(np.array(poses), axis=0).tolist()
                    #     counts = 0
                        # pose_start = 0
                        # with open('/data/yrq/cover_pose.json', 'w', encoding='utf-8') as f:
                        #     fcntl.flock(f, fcntl.LOCK_EX)
                        #     # json.dump(np.mean(np.array(poses), axis=0).tolist(), f)
                        #     json.dump(res_pose, f)
                        #     f.flush()
                        #     os.fsync(f.fileno())
                        #     fcntl.flock(f, fcntl.LOCK_UN)

                        # with open('/data/yrq/pose_start.json', 'w', encoding='utf-8') as f:
                        #     fcntl.flock(f, fcntl.LOCK_EX)
                        #     json.dump(0, f)
                        #     f.flush()
                        #     os.fsync(f.fileno())
                        #     fcntl.flock(f, fcntl.LOCK_UN)
                        
                        # flow.run_open_cover(mean_pose)
                        # flow.run(2)
                        # break
                        # with open('/data/yrq/match_start.json', 'w', encoding='utf-8') as f:
                        #     fcntl.flock(f, fcntl.LOCK_EX)
                        #     json.dump(1, f)
                        #     f.flush()
                        #     os.fsync(f.fileno())
                        #     fcntl.flock(f, fcntl.LOCK_UN)
                # cv2.imshow("Color", color_img)
                        
            # else:
            #     time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("用户中断采集")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("采集结束，资源已释放")

if __name__ == "__main__":
    res = get_cover_pose()