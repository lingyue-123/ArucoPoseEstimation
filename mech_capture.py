#!/usr/bin/env python3
"""
Mech-Eye 相机视频流采集工具
支持：实时拉流显示、按键保存图像、按键启停视频录制
"""

import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np

# 确保 robovision 包可以被导入（保留原有路径逻辑）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Mech-Eye SDK 2.5.x 路径配置
_MECHEYE_SYSTEM_PATH = '/usr/local/lib/python3.8/dist-packages'
if _MECHEYE_SYSTEM_PATH not in sys.path:
    sys.path.append(_MECHEYE_SYSTEM_PATH)

# 导入 Mech-Eye 核心接口
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ARM64 优化
cv2.setNumThreads(1)
cv2.setUseOptimized(True)


class MechEyeRealTimeCamera:
    """简化版 Mech-Eye 实时相机类（仅保留视频流相关功能）"""
    def __init__(self):
        self.camera = Camera()
        self.is_open = False
        self.frame_width = None
        self.frame_height = None

    def open(self):
        """连接并打开 Mech-Eye 相机"""
        if find_and_connect(self.camera):
            self.is_open = True
            logger.info("Mech-Eye 相机连接成功")
            self._get_camera_resolution()
        else:
            raise RuntimeError("Mech-Eye 相机连接失败")

    def close(self):
        """断开相机连接"""
        if self.is_open:
            self.camera.disconnect()
            self.is_open = False
            logger.info("Mech-Eye 相机已断开连接")

    def _get_camera_resolution(self):
        """获取相机分辨率"""
        frame_2d = Frame2D()
        error = self.camera.capture_2d(frame_2d)
        if error.is_ok():
            sz = frame_2d.image_size()
            self.frame_width = sz.width
            self.frame_height = sz.height
            logger.info(f"Mech-Eye 相机分辨率: {self.frame_width}x{self.frame_height}")
        else:
            show_error(error)
            raise RuntimeError("无法获取 Mech-Eye 相机分辨率")

    def read_frame(self):
        """读取单帧图像"""
        if not self.is_open:
            return False, None
        
        frame_2d = Frame2D()
        error = self.camera.capture_2d(frame_2d)
        if not error.is_ok():
            show_error(error)
            return False, None
        
        # 处理图像格式转换
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image_data = frame_2d.get_gray_scale_image().data()
            # 单通道转三通道（方便显示和录制）
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image_data = cv2.cvtColor(frame_2d.get_color_image().data(), cv2.COLOR_RGB2BGR)
        else:
            logger.error("未知的 Mech-Eye 图像格式")
            return False, None
        
        return True, image_data


class VideoRecorder:
    """视频录制器"""
    def __init__(self, output_dir, fps=30):
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.is_recording = False
        self.recorded_frames = 0
        self.save_path = None

    def start_recording(self, frame_size):
        """开始录制视频"""
        if self.is_recording:
            logger.warning("已在录制中，无需重复开始")
            return
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成带时间戳的视频文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.save_path,
            fourcc,
            self.fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            logger.error("无法初始化视频写入器")
            self.writer = None
            return
        
        self.is_recording = True
        self.recorded_frames = 0
        logger.info(f"开始录制视频: {self.save_path} (FPS: {self.fps})")

    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            logger.warning("未在录制中，无需停止")
            return
        
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        logger.info(f"停止录制视频: {self.save_path} | 总帧数: {self.recorded_frames}")

    def write_frame(self, frame):
        """写入单帧到视频文件"""
        if self.is_recording and self.writer:
            self.writer.write(frame)
            self.recorded_frames += 1


class ImageSaver:
    """图像保存器"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.count = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, frame):
        """保存单帧图像"""
        self.count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.output_dir,
            f"image_{timestamp}_{self.count:04d}.jpg"
        )
        cv2.imwrite(save_path, frame)
        logger.info(f"保存图像: {save_path}")
        return save_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Mech-Eye 相机视频流采集工具')
    parser.add_argument('--output-dir', type=str, default='data/mecheye_capture',
                        help='输出目录（保存图像和视频）')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频录制帧率（默认30）')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 初始化核心组件
    camera = MechEyeRealTimeCamera()
    image_saver = ImageSaver(os.path.join(args.output_dir, 'images'))
    video_recorder = VideoRecorder(os.path.join(args.output_dir, 'videos'), fps=args.fps)

    try:
        # 打开相机
        camera.open()

        # 初始化显示窗口
        win_name = "Mech-Eye 视频流 [s=保存图像 | r=开始/停止录制 | q=退出]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 1024)

        logger.info("=" * 60)
        logger.info("操作说明:")
        logger.info("  s 键: 保存当前帧为图像文件")
        logger.info("  r 键: 切换视频录制状态（开始/停止）")
        logger.info("  q 键/Esc: 退出程序")
        logger.info("=" * 60)

        # 主循环
        while True:
            # 读取相机帧
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                logger.warning("读取相机帧失败，重试...")
                time.sleep(0.01)
                key = cv2.waitKey(10) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            # 绘制录制状态提示
            display_frame = frame.copy()
            if video_recorder.is_recording:
                # 红色录制提示
                cv2.putText(
                    display_frame,
                    "RECORDING",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    display_frame,
                    f"Frames: {video_recorder.recorded_frames}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # 显示画面
            cv2.imshow(win_name, cv2.resize(display_frame, (1280, 1024), interpolation=cv2.INTER_LINEAR))

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                # 退出程序
                break
            elif key == ord('s'):
                # 保存图像
                image_saver.save(frame)
            elif key == ord('r'):
                # 切换录制状态
                if not video_recorder.is_recording:
                    # 开始录制
                    video_recorder.start_recording((camera.frame_width, camera.frame_height))
                else:
                    # 停止录制
                    video_recorder.stop_recording()

            # 如果正在录制，写入当前帧
            if video_recorder.is_recording:
                video_recorder.write_frame(frame)

    except Exception as e:
        logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 资源清理
        cv2.destroyAllWindows()
        
        # 确保停止录制
        if video_recorder.is_recording:
            video_recorder.stop_recording()
        
        # 关闭相机
        camera.close()
        
        logger.info("程序正常退出")


if __name__ == '__main__':
    main()
