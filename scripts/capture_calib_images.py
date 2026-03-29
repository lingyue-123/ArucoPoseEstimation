#!/usr/bin/env python3
"""
内参标定图像采集 | Capture Calibration Images

实时预览相机画面，按 s 保存当前帧，按 q 退出。

用法:
    python scripts/capture_calib_images.py --camera hikvision_20mp --output-dir intrinsic_calib/data_hkws_2000w_new/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from robovision.cameras import build_camera
from robovision.io.image_saver import ImageSaver


def main():
    parser = argparse.ArgumentParser(description="采集内参标定图像")
    parser.add_argument("--camera", type=str, default=None,
                        help="cameras.yaml 中的相机名称（默认使用 default_camera）")
    parser.add_argument("--output-dir", type=str, default="intrinsic_calib/calib_images",
                        help="图像保存目录")
    parser.add_argument("--preview-width", type=int, default=1280,
                        help="预览窗口宽度（像素）")
    args = parser.parse_args()

    cam = build_camera(args.camera)
    cam.open()
    saver = ImageSaver(args.output_dir)
    intrinsics = cam.get_intrinsics()

    print(f"相机已打开，分辨率: {intrinsics.width}x{intrinsics.height}")
    print(f"保存目录: {os.path.abspath(args.output_dir)}")
    print("按 s 保存, q 退出")

    win_name = "Calibration Capture"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, args.preview_width,
                     int(args.preview_width * intrinsics.height / intrinsics.width))

    frame = None
    try:
        while True:
            ok, new_frame = cam.read_frame()  # non-blocking (timeout=0)
            if ok and new_frame is not None:
                frame = new_frame
                scale = args.preview_width / frame.shape[1]
                display = cv2.resize(frame, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                cv2.imshow(win_name, display)

            key = cv2.waitKey(30) & 0xFF  # always pump GUI event loop

            if key == ord('s') and frame is not None:
                path = saver.save(frame)  # save full-res
                if path:
                    print(f"[{saver.count}] 已保存: {path}")
            elif key == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"共保存 {saver.count} 张图像")


if __name__ == "__main__":
    main()
