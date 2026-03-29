#!/usr/bin/env python3
"""
内参标定验证入口脚本 | Verify Intrinsic Calibration

从 cameras.yaml 读取内参，支持两种验证模式：
  A. 离线图片文件夹（--image-dir）：遍历所有图片，自动统计重投影误差并输出结果。
  B. 实时相机（不提供 --image-dir）：打开相机实时预览，s=采集样本，q=退出后统计输出。

变更记录：
  v0.2.1（2026-03-03）：删除冗余的 intrinsic.txt 读取（--intrinsic / --intrinsic-index），
                         新增 --image-dir 离线验证模式。

用法示例：
    # 离线验证
    python scripts/verify_intrinsic.py \\
        --camera hikvision_normal \\
        --image-dir intrinsic_calib/data800/

    # 实时验证
    python scripts/verify_intrinsic.py --camera hikvision_normal
"""

import argparse
import glob
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from robovision.cameras import CameraIntrinsics, build_camera
from robovision.config.loader import get_config
from robovision.detection.chessboard import ChessboardDetector

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

_IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp')


def parse_args():
    parser = argparse.ArgumentParser(description='内参标定验证')
    parser.add_argument('--camera', type=str, default=None,
                        help='cameras.yaml 中的相机名称')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='离线图片文件夹路径（支持 png/jpg/bmp）；不提供则使用实时相机')
    return parser.parse_args()


def _collect_images(image_dir: str):
    """收集文件夹内所有图片路径，按文件名排序。"""
    paths = []
    for ext in _IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    return paths


def _print_summary(errors):
    if not errors:
        logger.warning("未采集到有效样本，无法统计。")
        return
    arr = np.array(errors)
    print("\n========== 内参验证结果 ==========")
    print(f"样本数:           {len(arr)}")
    print(f"均值重投影误差:   {arr.mean():.4f} px")
    print(f"标准差:           {arr.std():.4f} px")
    print(f"最大值:           {arr.max():.4f} px")
    print(f"最小值:           {arr.min():.4f} px")


def run_offline(intrinsics: CameraIntrinsics, detector: ChessboardDetector, image_dir: str):
    """模式 A：遍历离线图片文件夹统计重投影误差。"""
    paths = _collect_images(image_dir)
    if not paths:
        logger.error("文件夹 '%s' 中未找到图片（png/jpg/bmp）", image_dir)
        sys.exit(1)
    logger.info("共找到 %d 张图片，开始离线验证...", len(paths))

    errors = []
    for i, p in enumerate(paths, 1):
        frame = cv2.imread(p)
        if frame is None:
            logger.warning("[%d/%d] 无法读取: %s", i, len(paths), p)
            continue
        result = detector.detect(frame)
        if result is None:
            logger.info("[%d/%d] 未检测到棋盘格: %s", i, len(paths), os.path.basename(p))
        else:
            err = result['reproj_err']
            errors.append(err)
            logger.info("[%d/%d] reproj=%.2f px  %s", i, len(paths), err, os.path.basename(p))

    _print_summary(errors)


def run_live(intrinsics: CameraIntrinsics, detector: ChessboardDetector, camera_name: str, cfg):
    """模式 B：实时相机预览，s=采集，q=退出。"""
    camera = build_camera(camera_name, cfg)
    camera.open()

    errors = []
    win = "内参验证"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    logger.info("就绪。s=采集误差样本  q=退出")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            result = detector.detect(frame)
            vis = detector.draw_result(frame, result)

            if result:
                cv2.putText(vis, f"reproj={result['reproj_err']:.2f}px",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            h = vis.shape[0]
            mean_err = float(np.mean(errors)) if errors else 0.0
            cv2.putText(vis,
                        f"样本={len(errors)} 均值reproj={mean_err:.2f}px | s=采集 q=退出",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            scale = 0.4
            disp = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s') and result:
                errors.append(result['reproj_err'])
                logger.info("[%d] reproj=%.2f px", len(errors), result['reproj_err'])
    finally:
        cv2.destroyAllWindows()
        camera.close()

    _print_summary(errors)


def main():
    args = parse_args()
    cfg = get_config()
    chess_cfg = cfg.get_chessboard()

    # 从 cameras.yaml 读取内参（两种模式均使用此来源）
    cam_cfg = cfg.get_camera(args.camera)
    intrinsics = CameraIntrinsics.from_config(cam_cfg.intrinsics, name=args.camera)
    logger.info("相机 '%s' 内参: fx=%.2f fy=%.2f",
                args.camera, intrinsics.camera_matrix[0, 0], intrinsics.camera_matrix[1, 1])

    detector = ChessboardDetector.from_config(intrinsics, chess_cfg)

    if args.image_dir:
        run_offline(intrinsics, detector, args.image_dir)
    else:
        run_live(intrinsics, detector, args.camera, cfg)


if __name__ == '__main__':
    main()
