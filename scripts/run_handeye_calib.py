#!/usr/bin/env python3
"""
手眼标定计算入口脚本 | Run Hand-Eye Calibration

替代 Handeyecalib/hand_eye_calib.py 的 main() 函数。
从文件加载数据，执行手眼标定，保存结果。

用法：
    python scripts/run_handeye_calib.py
    python scripts/run_handeye_calib.py --full
    python scripts/run_handeye_calib.py --compare-methods --outlier-rejection --refine
    python scripts/run_handeye_calib.py --loo
    python scripts/run_handeye_calib.py --method HORAUD --outlier-rejection --max-error 3.0
"""

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from robovision.calibration.hand_eye import (
    run_calibration,
    print_results,
    print_loo_results,
    save_hand_eye_result,
    leave_one_out_analysis,
    HAND_EYE_METHODS,
    _prepare_input_lists,
)
from robovision.io.pose_file import load_pose_file

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='手眼标定计算')
    parser.add_argument('--board', type=str, default='data/handeye/board_to_cam.txt',
                        help='棋盘格位姿文件（board2cam，单位：毫米）')
    parser.add_argument('--robot', type=str, default='data/handeye/robot_tcp.txt',
                        help='机械臂位姿文件（end2base，单位：毫米）')
    parser.add_argument('--output', type=str, default='data/handeye/hand_eye_result.txt',
                        help='输出结果文件（4x4 矩阵）')
    parser.add_argument('--no-save', action='store_true', help='仅计算，不保存结果')

    # 优化选项
    parser.add_argument('--compare-methods', action='store_true',
                        help='尝试所有方法并选最优')
    parser.add_argument('--method', type=str, default=None,
                        choices=[k.lower() for k in HAND_EYE_METHODS],
                        help='指定标定方法')
    parser.add_argument('--outlier-rejection', action='store_true',
                        help='启用迭代异常点剔除')
    parser.add_argument('--max-error', type=float, default=5.0,
                        help='异常点剔除阈值（mm，默认 5.0）')
    parser.add_argument('--min-poses', type=int, default=8,
                        help='最少保留组数（默认 8）')
    parser.add_argument('--refine', action='store_true',
                        help='启用非线性优化（Levenberg-Marquardt）')
    parser.add_argument('--loo', action='store_true',
                        help='留一法分析')
    parser.add_argument('--full', action='store_true',
                        help='等效于 --compare-methods --outlier-rejection --refine')
    return parser.parse_args()


def main():
    args = parse_args()

    # --full 展开
    if args.full:
        args.compare_methods = True
        args.outlier_rejection = True
        args.refine = True

    logger.info("加载棋盘格位姿文件: %s", args.board)
    cam_data_arr = load_pose_file(args.board)
    logger.info("加载机械臂位姿文件: %s", args.robot)
    robot_data_arr = load_pose_file(args.robot)

    if len(cam_data_arr) != len(robot_data_arr):
        logger.error("位姿数量不匹配: board=%d, robot=%d",
                     len(cam_data_arr), len(robot_data_arr))
        sys.exit(1)

    logger.info("共 %d 组标定数据，开始计算...", len(cam_data_arr))

    cam_data = cam_data_arr.tolist()
    robot_data = robot_data_arr.tolist()

    result = run_calibration(
        cam_data=cam_data,
        robot_data=robot_data,
        compare_methods=args.compare_methods,
        outlier_rejection=args.outlier_rejection,
        max_error_mm=args.max_error,
        min_poses=args.min_poses,
        refine=args.refine,
        method=args.method,
    )

    # 打印结果
    print_results(result)

    # 留一法分析
    if args.loo:
        import cv2
        R_g2b, t_g2b, R_t2c, t_t2c = _prepare_input_lists(cam_data, robot_data)
        method_id = HAND_EYE_METHODS.get(
            (args.method or 'tsai').upper(), cv2.CALIB_HAND_EYE_TSAI
        )
        loo = leave_one_out_analysis(R_g2b, t_g2b, R_t2c, t_t2c, method=method_id)

        # 基线误差
        from robovision.calibration.hand_eye import compute_reprojection_error
        R_c2g_all, t_c2g_all = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=method_id,
        )
        _, _, baseline_error, _ = compute_reprojection_error(
            R_c2g_all, t_c2g_all, R_g2b, t_g2b, R_t2c, t_t2c,
        )
        print_loo_results(loo, baseline_error)

    # 保存结果
    if not args.no_save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_hand_eye_result(args.output, result)
        logger.info("结果已保存到: %s", args.output)

    return result


if __name__ == '__main__':
    main()
