"""
内参标定工具 | Intrinsic Calibration Utilities

向后兼容现有 intrinsic_calib/intrinsic.txt 文件格式。
"""

import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicResult:
    """单组相机内参。"""
    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray    # (N,)


def load_intrinsic_txt(path: str) -> List[IntrinsicResult]:
    """
    加载 intrinsic_calib/intrinsic.txt 格式的内参文件。

    文件格式（两组，tab分隔）：
        camera intrinsic:
        K[0][0]\t0\tK[0][2]
        0\tK[1][1]\tK[1][2]
        0\t0\t1

        dist:
        d0, d1, d2, d3, d4,

    Args:
        path: intrinsic.txt 文件路径

    Returns:
        List[IntrinsicResult]，每个文件中有几组就返回几个
    """
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('camera intrinsic:')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        # 解析 3x3 矩阵（前3行）
        try:
            K_rows = []
            for i in range(3):
                row = [float(x) for x in lines[i].replace(',', ' ').split()]
                K_rows.append(row)
            K = np.array(K_rows, dtype=np.float64)
        except (ValueError, IndexError) as e:
            logger.warning("解析相机矩阵失败: %s", e)
            continue

        # 解析畸变系数
        dist_line_idx = None
        for j, l in enumerate(lines):
            if l.lower().startswith('dist'):
                dist_line_idx = j + 1
                break
        if dist_line_idx is None or dist_line_idx >= len(lines):
            logger.warning("未找到 dist 行")
            continue
        try:
            dist_values = [float(x) for x in lines[dist_line_idx].replace(',', ' ').split() if x]
            dist = np.array(dist_values, dtype=np.float64)
        except ValueError as e:
            logger.warning("解析畸变系数失败: %s", e)
            continue

        results.append(IntrinsicResult(camera_matrix=K, dist_coeffs=dist))

    logger.info("从 %s 加载了 %d 组内参", path, len(results))
    return results


def save_intrinsic_txt(path: str, results: List[IntrinsicResult]) -> None:
    """
    将内参写入 intrinsic.txt 格式（向后兼容）。

    Args:
        path: 输出文件路径
        results: 内参列表
    """
    lines = []
    for r in results:
        K = r.camera_matrix
        dist = r.dist_coeffs
        lines.append("camera intrinsic: ")
        for row in K:
            lines.append('\t'.join(f'{v:.6g}' for v in row))
        lines.append("")
        lines.append("dist: ")
        lines.append(', '.join(f'{v}' for v in dist) + ', ')
        lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info("内参已保存到 %s", path)
