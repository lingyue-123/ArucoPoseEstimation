"""
黑色方块检测器 | Black Square / Ellipse Detector

移自 HK_cam/hough.py，保持算法逻辑不变。
通过 IOU 过滤拟合椭圆，三圆组合模式匹配。
"""

import itertools
import logging
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def detect_black_circles(
    image: np.ndarray,
    area_min: int = 8000,
    area_max: int = 100000,
    aspect_ratio_min: float = 0.5,
    aspect_ratio_max: float = 2.0,
    iou_threshold: float = 0.9,
) -> Optional[tuple]:
    """
    在图像中检测三圆排列的黑色标记（移自 HK_cam/hough.py）。

    算法：
    1. 二值化 + 形态学开运算
    2. 轮廓提取 + 椭圆拟合
    3. 面积/长宽比/IOU 过滤
    4. 三圆组合模式匹配（尺寸关系 + 距离约束）

    Args:
        image: BGR 或灰度图像
        area_min: 最小轮廓面积（像素²）
        area_max: 最大轮廓面积（像素²）
        aspect_ratio_min: 最小长宽比
        aspect_ratio_max: 最大长宽比
        iou_threshold: 椭圆与轮廓的 IOU 阈值

    Returns:
        (combo_ellipses, result_img) 或 None（未检测到时）
        combo_ellipses: 三个椭圆的元组
        result_img: 绘制了检测结果的图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    result_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_min or area > area_max:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (w, h), angle = ellipse

            aspect_ratio = h / w if h > w else w / h
            if aspect_ratio < aspect_ratio_min or aspect_ratio > aspect_ratio_max:
                continue

            mask_ellipse = np.zeros(binary.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)
            mask_contour = np.zeros(binary.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], 0, 255, -1)

            intersection = cv2.bitwise_and(mask_ellipse, mask_contour)
            union = cv2.bitwise_or(mask_ellipse, mask_contour)
            union_area = np.sum(union > 0)
            if union_area == 0:
                continue
            iou = np.sum(intersection > 0) / union_area
            if iou <= iou_threshold:
                continue

            ellipses.append(ellipse)
        except Exception:
            continue

    if len(ellipses) < 3:
        return None

    # 三圆组合匹配
    combo_true = ()
    for combo in itertools.combinations(ellipses, 3):
        centers = [e[0] for e in combo]
        widths = sorted([e[1][1] for e in combo])  # 短轴排序

        # 尺寸约束：最大圆 >= 1.5× 中等圆，最大圆 <= 1.1× 中等圆（另一对）
        if widths[2] <= 1.1 * widths[1] and widths[1] >= 1.5 * widths[0]:
            pass
        else:
            continue

        # 距离约束：三圆间距均在最大圆直径 2 倍以内
        dists = cdist(np.array(centers), np.array(centers))
        d01, d02, d12 = dists[0][1], dists[0][2], dists[1][2]
        if d01 <= 2 * widths[2] and d02 <= 2 * widths[2] and d12 <= 2 * widths[2]:
            combo_true = combo
            break

    if not combo_true:
        return None

    # 绘制结果
    for ellipse in combo_true:
        (cx, cy), (w, h), angle = ellipse
        cv2.ellipse(result_img, ellipse, (0, 255, 0), 4)
        cv2.circle(result_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(
            result_img,
            f"{int(cx)} {int(cy)} {int(angle)}",
            (int(cx) - 200, int(cy) - 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return combo_true, result_img
