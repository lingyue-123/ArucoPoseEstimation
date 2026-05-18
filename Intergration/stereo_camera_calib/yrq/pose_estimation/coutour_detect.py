import cv2
import numpy as np

def detect_charging_port_cover(image_path):
    """
    使用传统图像处理方法检测充电口盖轮廓，
    绘制外接矩形、中心点，输出中心坐标
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    original = img.copy()
    height, width = img.shape[:2]

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. 边缘检测 (Canny)
    low_thresh = 50
    high_thresh = 150
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # 5. 形态学闭运算连接断裂边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 筛选轮廓（面积、宽高比、矩形度）
    candidate_contours = []
    min_area = (width * height) * 0.02
    max_area = (width * height) * 0.5

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 3:
            continue

        rect_area = w * h
        solidity = area / float(rect_area) if rect_area != 0 else 0
        if solidity < 0.4:
            continue

        candidate_contours.append(cnt)

    # 如果没有找到合适轮廓，放宽面积限制再试
    if len(candidate_contours) == 0:
        print("未检测到符合条件的轮廓，尝试放宽面积范围...")
        min_area = (width * height) * 0.005
        max_area = (width * height) * 0.7
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            candidate_contours.append(cnt)

    # 8. 绘制所有候选轮廓（绿色）
    cv2.drawContours(original, candidate_contours, -1, (0, 255, 0), 2)

    # 9. 定位最大轮廓（认为是充电口盖）
    if candidate_contours:
        candidate_contours.sort(key=cv2.contourArea, reverse=True)
        max_contour = candidate_contours[0]

        # 绘制最大轮廓（红色粗线）
        cv2.drawContours(img, [max_contour], -1, (0, 0, 255), 4)
        cv2.putText(img, "Charging Port Cover", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 10. 计算外接矩形（轴对齐矩形）
        x, y, w, h = cv2.boundingRect(max_contour)
        # 绘制外接矩形（蓝色线条）
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # 11. 计算外接矩形的中心点坐标
        center_x = x + w // 2
        center_y = y + h // 2
        center_point = (center_x, center_y)

        # 绘制中心点（红色圆点）
        cv2.circle(img, center_point, 8, (0, 0, 255), -1)
        cv2.circle(img, center_point, 12, (255, 255, 255), 2)  # 外白圈增强可见性

        # 在图像上标注中心坐标
        coord_text = f"({center_x}, {center_y})"
        cv2.putText(img, coord_text, (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 控制台输出坐标
        print(f"充电口盖外接矩形中心点坐标: ({center_x}, {center_y})")
        print(f"外接矩形参数: x={x}, y={y}, w={w}, h={h}")
    else:
        print("未能检测到充电口盖轮廓")

    # 12. 显示结果
    # cv2.imshow("Original + Candidate Contours", original)
    cv2.imshow("Final Detection", img)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Closed Edges", closed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_charging_port_cover_simple(img):
    height, width = img.shape[:2]
    
    # 定义一个阈值，用于过滤下半部分（例如只保留图像上半部分70%的区域）
    # 可以根据需要调整这个比例，0.5表示只保留上半部分，0.7表示保留上70%
    upper_ratio = 0.6  # 保留图像上部60%的区域
    
    # 1. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 边缘检测 (Canny)
    low_thresh = 50
    high_thresh = 150
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # 4. 形态学闭运算连接断裂边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 筛选轮廓（面积、宽高比、矩形度、以及位置过滤）
    candidate_contours = []
    min_area = (width * height) * 0.02
    max_area = (width * height) * 0.5
    
    # 定义上半部分的阈值（y坐标小于此值）
    upper_threshold = int(height * upper_ratio)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # 过滤画面下半部分的轮廓：如果轮廓的中心点或顶部在图像下半部分，则跳过
        # 方法1: 检查轮廓的中心点y坐标
        center_y = y + h // 2
        if center_y > upper_threshold:
            continue
        
        # 方法2: 或者检查轮廓的顶部y坐标（更严格）
        # if y > upper_threshold:
        #     continue
        
        aspect_ratio = w / float(h) if h != 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 3:
            continue

        rect_area = w * h
        solidity = area / float(rect_area) if rect_area != 0 else 0
        if solidity < 0.4:
            continue

        candidate_contours.append(cnt)

    # 如果没有找到合适轮廓，放宽面积限制再试
    if len(candidate_contours) == 0:
        print("未检测到符合条件的轮廓，尝试放宽面积范围...")
        min_area = (width * height) * 0.005
        max_area = (width * height) * 0.7
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 放宽限制后也要过滤下半部分的轮廓
            center_y = y + h // 2
            if center_y > upper_threshold:
                continue
            
            candidate_contours.append(cnt)

    # 7. 定位最大轮廓（认为是充电口盖）
    if candidate_contours:
        candidate_contours.sort(key=cv2.contourArea, reverse=True)
        max_contour = candidate_contours[0]
    
    # 8. 计算外接矩形（轴对齐矩形）
        x, y, w, h = cv2.boundingRect(max_contour)
        # 9. 计算外接矩形的边界坐标
        _left = x
        _right = _left + w
        _top = y
        _bottom = y + h
        return _left, _right, _top, _bottom
    else:
        # 如果没有找到轮廓，返回全图范围或者None
        print("未能检测到充电口盖轮廓")
        return 0, width, 0, height
    
if __name__ == "__main__":
    # 请替换为你的图片路径
    detect_charging_port_cover("stereo_1_Color.png")
    detect_charging_port_cover("stereo_2_Color.png")
    detect_charging_port_cover("stereo_3_Color.png")
    detect_charging_port_cover("stereo_4_Color.png")
    detect_charging_port_cover("stereo_5_Color.png")
    print(detect_charging_port_cover_simple(cv2.imread("stereo_1_Color.png")))
    print(detect_charging_port_cover_simple(cv2.imread("stereo_2_Color.png")))
    print(detect_charging_port_cover_simple(cv2.imread("stereo_3_Color.png")))
    print(detect_charging_port_cover_simple(cv2.imread("stereo_4_Color.png")))
    print(detect_charging_port_cover_simple(cv2.imread("stereo_5_Color.png")))
    