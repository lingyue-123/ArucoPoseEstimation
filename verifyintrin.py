import cv2
import numpy as np
import os
import argparse
import traceback
from pathlib import Path

# =========================
# 核心配置（请根据你的实际情况修改这部分！）
# =========================
# 1. 你要验证的相机内参（4K分辨率，与图片分辨率匹配）
CAMERA_MATRIX = np.array([[5031.01, 0, 1527.09],
                          [0, 5034.24, 1068.69],
                          [0, 0, 1]], dtype=np.float64)

# 2. 你要验证的畸变系数
DIST_COEFFS = np.array([0.0186325, -1.30301, 0.00317111, -0.0028216, 11.5479], dtype=np.float64)

# 3. 棋盘格参数（与你实际标定板一致）
BOARD_ROWS = 8          # 标定板每列内角点数量（垂直方向）
BOARD_COLS = 11         # 标定板每行内角点数量（水平方向）
SQUARE_SIZE = 0.025     # 棋盘格方格尺寸（米），比如25mm=0.025m

# 4. 图片格式（支持jpg/png/bmp等）
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# =========================
# 工具函数
# =========================
def create_board_obj_pts(rows, cols, square_size):
    """生成棋盘格3D世界点（仅Z=0的平面点）"""
    obj_pts = np.zeros((rows * cols, 3), np.float64)
    obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return obj_pts

def calculate_reprojection_error(obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs):
    """
    计算重投影误差
    参数：
        obj_pts: 3D世界点 (N,3)
        img_pts: 实际检测的2D像素点 (N,2)
        rvec/tvec: SolvePnP解算的位姿
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数
    返回：
        mean_error: 该图片的平均重投影误差（像素）
        per_point_errors: 每个角点的重投影误差（像素）
    """
    # 投影3D点到图像平面
    img_pts_proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts_proj = img_pts_proj.reshape(-1, 2)
    
    # 计算每个点的欧氏距离（像素）
    per_point_errors = np.linalg.norm(img_pts - img_pts_proj, axis=1)
    mean_error = np.mean(per_point_errors)
    
    return mean_error, per_point_errors

def process_single_image(img_path, board_obj_pts):
    """处理单张图片，返回重投影误差（失败返回None）"""
    try:
        # 读取图片（彩色转灰度）
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  读取失败：{img_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = gray.shape[:2]
        print(f"\n📸 处理图片：{os.path.basename(img_path)} (分辨率: {img_width}x{img_height})")
        
        # 鲁棒检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH 
            + cv2.CALIB_CB_NORMALIZE_IMAGE 
            + cv2.CALIB_CB_FAST_CHECK
        )
        
        if not ret:
            print(f"❌ 未检测到棋盘格角点：{img_path}")
            return None
        
        # 亚像素优化（提高角点精度）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        corners_subpix = corners_subpix.reshape(-1, 2)
        
        # 用给定内参解算位姿（EPNP算法鲁棒性高）
        success, rvec, tvec = cv2.solvePnP(
            board_obj_pts, corners_subpix, CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            print(f"❌ SolvePnP位姿解算失败：{img_path}")
            return None
        
        # 计算重投影误差
        mean_error, per_point_errors = calculate_reprojection_error(
            board_obj_pts, corners_subpix, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
        )
        
        # 输出单张图片结果
        print(f"✅ 平均重投影误差：{mean_error:.4f} 像素")
        print(f"   最大单点误差：{np.max(per_point_errors):.4f} 像素")
        print(f"   最小单点误差：{np.min(per_point_errors):.4f} 像素")
        
        return mean_error
    
    except Exception as e:
        print(f"❌ 处理图片异常 {img_path}：{str(e)}")
        traceback.print_exc()
        return None

def main(image_dir):
    """
    主函数：遍历图片文件夹，计算所有图片的重投影误差
    参数：
        image_dir: 标定板图片文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.isdir(image_dir):
        print(f"❌ 文件夹不存在：{image_dir}")
        return
    
    # 生成棋盘格3D世界点
    board_obj_pts = create_board_obj_pts(BOARD_ROWS, BOARD_COLS, SQUARE_SIZE)
    
    # 遍历文件夹中的图片
    image_paths = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(SUPPORTED_FORMATS):
            image_paths.append(os.path.join(image_dir, file))
    
    if len(image_paths) == 0:
        print(f"❌ 文件夹中未找到支持的图片格式 {SUPPORTED_FORMATS}")
        return
    
    print(f"========================================================")
    print(f"📊 相机内参重投影误差验证")
    print(f"========================================================")
    print(f"相机内参矩阵：")
    print(np.round(CAMERA_MATRIX, 2))
    print(f"\n畸变系数：")
    print(np.round(DIST_COEFFS, 6))
    print(f"\n棋盘格参数：{BOARD_ROWS}列x{BOARD_COLS}行 内角点，方格尺寸 {SQUARE_SIZE}m")
    print(f"\n待处理图片数量：{len(image_paths)}")
    print(f"========================================================\n")
    
    # 处理所有图片
    all_errors = []
    valid_images = 0
    invalid_images = 0
    
    for img_path in image_paths:
        error = process_single_image(img_path, board_obj_pts)
        if error is not None and not np.isnan(error):
            all_errors.append(error)
            valid_images += 1
        else:
            invalid_images += 1
    
    # 输出汇总结果
    print(f"\n========================================================")
    print(f"📈 重投影误差汇总")
    print(f"========================================================")
    print(f"有效图片数量：{valid_images}")
    print(f"无效图片数量：{invalid_images}")
    
    if valid_images > 0:
        mean_total_error = np.mean(all_errors)
        std_total_error = np.std(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        
        print(f"\n平均重投影误差：{mean_total_error:.4f} 像素")
        print(f"误差标准差：{std_total_error:.4f} 像素")
        print(f"最大单张误差：{max_error:.4f} 像素")
        print(f"最小单张误差：{min_error:.4f} 像素")
        
        # 工业标准判定
        if mean_total_error < 0.2:
            level = "🟢 优秀（实验室级精度）"
        elif mean_total_error < 0.5:
            level = "🟢 合格（工业级标准）"
        elif mean_total_error < 1.0:
            level = "🟡 可用（精度一般，建议重新标定）"
        else:
            level = "🔴 不合格（内参误差过大，必须重新标定）"
        
        print(f"\n判定结果：{level}")
        print(f"工业参考标准：平均重投影误差 < 0.5 像素")
    else:
        print(f"\n❌ 无有效图片，无法计算平均误差")
    print(f"========================================================")

# =========================
# 命令行入口
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='相机内参重投影误差验证工具')
    parser.add_argument('--image-dir', type=str, required=True, 
                        help='标定板图片文件夹路径（必填），例如：./calib_images')
    parser.add_argument('--board-rows', type=int, default=BOARD_ROWS,
                        help=f'标定板每列内角点数量，默认：{BOARD_ROWS}')
    parser.add_argument('--board-cols', type=int, default=BOARD_COLS,
                        help=f'标定板每行内角点数量，默认：{BOARD_COLS}')
    parser.add_argument('--square-size', type=float, default=SQUARE_SIZE,
                        help=f'棋盘格方格尺寸（米），默认：{SQUARE_SIZE}')
    
    args = parser.parse_args()
    
    # 覆盖默认参数（如果命令行传入）
    BOARD_ROWS = args.board_rows
    BOARD_COLS = args.board_cols
    SQUARE_SIZE = args.square_size
    
    # 运行主函数
    try:
        main(args.image_dir)
    except Exception as e:
        print(f"\n❌ 程序异常：{type(e).__name__} - {str(e)}")
        traceback.print_exc()
        exit(1)

