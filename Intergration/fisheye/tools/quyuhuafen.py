import cv2
import numpy as np

# =========================
# 配置
# =========================
IMAGE_PATH = 'runs/stream_capture/manual_20260417_115325_388874.jpg'
SAVE_PATH = 'roi_coords.npy'

# 显示窗口最大宽高，可按你的屏幕修改
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

# 存储“原图坐标系”下的多边形顶点
pts = []

# 读取原图
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")

orig_h, orig_w = img.shape[:2]

# =========================
# 计算显示缩放比例
# =========================
scale = min(MAX_DISPLAY_W / orig_w, MAX_DISPLAY_H / orig_h, 1.0)
disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

# 缩放后的显示图
display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

def redraw():
    """
    在缩放后的显示图上重绘点和线
    """
    canvas = display_img.copy()

    if len(pts) > 0:
        # 先把原图坐标映射到显示图坐标
        disp_pts = [(int(x * scale), int(y * scale)) for (x, y) in pts]

        for p in disp_pts:
            cv2.circle(canvas, p, 4, (0, 0, 255), -1)

        if len(disp_pts) > 1:
            for i in range(len(disp_pts) - 1):
                cv2.line(canvas, disp_pts[i], disp_pts[i + 1], (0, 255, 0), 2)

            # 闭合多边形
            cv2.line(canvas, disp_pts[-1], disp_pts[0], (0, 255, 0), 2)

    cv2.putText(
        canvas,
        "Left: add point | Right: undo | s: save | q: quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow('Calibration', canvas)

def draw_roi(event, x, y, flags, param):
    global pts

    # 鼠标左键：添加点
    if event == cv2.EVENT_LBUTTONDOWN:
        # 把显示图坐标映射回原图坐标
        orig_x = int(x / scale)
        orig_y = int(y / scale)

        # 防止越界
        orig_x = max(0, min(orig_x, orig_w - 1))
        orig_y = max(0, min(orig_y, orig_h - 1))

        pts.append((orig_x, orig_y))
        redraw()

    # 鼠标右键：撤销上一个点
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pts) > 0:
            pts.pop()
            redraw()

cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration', disp_w, disp_h)
cv2.setMouseCallback('Calibration', draw_roi)

print("操作说明: 左键打点，右键撤销，按 's' 保存并退出，按 'q' 直接退出。")
print(f"原图尺寸: {orig_w} x {orig_h}")
print(f"显示尺寸: {disp_w} x {disp_h}")
print(f"缩放比例: {scale:.4f}")

redraw()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        np.save(SAVE_PATH, np.array(pts, np.int32))
        print(f"坐标已保存至 {SAVE_PATH}: {pts}")
        break

    elif key == ord('q'):
        print("未保存，直接退出。")
        break

cv2.destroyAllWindows()
