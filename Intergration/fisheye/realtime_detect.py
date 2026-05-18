import argparse
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np

# =========================================================
# 兼容补丁：解决 torch.load() 旧权重时出现
# ModuleNotFoundError: No module named 'numpy._core'
# =========================================================
try:
    import numpy._core
except ModuleNotFoundError:
    import numpy.core as _np_core
    sys.modules['numpy._core'] = _np_core

    try:
        import numpy.core.multiarray as _np_multiarray
        sys.modules['numpy._core.multiarray'] = _np_multiarray
    except Exception:
        pass

    try:
        import numpy.core.umath as _np_umath
        sys.modules['numpy._core.umath'] = _np_umath
    except Exception:
        pass

    try:
        import numpy.core.numeric as _np_numeric
        sys.modules['numpy._core.numeric'] = _np_numeric
    except Exception:
        pass

    try:
        import numpy.core._multiarray_umath as _np_mau
        sys.modules['numpy._core._multiarray_umath'] = _np_mau
    except Exception:
        pass

import cv2
import torch

# 低延迟 RTSP 参数：保守一点，优先稳定
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"
)
os.environ["YOLO_IGNORE_REQUIREMENTS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import (
    LOGGER, Profile, check_img_size, check_requirements,
    colorstr, increment_path, non_max_suppression, print_args,
    scale_boxes, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def unwrap_prediction(pred):
    """将模型输出统一解包成 Tensor，供 non_max_suppression 使用。"""
    if isinstance(pred, torch.Tensor):
        return pred
    if isinstance(pred, (list, tuple)):
        for item in pred:
            out = unwrap_prediction(item)
            if isinstance(out, torch.Tensor):
                return out
    return pred


def load_roi(roi_path):
    """
    加载 ROI 多边形，返回 shape=(N,1,2) 的 int32 数组
    """
    roi = np.load(roi_path)
    if roi is None or len(roi) < 3:
        raise ValueError(f"ROI 点数量不足，无法构成多边形: {roi_path}")
    roi = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    return roi


def is_in_roi(xyxy, roi_polygon):
    """
    用检测框底边中心点判断是否进入 ROI
    xyxy: [x1, y1, x2, y2]
    返回:
        intrusion: bool
        foot_point: (x, y)
    """
    x1, y1, x2, y2 = map(int, xyxy)
    foot_x = (x1 + x2) // 2
    foot_y = y2
    res = cv2.pointPolygonTest(roi_polygon, (foot_x, foot_y), False)
    return res >= 0, (foot_x, foot_y)


def draw_roi_overlay(image, roi_polygon):
    """
    在图像上绘制 ROI 区域
    """
    overlay = image.copy()
    cv2.fillPoly(overlay, [roi_polygon], color=(0, 255, 255))
    image = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)
    cv2.polylines(image, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    return image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    YOLO 风格 letterbox:
    等比例缩放 + padding，返回:
    im, ratio, (dw, dh)
    """
    shape = im.shape[:2]  # current shape [h, w]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def create_timestamp_dir(root_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(root_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def resize_with_letterbox_for_display(image, target_w, target_h):
    """
    将图像按比例缩放并填充到固定显示窗口内，保证完整显示，不裁切。
    返回 display_image, scale
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image, 1.0

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale


class RTSPFrameGrabber:
    """
    低延迟 RTSP 拉流器：
    - 子线程持续拉流
    - 只保留最新帧
    - 断流自动重连
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.fail_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    def _open(self):
        backends = [
            None,               # 默认后端
            cv2.CAP_FFMPEG,     # FFMPEG
            cv2.CAP_ANY         # 任意可用后端
        ]

        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(self.rtsp_url)
                    backend_name = "DEFAULT"
                else:
                    cap = cv2.VideoCapture(self.rtsp_url, backend)
                    backend_name = str(backend)

                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"成功打开 RTSP 流，backend={backend_name}")
                    return cap
                else:
                    if cap is not None:
                        cap.release()
            except Exception as e:
                print(f"backend={backend} 打开失败: {e}")

        return None

    def _capture_loop(self):
        self.cap = self._open()

        if self.cap is None or not self.cap.isOpened():
            print("无法打开RTSP流，请检查：")
            print("1. RTSP 地址是否正确")
            print("2. 用户名/密码是否正确")
            print("3. 摄像头和本机网络是否互通")
            print("4. OpenCV 是否支持对应视频后端")
            self.running = False
            return

        print("采集线程已启动")
        fail_streak = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                fail_streak += 1
                print(f"读取失败 {fail_streak} 次，尝试重连...")

                if fail_streak >= 5:
                    if self.cap is not None:
                        self.cap.release()
                    time.sleep(0.5)
                    self.cap = self._open()
                    fail_streak = 0
                continue

            fail_streak = 0

            with self.lock:
                self.latest_frame = frame

        if self.cap is not None:
            self.cap.release()
        print("采集线程退出")

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


@smart_inference_mode()
def run(
    weights=ROOT / 'yolo.pt',
    source='rtsp://admin:@192.168.1.10:554/stream1',
    data=ROOT / 'data/coco.yaml',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=True,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    display_fps=30,
    save_root_dir='runs/stream_capture',
    auto_save=False,
    sample_fps=5,
    window_width=1280,
    window_height=720,
    project=ROOT / 'runs/detect',
    name='exp_rtsp',
    exist_ok=False,
    roi_path='roi_coords.npy',
    intrusion_classes=('Pedestrian', 'Car'),
):
    os.makedirs(save_root_dir, exist_ok=True)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 加载 ROI
    roi_polygon = load_roi(roi_path)
    print(f"[INFO] ROI loaded from {roi_path}: {roi_polygon.reshape(-1, 2).tolist()}")
    print(f"[INFO] Intrusion classes: {intrusion_classes}")

    model.warmup(imgsz=(1 if pt or getattr(model, "triton", False) else 1, 3, *imgsz))

    grabber = RTSPFrameGrabber(source)
    grabber.start()

    window_name = "RTSP Real-time Detection"
    if view_img:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.moveWindow(window_name, 100, 100)

    auto_save_mode = auto_save
    current_session_dir = None
    save_counter = 0
    last_save_time = 0.0

    if auto_save_mode:
        current_session_dir = create_timestamp_dir(save_root_dir)
        last_save_time = time.time() - 1.0 / max(sample_fps, 1)
        print(f"开始自动保存，目录：{current_session_dir}")

    seen = 0
    dt = (Profile(), Profile(), Profile())
    last_display_time = 0.0
    last_log_time = 0.0

    print("操作提示：")
    print("请先点击图像窗口，再按键控制")
    print("q: 退出程序")
    print("s: 开始/停止自动保存检测图")
    print("m: 手动保存当前检测图")

    try:
        while True:
            now = time.time()
            if display_fps > 0 and (now - last_display_time < 1.0 / display_fps):
                time.sleep(0.001)
                continue
            last_display_time = now

            frame = grabber.read()
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            im0 = frame.copy()

            # 预处理
            with dt[0]:
                img, ratio, pad = letterbox(im0, new_shape=imgsz, stride=stride, auto=False)
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
                img = np.ascontiguousarray(img)

                im = torch.from_numpy(img).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255.0
                if len(im.shape) == 3:
                    im = im[None]

            # 推理
            with dt[1]:
                pred = model(im, augment=augment, visualize=False)
                pred = unwrap_prediction(pred)
                if not isinstance(pred, torch.Tensor):
                    raise TypeError(f"model(im) 的返回值无法用于 non_max_suppression，当前类型为: {type(pred)}")

            # NMS
            with dt[2]:
                pred = non_max_suppression(
                    pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
                )

            det = pred[0]
            seen += 1

            # 先画 ROI
            im0 = draw_roi_overlay(im0, roi_polygon)

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            det_count = 0
            intrusion_count = 0

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape, ratio_pad=(ratio, pad)).round()
                det_count = len(det)

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    class_name = names[c]
                    xyxy_int = [int(v) for v in xyxy]

                    intrusion = False
                    foot_point = None

                    # 只对指定类别做入侵判断
                    if class_name in intrusion_classes:
                        intrusion, foot_point = is_in_roi(xyxy_int, roi_polygon)
                        if intrusion:
                            intrusion_count += 1
                            print(f"[INTRUSION] {class_name} entered ROI at {foot_point}, conf={float(conf):.2f}")

                    if save_txt:
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        txt_path = str(save_dir / 'labels' / f'frame_{seen:06d}.txt')
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 标签与颜色
                    if intrusion:
                        box_color = (0, 0, 255)  # 红色
                        label = None if hide_labels else (
                            f"INTRUSION {class_name}" if hide_conf else f"INTRUSION {class_name} {conf:.2f}"
                        )
                    else:
                        box_color = colors(c, True)
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                        )

                    annotator.box_label(xyxy, label, color=box_color)

                    # 画底边中心点
                    if foot_point is not None:
                        point_color = (0, 0, 255) if intrusion else (255, 0, 0)
                        cv2.circle(im0, foot_point, 5, point_color, -1)

                    if save_crop:
                        crop_dir = save_dir / 'crops' / names[c]
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        save_one_box(
                            xyxy,
                            im0.copy(),
                            file=crop_dir / f'frame_{seen:06d}.jpg',
                            BGR=True
                        )

            result = annotator.result()

            # 叠加信息
            mode_text = "AUTO SAVE ON" if auto_save_mode else "PREVIEW"
            cv2.putText(result, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(result, f"Detections: {det_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(result, f"Intrusions: {intrusion_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(result, f"Infer: {dt[1].dt * 1E3:.1f} ms", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if auto_save_mode and current_session_dir:
                folder_name = os.path.basename(current_session_dir)
                cv2.putText(result, f"Folder: {folder_name}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 显示时按窗口大小等比例缩放，保证完整显示
            display_img = result
            if view_img:
                display_img, _ = resize_with_letterbox_for_display(result, window_width, window_height)
                cv2.imshow(window_name, display_img)

            # 自动保存图像（保存原始检测结果，不保存缩放后的显示图）
            if auto_save_mode and current_session_dir is not None:
                current_time = time.time()
                if current_time - last_save_time >= 1.0 / max(sample_fps, 1):
                    img_name = f"{save_counter:06d}.jpg"
                    img_path = os.path.join(current_session_dir, img_name)
                    ok = cv2.imwrite(img_path, result)
                    if ok:
                        save_counter += 1
                        last_save_time = current_time

            # 每 2 秒打印一次状态，避免终端刷屏
            current_log_time = time.time()
            if current_log_time - last_log_time > 2.0:
                print(f"[INFO] det={det_count}, intrusion={intrusion_count}, infer={dt[1].dt * 1E3:.1f} ms, autosave={auto_save_mode}")
                last_log_time = current_log_time

            # 键盘控制：在图像窗口中按
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("收到退出指令，正在结束程序...")
                break

            elif key == ord('s'):
                auto_save_mode = not auto_save_mode
                if auto_save_mode:
                    current_session_dir = create_timestamp_dir(save_root_dir)
                    save_counter = 0
                    last_save_time = time.time() - 1.0 / max(sample_fps, 1)
                    print(f"开始自动保存到: {current_session_dir}")
                else:
                    print("停止自动保存")
                    current_session_dir = None

            elif key == ord('m'):
                manual_name = datetime.now().strftime("manual_%Y%m%d_%H%M%S_%f.jpg")
                img_path = os.path.join(save_root_dir, manual_name)
                ok = cv2.imwrite(img_path, result)
                if ok:
                    print(f"手动保存成功: {img_path}")
                else:
                    print(f"手动保存失败: {img_path}")

    finally:
        grabber.stop()
        cv2.destroyAllWindows()

    if seen > 0:
        t = tuple(x.t / seen * 1E3 for x in dt)
        LOGGER.info(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per frame at shape {(1, 3, *imgsz)}" % t
        )

    if save_txt:
        s = f"\n{len(list((save_dir / 'labels').glob('*.txt')))} labels saved to {save_dir / 'labels'}"
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path')
    parser.add_argument('--source', type=str, default='rtsp://admin:@192.168.1.10:554/stream1', help='RTSP URL')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per frame')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in txt')
    parser.add_argument('--save-crop', action='store_true', help='save cropped boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX')
    parser.add_argument('--display-fps', type=int, default=30, help='display fps limit')
    parser.add_argument('--save-root-dir', type=str, default='runs/stream_capture', help='manual/auto image save dir')
    parser.add_argument('--auto-save', action='store_true', help='start with auto save enabled')
    parser.add_argument('--sample-fps', type=int, default=5, help='auto save fps')
    parser.add_argument('--window-width', type=int, default=1280, help='display window width')
    parser.add_argument('--window-height', type=int, default=720, help='display window height')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp_rtsp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--roi-path', type=str, default='roi_coords.npy', help='ROI polygon numpy file')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)