from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import onnxruntime
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

# 类别定义
CLASSES = ['class0']


class YOLOv5:
    """YOLOv5 ONNX推理类"""

    def __init__(self, onnx_path: str):
        """
        初始化YOLOv5模型

        Args:
            onnx_path: ONNX模型文件路径
        """
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path, 
            providers=['CUDAExecutionProvider']
        )
        self.input_name = self._get_input_name()
        self.output_name = self._get_output_name()

    def _get_input_name(self) -> List[str]:
        """获取输入节点名称"""
        return [node.name for node in self.onnx_session.get_inputs()]

    def _get_output_name(self) -> List[str]:
        """获取输出节点名称"""
        return [node.name for node in self.onnx_session.get_outputs()]

    def _get_input_feed(self, img_tensor: np.ndarray) -> dict:
        """构建输入数据字典"""
        return {name: img_tensor for name in self.input_name}

    def inference(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行推理

        Args:
            image: 图像路径或numpy数组

        Returns:
            pred: 预测结果
            img: 原始图像
        """
        # 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise ValueError("Input must be a file path (str) or an image array (np.ndarray)")

        if img is None:
            raise ValueError("Image not loaded correctly. Check the input path or image data.")

        # 预处理
        or_img = cv2.resize(img, (640, 640))
        _img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB 和 HWC2CHW
        _img = _img.astype(dtype=np.float32)
        _img /= 255.0
        _img = np.expand_dims(_img, axis=0)

        # 推理
        input_feed = self._get_input_feed(_img)
        pred = self.onnx_session.run(None, input_feed)[0]

        return pred, img


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    """
    非极大值抑制

    Args:
        dets: 检测框数组 [N, 5] (x1, y1, x2, y2, score)
        thresh: IOU阈值

    Returns:
        keep: 保留的索引列表
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]

    return keep


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    将xywh格式转换为xyxy格式

    Args:
        x: 输入数组 [N, 4] (x_center, y_center, width, height)

    Returns:
        y: 输出数组 [N, 4] (x1, y1, x2, y2)
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# def filter_box(org_box: np.ndarray, conf_thres: float, iou_thres: float) -> np.ndarray:
#     """
#     过滤检测框（按置信度和NMS）

#     Args:
#         org_box: 原始检测框输出
#         conf_thres: 置信度阈值
#         iou_thres: NMS IOU阈值

#     Returns:
#         output: 过滤后的检测框
#     """
#     org_box = np.squeeze(org_box)
#     conf = org_box[..., 4] > conf_thres
#     box = org_box[conf == True]

#     # 获取类别
#     cls_conf = box[..., 5:]
#     cls = [int(np.argmax(c)) for c in cls_conf]
#     all_cls = list(set(cls))

#     output = []
#     for curr_cls in all_cls:
#         curr_cls_box = []
#         for j in range(len(cls)):
#             if cls[j] == curr_cls:
#                 box[j][5] = curr_cls
#                 curr_cls_box.append(box[j][:6])

#         curr_cls_box = np.array(curr_cls_box)
#         curr_cls_box = xywh2xyxy(curr_cls_box)
#         curr_out_box = nms(curr_cls_box, iou_thres)

#         for k in curr_out_box:
#             output.append(curr_cls_box[k])

#     return np.array(output)

def filter_box(org_box: np.ndarray, conf_thres: float, iou_thres: float) -> np.ndarray:
    """
    过滤检测框（按置信度和NMS）
    """
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]

    # 获取类别
    cls_conf = box[..., 5:]
    cls = [int(np.argmax(c)) for c in cls_conf]
    all_cls = list(set(cls))

    output = []
    for curr_cls in all_cls:
        curr_cls_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])

        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        
        # ========== 关键修改：添加坐标裁剪 ==========
        # 裁剪到 [0, 640] 范围
        curr_cls_box[:, 0] = np.clip(curr_cls_box[:, 0], 0, 640)  # x1
        curr_cls_box[:, 1] = np.clip(curr_cls_box[:, 1], 0, 640)  # y1
        curr_cls_box[:, 2] = np.clip(curr_cls_box[:, 2], 0, 640)  # x2
        curr_cls_box[:, 3] = np.clip(curr_cls_box[:, 3], 0, 640)  # y2
        
        # 确保宽高为正（避免后续计算错误）
        curr_cls_box[:, 2] = np.maximum(curr_cls_box[:, 0] + 1, curr_cls_box[:, 2])
        curr_cls_box[:, 3] = np.maximum(curr_cls_box[:, 1] + 1, curr_cls_box[:, 3])
        # =========================================
        
        curr_out_box = nms(curr_cls_box, iou_thres)

        for k in curr_out_box:
            output.append(curr_cls_box[k])

    return np.array(output) if output else np.array([])

def draw(image: np.ndarray, box_data: np.ndarray) -> None:
    """
    在图像上绘制检测框

    Args:
        image: 输入图像
        box_data: 检测框数据
    """
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print(f'class: {CLASSES[cl]}, score: {score}')
        print(f'box coordinate left,top,right,down: [{top}, {left}, {right}, {bottom}]')

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image, f'{CLASSES[cl]} {score:.2f}',
            (top, left),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )


def clip(image: np.ndarray, box_data: np.ndarray) -> Optional[Tuple]:
    """
    根据检测框裁剪图像区域

    Args:
        image: 原始图像
        box_data: 检测框数据

    Returns:
        裁剪后的图像及坐标，如果检测失败返回None
    """
    if box_data.shape[0] != 1:
        print("DC port detection failed. Please try again.")
        return None

    height, width = image.shape[:2]
    scale_h = height / 640
    scale_w = width / 640

    box_data = np.squeeze(box_data)
    left, top, right, bottom = box_data[:4].astype(np.int32)

    _top = int(top * scale_h)
    _bottom = int(bottom * scale_h)
    _left = int(left * scale_w)
    _right = int(right * scale_w)

    return image[_top:_bottom, _left:_right], _top, _left, _bottom, _right

def get_pix_co(onnx_path, input_path):
    onnx_path = "/home/nvidia/Downloads/LN/HK_2/checkpoint/best_small_20260305.onnx"
    input_path = '/home/nvidia/Downloads/LN/HK_2/Image_20260323155538546.png'

    model = YOLOv5(onnx_path)
    output, img = model.inference(input_path)
    outbox = filter_box(output, 0.5, 0.5)

    if outbox.shape[0] != 1:
        print("Detection failed. Please try again.")
        return None
    
    height, width = img.shape[:2]
    scale_h = height / 640
    scale_w = width / 640

    outbox = np.squeeze(outbox)
    left, top, right, bottom = outbox[:4].astype(np.int32)

    _top = int(top * scale_h)
    _bottom = int(bottom * scale_h)
    _left = int(left * scale_w)
    _right = int(right * scale_w)

    left_top = (_left, _top)
    left_bottom = (_left, _bottom)
    right_top = (_right, _top)
    right_bottom = (_right, _bottom)

    return left_top, left_bottom, right_top, right_bottom
