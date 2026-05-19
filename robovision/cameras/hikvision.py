"""
海康威视工业相机封装 | Hikvision Industrial Camera

封装 MVS SDK（MvCameraControl_class），提供标准 CameraInterface 接口。
SDK 安装路径：/opt/MVS/Samples/aarch64/Python/MvImport

功能：
- 自动枚举设备（GigE / USB3）
- 后台取流线程（非阻塞读帧）
- HB 压缩格式自动解码
- 线程安全的最新帧缓存
"""

import logging
import platform
import sys
import threading
from ctypes import *
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraInterface, CameraIntrinsics

logger = logging.getLogger(__name__)

# 添加 MVS SDK Python 路径
_MVS_PATH = '/opt/MVS/Samples/aarch64/Python/MvImport'
if _MVS_PATH not in sys.path:
    sys.path.append(_MVS_PATH)


def _import_mvs():
    """延迟导入 MVS SDK，避免无 SDK 环境报错。"""
    try:
        from MvCameraControl_class import (
            MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
            MV_GIGE_DEVICE, MV_USB_DEVICE, MV_GENTL_GIGE_DEVICE,
            MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE, MV_GENTL_XOF_DEVICE,
            MV_ACCESS_Exclusive, MV_TRIGGER_MODE_OFF,
            MV_FRAME_OUT, MV_CC_HB_DECODE_PARAM, MV_CC_PIXEL_CONVERT_PARAM_EX,
            PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_Mono8,
        )
        from MvCameraControl_class import MVCC_FLOATVALUE
        return {'MvCamera': MvCamera, 'MV_CC_DEVICE_INFO_LIST': MV_CC_DEVICE_INFO_LIST,
                'MV_CC_DEVICE_INFO': MV_CC_DEVICE_INFO, 'MV_GIGE_DEVICE': MV_GIGE_DEVICE,
                'MV_USB_DEVICE': MV_USB_DEVICE, 'MV_GENTL_GIGE_DEVICE': MV_GENTL_GIGE_DEVICE,
                'MV_GENTL_CAMERALINK_DEVICE': MV_GENTL_CAMERALINK_DEVICE,
                'MV_GENTL_CXP_DEVICE': MV_GENTL_CXP_DEVICE, 'MV_GENTL_XOF_DEVICE': MV_GENTL_XOF_DEVICE,
                'MV_ACCESS_Exclusive': MV_ACCESS_Exclusive, 'MV_TRIGGER_MODE_OFF': MV_TRIGGER_MODE_OFF,
                'MV_FRAME_OUT': MV_FRAME_OUT, 'MV_CC_HB_DECODE_PARAM': MV_CC_HB_DECODE_PARAM,
                'MV_CC_PIXEL_CONVERT_PARAM_EX': MV_CC_PIXEL_CONVERT_PARAM_EX,
                'PixelType_Gvsp_RGB8_Packed': PixelType_Gvsp_RGB8_Packed,
                'PixelType_Gvsp_Mono8': PixelType_Gvsp_Mono8,
                'MVCC_FLOATVALUE': MVCC_FLOATVALUE}
    except ImportError as e:
        raise RuntimeError(
            f"MVS SDK 未找到，请确认安装路径 {_MVS_PATH}\n错误: {e}"
        )


# 海康 HB / Mono 像素格式 ID 集合（用于格式判断）
_HB_PIXEL_TYPES = frozenset([
    0x80000001, 0x80000002, 0x80000003, 0x80000004,  # HB_Mono variants
    0x80000010, 0x80000011, 0x80000020, 0x80000021,  # HB_RGB/BGR variants
    0x8001000C, 0x8001000D,  # HB_BayerRG variants
])
_MONO_PIXEL_TYPES = frozenset([
    0x01080001,  # Mono8
    0x01100003, 0x010C0004,  # Mono10/12
    0x01100005, 0x010C0006,  # Mono10/12 Packed
    0x01100007,  # Mono14
    0x01100025,  # Mono16
])


class HikvisionCamera(CameraInterface):
    """
    海康威视工业相机（GigE / USB3）。

    Args:
        intrinsics: 相机内参（来自 config/cameras.yaml）
        device_index: 设备列表索引，None 时交互选择
    """

    def __init__(self, intrinsics: CameraIntrinsics, device_index: Optional[int] = None):
        self._intrinsics = intrinsics
        self._device_index = device_index
        self._cam = None
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._dst_buf = None

    def open(self) -> None:
        """枚举设备、选择并初始化相机，启动取流线程。"""
        mvs = _import_mvs()
        MvCamera = mvs['MvCamera']
        MV_CC_DEVICE_INFO_LIST = mvs['MV_CC_DEVICE_INFO_LIST']
        MV_CC_DEVICE_INFO = mvs['MV_CC_DEVICE_INFO']
        MV_ACCESS_Exclusive = mvs['MV_ACCESS_Exclusive']
        MV_TRIGGER_MODE_OFF = mvs['MV_TRIGGER_MODE_OFF']

        MvCamera.MV_CC_Initialize()
        device_list = MV_CC_DEVICE_INFO_LIST()
        tl_type = (mvs['MV_GIGE_DEVICE'] | mvs['MV_USB_DEVICE'] |
                   mvs['MV_GENTL_CAMERALINK_DEVICE'] | mvs['MV_GENTL_CXP_DEVICE'] |
                   mvs['MV_GENTL_XOF_DEVICE'])

        ret = MvCamera.MV_CC_EnumDevices(tl_type, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            raise RuntimeError(f"未找到海康相机 (ret=0x{ret:x})")

        logger.info("发现 %d 台海康设备", device_list.nDeviceNum)
        for i in range(device_list.nDeviceNum):
            info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType in (mvs['MV_GIGE_DEVICE'], mvs['MV_GENTL_GIGE_DEVICE']):
                ip = info.SpecialInfo.stGigEInfo.nCurrentIp
                ip_str = f"{(ip>>24)&0xff}.{(ip>>16)&0xff}.{(ip>>8)&0xff}.{ip&0xff}"
                model = ''.join(chr(c) for c in info.SpecialInfo.stGigEInfo.chModelName if c)
                logger.info("[%d] GigE model=%s ip=%s", i, model, ip_str)
            elif info.nTLayerType == mvs['MV_USB_DEVICE']:
                model = ''.join(chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c)
                logger.info("[%d] USB model=%s", i, model)

        idx = self._device_index
        if idx is None:
            idx = int(input("请输入设备编号: "))
        if idx >= device_list.nDeviceNum:
            raise ValueError(f"设备编号 {idx} 超出范围 (共 {device_list.nDeviceNum} 台)")

        cam = MvCamera()
        dev_info = cast(device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
        if cam.MV_CC_CreateHandle(dev_info) != 0:
            raise RuntimeError("CreateHandle 失败")
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            raise RuntimeError("OpenDevice 失败")

        if dev_info.nTLayerType in (mvs['MV_GIGE_DEVICE'], mvs['MV_GENTL_GIGE_DEVICE']):
            pkt_size = cam.MV_CC_GetOptimalPacketSize()
            if pkt_size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if cam.MV_CC_StartGrabbing() != 0:
            raise RuntimeError("StartGrabbing 失败")

        self._cam = cam
        self._mvs = mvs
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        logger.info("海康相机初始化成功，取流线程已启动")

    def _is_hb_format(self, pixel_type: int) -> bool:
        return pixel_type in _HB_PIXEL_TYPES

    def _is_mono_format(self, pixel_type: int) -> bool:
        return pixel_type in _MONO_PIXEL_TYPES

    def _grab_loop(self) -> None:
        """后台取流线程。"""
        mvs = self._mvs
        MV_FRAME_OUT = mvs['MV_FRAME_OUT']
        MV_CC_HB_DECODE_PARAM = mvs['MV_CC_HB_DECODE_PARAM']
        MV_CC_PIXEL_CONVERT_PARAM_EX = mvs['MV_CC_PIXEL_CONVERT_PARAM_EX']
        PixelType_Gvsp_RGB8_Packed = mvs['PixelType_Gvsp_RGB8_Packed']
        PixelType_Gvsp_Mono8 = mvs['PixelType_Gvsp_Mono8']

        out_frame = MV_FRAME_OUT()
        memset(byref(out_frame), 0, sizeof(out_frame))

        while not self._stop_event.is_set():
            ret = self._cam.MV_CC_GetImageBuffer(out_frame, 1000)
            if out_frame.pBufAddr is None or ret != 0:
                continue

            convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(convert_param), 0, sizeof(convert_param))

            if self._is_hb_format(out_frame.stFrameInfo.enPixelType):
                decode_param = MV_CC_HB_DECODE_PARAM()
                buf_len = out_frame.stFrameInfo.nWidth * out_frame.stFrameInfo.nHeight * 3
                decode_buf = (c_ubyte * buf_len)()
                decode_param.pSrcBuf = out_frame.pBufAddr
                decode_param.nSrcLen = out_frame.stFrameInfo.nFrameLen
                decode_param.pDstBuf = decode_buf
                decode_param.nDstBufSize = buf_len
                if self._cam.MV_CC_HBDecode(decode_param) != 0:
                    self._cam.MV_CC_FreeImageBuffer(out_frame)
                    continue
                convert_param.pSrcData = decode_param.pDstBuf
                convert_param.nSrcDataLen = decode_param.nDstBufLen
                convert_param.enSrcPixelType = decode_param.enDstPixelType
            else:
                convert_param.pSrcData = out_frame.pBufAddr
                convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen
                convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType

            is_mono = self._is_mono_format(convert_param.enSrcPixelType)
            dst_type = PixelType_Gvsp_Mono8 if is_mono else PixelType_Gvsp_RGB8_Packed
            ch = 1 if is_mono else 3
            w, h = out_frame.stFrameInfo.nWidth, out_frame.stFrameInfo.nHeight
            dst_len = ch * w * h
            if self._dst_buf is None or len(self._dst_buf) < dst_len:
                self._dst_buf = (c_ubyte * dst_len)()
            dst_buf = self._dst_buf

            convert_param.nWidth = w
            convert_param.nHeight = h
            convert_param.enDstPixelType = dst_type
            convert_param.pDstBuffer = dst_buf
            convert_param.nDstBufferSize = dst_len

            if self._cam.MV_CC_ConvertPixelTypeEx(convert_param) != 0:
                self._cam.MV_CC_FreeImageBuffer(out_frame)
                continue

            arr = np.frombuffer(dst_buf, dtype=np.uint8, count=dst_len)
            if is_mono:
                img = arr.reshape(h, w)
            else:
                img = arr.reshape(h, w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            with self._frame_lock:
                self._latest_frame = img.copy()

            self._frame_event.set()
            self._cam.MV_CC_FreeImageBuffer(out_frame)

    def read_frame(self, timeout: float = 0.0) -> Tuple[bool, Optional[np.ndarray]]:
        """读取最新帧。timeout>0 时等待新帧到达，0 为非阻塞。"""
        if timeout > 0:
            if not self._frame_event.wait(timeout):
                return False, None
            self._frame_event.clear()
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame

    def close(self) -> None:
        """停止取流，释放相机资源。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cam is not None:
            self._cam.MV_CC_StopGrabbing()
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._mvs['MvCamera'].MV_CC_Finalize()
        self._cam = None
        logger.info("海康相机资源已释放")

    # --- 曝光 / 增益控制 ---

    def _require_cam(self):
        if self._cam is None:
            raise RuntimeError("相机未初始化，请先调用 open()")

    def set_exposure_auto(self, enable: bool) -> bool:
        """开关自动曝光。enable=True -> Continuous, enable=False -> Off."""
        self._require_cam()
        try:
            ret = self._cam.MV_CC_SetEnumValue("ExposureAuto", 2 if enable else 0)
            return ret == 0
        except Exception:
            return False

    def set_exposure_time(self, us: float) -> bool:
        """设置手动曝光时间（微秒），需先关闭自动曝光。"""
        self._require_cam()
        try:
            self._cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            ret = self._cam.MV_CC_SetFloatValue("ExposureTime", float(us))
            return ret == 0
        except Exception:
            return False

    def get_exposure_time(self) -> Optional[float]:
        """获取当前曝光时间（微秒）。"""
        self._require_cam()
        try:
            st = self._mvs['MVCC_FLOATVALUE']()
            memset(byref(st), 0, sizeof(st))
            ret = self._cam.MV_CC_GetFloatValue("ExposureTime", st)
            if ret == 0:
                return st.fCurValue
        except Exception:
            pass
        return None

    def set_gain(self, db: float) -> bool:
        """设置模拟/数字增益（dB），需先关闭自动增益。"""
        self._require_cam()
        try:
            self._cam.MV_CC_SetEnumValue("GainAuto", 0)
            ret = self._cam.MV_CC_SetFloatValue("Gain", float(db))
            return ret == 0
        except Exception:
            return False

    def get_gain(self) -> Optional[float]:
        """获取当前增益（dB）。"""
        self._require_cam()
        try:
            st = self._mvs['MVCC_FLOATVALUE']()
            memset(byref(st), 0, sizeof(st))
            ret = self._cam.MV_CC_GetFloatValue("Gain", st)
            if ret == 0:
                return st.fCurValue
        except Exception:
            pass
        return None

    def set_gain_auto(self, enable: bool) -> bool:
        """开关自动增益。enable=True -> Continuous, enable=False -> Off."""
        self._require_cam()
        try:
            ret = self._cam.MV_CC_SetEnumValue("GainAuto", 2 if enable else 0)
            return ret == 0
        except Exception:
            return False

    def setup_auto_exposure(self, target_brightness=128, exposure_limit_ms=50, gain_limit_db=12) -> bool:
        """
        启用硬件自动曝光 + 自动增益，并设置上下限约束。
        返回 True 表示全部成功（失败项仅记录日志不阻塞）。
        """
        self._require_cam()
        ok = True

        try:
            self._cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        except Exception:
            logger.warning("Failed to enable ExposureAuto")
            ok = False

        try:
            self._cam.MV_CC_SetEnumValue("GainAuto", 2)
        except Exception:
            logger.warning("Failed to enable GainAuto")
            ok = False

        for param, val in [
            ("AutoExposureTimeLowerLimit", 10.0),
            ("AutoExposureTimeUpperLimit", float(exposure_limit_ms) * 1000),
            ("AutoGainLowerLimit", 0.0),
            ("AutoGainUpperLimit", float(gain_limit_db)),
        ]:
            try:
                self._cam.MV_CC_SetFloatValue(param, val)
            except Exception:
                logger.debug("Failed to set %s (may not be supported)", param)

        try:
            self._cam.MV_CC_SetFloatValue("AutoTargetBrightness", float(target_brightness))
        except Exception:
            logger.debug("AutoTargetBrightness not supported on this model")

        return ok

    def set_ae_roi(self, x: int, y: int, w: int, h: int) -> bool:
        """
        设置自动曝光 ROI 区域（像素坐标）。w<=0 或 h<=0 时清除 ROI。

        尝试多种 GenICam 参数名称以兼容不同 Hikvision 型号。
        """
        self._require_cam()
        clear = (w <= 0 or h <= 0)

        param_sets = [
            ("AEAreaSelector", "AEAreaWidth", "AEAreaHeight",
             "AEAreaOffsetX", "AEAreaOffsetY"),
            ("AutoExposureROISelector", "AutoExposureROIWidth",
             "AutoExposureROIHeight", "AutoExposureROIOffsetX",
             "AutoExposureROIOffsetY"),
        ]

        for selector, wk, hk, oxk, oyk in param_sets:
            try:
                self._cam.MV_CC_SetEnumValue(selector, 0)
                if clear:
                    self._cam.MV_CC_SetIntValue(wk, 0)
                    self._cam.MV_CC_SetIntValue(hk, 0)
                    self._cam.MV_CC_SetIntValue(oxk, 0)
                    self._cam.MV_CC_SetIntValue(oyk, 0)
                else:
                    self._cam.MV_CC_SetIntValue(wk, int(w))
                    self._cam.MV_CC_SetIntValue(hk, int(h))
                    self._cam.MV_CC_SetIntValue(oxk, int(x))
                    self._cam.MV_CC_SetIntValue(oyk, int(y))
                logger.debug("AE ROI set: x=%d y=%d w=%d h=%d (via %s)",
                             x, y, w, h, selector)
                return True
            except Exception:
                continue

        # 相机不支持 AE ROI, 回退: 暂时关闭 AE 手动调到目标亮度
        logger.debug("AE ROI not supported, falling back to manual exposure servo")
        return False

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics
