# Charging Unplug
磁吸方案取枪分支

## 硬件平台

- 机械臂：jaka
- 相机：mech、海康

## 环境配置
- **mech相机**：mecheye 
    - 安装方式：pip install MechEyeApi
    - 注意不要加sudo，否则会强制安装到系统Python环境中，而不是当前的conda环境；此外MechEyeApi包只适用python版本为3.7至3.11

- **jaka python sdk**: jkrc
    - jkrc 不是 PyPI 公开包，无法用 pip install jkrc 直接安装。它是节卡（JAKA）机器人官方 Python SDK 的核心动态库模块，必须从官方 SDK 压缩包中手动配置安装。
    - 配置方式一： **jkrc.so** 和 **libjakaAPI.so** 路径到某个python的用户库：
        ```
        # 1. 自动创建用户库目录（不存在就新建）
        mkdir -p $(python -m site --user-site)
        # 2. 自动生成 jaka.pth，写入SDK路径
        echo "/home/nvidia/Downloads/jaka-python-sdk" > $(python -m site --user-site)/jaka.pth
        # 3. 刷新系统动态库（关键）
        sudo ldconfig
        ```
    - 配置方式二：代码中在线配置：
        ```
        JAKA_SDK_PATH = "/home/nvidia/Downloads/jaka-python-sdk"

        # 1. 配置 Linux 动态库环境（让系统找到 libjakaAPI.so）
        if sys.platform.startswith("linux"):
            os.environ["LD_LIBRARY_PATH"] = f"{JAKA_SDK_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"

        # 2. 让 Python 找到 jkrc.so 模块
        sys.path.insert(0, JAKA_SDK_PATH)

        import jkrc
        print('JAKA SDK imported successfully.')
        ```

## JAKA 机械臂沿法兰 Z 轴移动时的矩阵逻辑：
- 目标位姿应由当前 TCP 位姿齐次矩阵右乘局部 Z 轴平移矩阵得到：`T_new = T_current @ T_z`

- 这样表示“先到当前工具位姿，再沿工具坐标系 Z 轴移动”

- 如果写成 `T_new = T_z @ T_current`，则表示“先在基坐标系中平移，再变换到当前工具位姿”，不符合沿法兰 Z 轴插入/反插入的语义

## 夹爪按键操作流程说明

| 动作 | 按键 |
|------|------|
| 开盖 | `g` |
| 取枪 | `m` -> `a` -> `c` |
| 插枪 | `h` -> `m` -> `5` -> `e` -> `s` |
| 归枪 | `j` |
| 关盖 | `k` |

> **说明**：当前操作无力控、无鱼眼、无双目。

## 户外光照鲁棒：自动曝光 + 亮度收敛

### 两层 AE 策略

| 层 | 机制 | 响应 | 作用 |
|---|------|------|------|
| L1 硬件AE | 相机 `ExposureAuto=Continuous` + `GainAuto=Continuous` | 逐帧实时 | 处理快速光照变化 |
| L2 ROI锁定 | 检测到 marker 后锁 AE 曝光区域到 marker bbox；每 15 帧 nudge 曝光上限 | 逐帧/秒级 | 防止背景干扰 + 引导硬件AE |

### 亮度收敛（对应 `r` / `m` 按键流程）

| 操作 | 行为 |
|------|------|
| 按 `r` | 录制位姿参考 + 同时保存当前 `(exposure_time, gain, marker_roi_brightness)` 到 `data/aruco/aruco_exp_ref.txt` |
| 按 `m` | 先锁定曝光参数到录制值 → 亮度收敛循环（≤0.6s, 只调曝光时间不调增益）→ 执行对准 → 恢复硬件AE |

### 曝光参考文件

| 文件 | 用途 |
|------|------|
| `data/aruco/aruco_exp_ref.txt` | 插枪 marker 曝光参考 |
| `data/aruco/aruco_exp_ref_takegun.txt` | 取枪 marker 曝光参考 |

格式：`exposure_time_us,gain_db,marker_roi_brightness`

### 相关参数配置

在 `config/cameras.yaml` 中各相机可配置 `auto_exposure` 节：

```yaml
auto_exposure:
  enabled: true
  target_brightness: 128       # 目标灰度中值(0-255)
  deadband: 12                 # 亮度死区(±灰度)
  adjust_interval: 15          # 软件调整间隔(帧)
  exposure_limit_ms: 50.0      # 最大曝光时间(防运动模糊)
  gain_limit_db: 12.0          # 最大增益(防噪声)
```

命令行 `--no-ae` 可禁用所有自动曝光功能。

### 底层 API（仅 Hikvision 实现）

| 方法 | 说明 |
|------|------|
| `camera.set_exposure_auto(True/False)` | 开关硬件AE |
| `camera.set_exposure_time(us)` | 手动曝光时间（微秒） |
| `camera.get_exposure_time()` | 读取当前曝光 |
| `camera.set_gain(db)` | 手动增益（dB） |
| `camera.get_gain()` | 读取当前增益 |
| `camera.set_gain_auto(True/False)` | 开关自动增益 |
| `camera.set_ae_roi(x, y, w, h)` | 设置 AE ROI 区域 |
| `camera.setup_auto_exposure(target, limit_ms, limit_db)` | 一键启用硬件AE+上下限 |

## ArUco 光照鲁棒性实验 (`aruco_lighting_robustness`)

### 核心目标

测量外部光照变化下 ArUco 位姿估计 (`[M->C]` = Marker→Camera) 的稳定性。仅连接相机，不连接机械臂。

### 运行方式

```bash
python scripts/aruco_lighting_robustness.py      --camera mecheye
python scripts/aruco_lighting_robustness_v2.py   --camera mecheye --lighting-robust
```

### 两个版本对比

| | v1 | v2 |
|---|----|----|
| 文件 | `aruco_lighting_robustness.py` | `aruco_lighting_robustness_v2.py` |
| 光照鲁棒线程 | 只调增益 | **增益优先 → 曝光兜底** |
| 增益上限 | 12 dB | 12 dB |
| 曝光调整 | 无 | 乘法步进 5%，范围 0.1ms~100ms |
| 增益到顶容差 | 无（会卡住不动曝光） | `GAIN_LIMIT_DB - LR_GAIN_STEP`（12 - 0.4 = 11.6） |

### 三个线程

| 线程 | 循环间隔 | 职责 |
|------|---------|------|
| `_detection_loop` | 逐帧 | 读帧 → ArUco 检测 → `latest_det` |
| `_lighting_robust_loop`（可选，`--lighting-robust`） | 0.15 s | 对比 ROI 亮度 vs 参考 → 调增益/曝光 |
| 主线程 | 逐帧 | 可视化 + 键盘交互 |

### ArUco 检测流水线

两种模式共享同一套底层库：

- **标准模式**（默认）：`ArucoDetector` — CLAHE + 高斯模糊 + 形态学 → 降分辨率粗检测 → 映射回全分辨率角点 → 卡尔曼角点滤波 → subpix 精化 → **多方法 PnP 择优**（`IPPE_SQUARE` / `ITERATIVE`）→ SLERP 旋转平滑 + EMA 平移平滑 → 异常门控（`WARN`/`HOLD`）
- **RAW 模式**（`--raw`）：`detect_raw_frame` — CLAHE → `detectMarkers` → subpix → 固定 `SOLVEPNP_IPPE_SQUARE`，无时序滤波

### 可视化窗口含义

#### 图像叠加层
| 元素 | 含义 |
|------|------|
| marker 彩色边框 | 青色=正常, 橙色=WARN(`!`), 红色=HOLD(异常锁定) |
| 四个角点 + 坐标值 | 检测到的亚像素角点像素坐标 |
| `[M->C] t(mm): X Y Z` | Marker→Camera 平移（mm） |
| `[M->C] Euler(ZYX): rx ry rz` | Marker→Camera 欧拉角（ZYX 内旋，deg） |
| `reproj=...px` | 重投影误差，PnP 解算质量指标 |

#### 文本叠加行（从上到下）
| 行 | 含义 |
|----|------|
| `ID{x} [M->C]: t=... Euler=...` | 当前帧 [M->C] 完整位姿 |
| `Err vs Ref: d=(dx,dy,dz) mm \|trans\|=... rot=...` | **当前 [M->C] vs 按 r 时的参考 [M->C]**，误差在参考 marker 坐标系下分解。相机和 marker 均不动时，任何偏离即光照导致的漂移。颜色分级: 绿<2mm, 黄<10mm, 红≥10mm |
| `ROI Bri: ID{x}=...` | marker 区域（角点外扩 20px）中值亮度 0-255 |
| `Bri Ref: ID{x}=ref d=±...` | 参考亮度值 + 当前偏差 |
| `Exposure/Gain` | 相机当前曝光时间和增益 |
| 底部状态栏 | 检测模式 + Kalman + ArUco 可见数 + 参考状态 + 光照鲁棒状态(`LR`) + 快捷键 |

### 光照鲁棒线程算法

#### v2 两级调整（`aruco_lighting_robustness_v2.py:286-320`）

```
每 0.15s:
  取多 marker 的 ROI 亮度偏差均值 avg_dev
  if |avg_dev| ≤ LR_DEADBAND (1) → 不调整
  elif avg_dev < 0（偏暗）:
      if cur_gain < GAIN_LIMIT_DB - LR_GAIN_STEP (11.6):
          gain += 0.4 dB
      else:
          exp ×= 1.05 (上限 100ms)
  else（偏亮）:
      if cur_gain > LR_GAIN_STEP (0.4):
          gain -= 0.4 dB
      else:
          exp ×= 0.95 (下限 0.1ms)
```

#### v1 单级调整（`aruco_lighting_robustness.py:281-296`）

```
每 0.3s:
  avg_dev 超出死区 → gain ±0.4 dB（上限 12 dB，下限 0）
  不调整曝光时间
```

### 光照鲁棒超参数

| 常量 | v1 | v2 | 含义 |
|------|----|----|------|
| `LR_DEADBAND` | 1 | 1 | 亮度死区（±灰度值） |
| `LR_GAIN_STEP` | 0.4 | 0.4 | 增益每步调整量（dB） |
| `GAIN_LIMIT_DB` | 12 | 12 | 增益上限（dB） |
| `LR_EXP_LIMIT_US` | — | 100,000 | 曝光上限（μs） |
| `LR_EXP_MIN_US` | — | 100 | 曝光下限（μs） |
| `LR_EXP_RATIO` | — | 0.05 | 曝光每步比例（5%） |
| 调整间隔 | 0.3 s | 0.15 s | 循环间隔 |

### 键盘操作

| 按键 | 功能 |
|------|------|
| `r` | 保存当前帧所有 marker 的 `[M->C]` 4×4 矩阵 + ROI 亮度为参考，开始显示误差 |
| `c` | 清除所有参考 |
| `s` | 记录当前帧 `[M->C]` + reproj + 亮度 + 偏差到终端（限频 `--record-interval` 秒） |
| `q` / ESC | 退出 |

### 命令行参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `--camera` | None | `cameras.yaml` 中的相机名 |
| `--marker-ids` | `0,1` | 监测的 marker ID 列表 |
| `--raw` | False | 使用 RAW 检测模式 |
| `--no-temporal-filter` | False | 关闭卡尔曼 + 时序平滑 |
| `--lighting-robust` | False | 启用光照鲁棒线程 |
| `--record-interval` | 1.0 | `s` 键最小记录间隔（秒） |
| `--debug` | False | 调试日志 |
