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
