# ArucoPoseEstimation


## 硬件环境

- 平台：NVIDIA Jetson (ARM64 / Tegra)
- 机械臂：JAKA 协作机器人 / Modbus TCP 机械臂
- 操作系统：Linux (Ubuntu)

## 机械臂驱动选择

支持两种机械臂驱动，在 `config/robot.yaml` 中同时配置，运行时按名选择：

| 驱动 | IP | 说明 | 依赖 |
|------|-----|------|------|
| `modbus` | 192.168.1.133 | Modbus TCP 通用机械臂 | pymodbus |
| `jaka` | 192.168.1.106 | JAKA 协作机器人 | jkrc SDK |

`robot.yaml` 中通过 `default_driver` 设置默认驱动（当前为 `modbus`），每个驱动有独立的 IP、端口、SDK 路径和网络配置。运行时可通过命令行 `--robot-driver` 覆盖：

```bash
# 使用配置文件默认驱动（当前为 modbus）
python scripts/collect_aruco_poses.py --camera hikvision_normal

# 命令行指定 JAKA 驱动
python scripts/collect_aruco_poses.py --camera hikvision_normal --robot-driver jaka

# 命令行指定 Modbus + 自定义 IP
python scripts/collect_aruco_poses.py --camera hikvision_normal --robot-driver modbus --robot-ip 192.168.1.200

# 不连接机械臂
python scripts/collect_aruco_poses.py --camera hikvision_normal --no-robot
```

优先级：**命令行 `--robot-driver` / `--robot-ip` > `config/robot.yaml`**。

## 网络拓扑与自动配置

系统运行需要两个以太网接口分别连接相机和机械臂。网络配置**已集成到代码中**，运行脚本时自动完成，无需手动执行 shell 脚本。

| 接口 | 本机 IP | 目标设备 | 目标 IP | 配置来源 |
|------|---------|----------|---------|----------|
| eth0 | 192.168.1.132/24 | Modbus 机械臂 | 192.168.1.133 | `config/robot.yaml` |
| eth1 | 192.168.1.100/24 | 海康威视相机 | 192.168.1.64 | `config/cameras.yaml` |

两个接口处于同一子网 `192.168.1.x`，通过 host route (`/32`) 区分流量走向。

自动配网流程（`robovision/network.py` → `ensure_interface()`）：
1. `ip link set <iface> up` — 启用接口
2. `ip addr add` — 分配本机 IP（已存在则跳过）
3. `ip route replace <target>/32 dev <iface>` — 添加 host route
4. 可选：删除默认路由（防止 VPN 冲突）
5. `ping -c 1` — 验证可达性

> **注意**：需要 `sudo` 权限执行网络配置命令。

## 手眼标定流程

**配置**：Eye-in-Hand（相机固定在机械臂末端），求解相机→末端的固定变换 `T_cam2gripper`。
**算法**：支持 OpenCV 全部 5 种闭式方法（TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS），可自动选最优；支持迭代异常点剔除、Levenberg-Marquardt 非线性精炼、留一法诊断。最少需要 15 组配对姿态数据。

### 第 1 步：`collect_handeye_data.py` — 采集配对数据

连接相机与机械臂，将棋盘格固定在桌面，手动移动机械臂到不同姿态后按 `s` 同步保存：

```bash
python scripts/collect_handeye_data.py --camera hikvision_normal
```

| 按键 | 操作 |
|------|------|
| `s` | 保存当前棋盘格位姿 + 机械臂 TCP（需同时检测到棋盘格）|
| `c` | 清空已保存数据重新采集 |
| `q` / `ESC` | 退出 |

采集结果：
- `Handeyecalib/board_to_cam.txt` — 棋盘格→相机位姿（米，XYZ内旋）
- `Handeyecalib/robot_tcp.txt` — 机械臂末端→基座位姿（JAKA TCP 原始单位）

### 第 2 步：`run_handeye_calib.py` — 计算标定结果

```bash
# 基本用法（默认 TSAI 方法，与旧版行为一致）
python scripts/run_handeye_calib.py

# 全量优化：多方法对比 + 异常点剔除 + 非线性精炼
python scripts/run_handeye_calib.py --full

# 仅多方法对比（自动选最优）
python scripts/run_handeye_calib.py --compare-methods

# 指定方法 + 异常点剔除（阈值 3mm）
python scripts/run_handeye_calib.py --method horaud --outlier-rejection --max-error 3.0

# 留一法分析（定位坏点）
python scripts/run_handeye_calib.py --loo

# 自定义数据路径
python scripts/run_handeye_calib.py \
    --board Handeyecalib/board_to_cam.txt \
    --robot Handeyecalib/robot_tcp.txt \
    --output Handeyecalib/hand_eye_result.txt
```

#### 优化参数

| 参数 | 说明 |
|------|------|
| `--compare-methods` | 尝试所有 5 种方法，按误差排序，自动选最优 |
| `--method NAME` | 指定方法（tsai/park/horaud/andreff/daniilidis）|
| `--outlier-rejection` | 迭代剔除误差最大的位姿，直到所有点误差 < 阈值 |
| `--max-error MM` | 剔除阈值（默认 5.0 mm）|
| `--min-poses N` | 最少保留组数（默认 8），防止过度剔除 |
| `--refine` | 在闭式解基础上用 Levenberg-Marquardt 非线性优化 |
| `--loo` | 留一法分析：逐个去掉每组数据，看对误差的影响 |
| `--full` | 等效于 `--compare-methods --outlier-rejection --refine` |

#### 优化流程

使用 `--full` 时的处理流程：

1. **多方法对比**：5 种方法各跑一次，按误差排序，选最优方法
2. **异常点剔除**：用最优方法迭代标定，每轮剔除误差最大的点（> 阈值）
3. **非线性精炼**：在闭式解基础上用 LM 优化 6 个参数（Rodrigues 3 + 平移 3），若精炼后误差反而变大则自动放弃
4. **结果报告**：输出方法对比表、逐点误差、剔除摘要

#### 留一法分析

`--loo` 用于诊断坏点：依次去掉每组数据重新标定，若去掉某点后误差显著下降（delta < -0.5 mm），说明该点是坏点。N=20 时仅需 20 次标定（每次 <1ms），完全可行。

输出结果包含：
- `cam2gripper` 平移向量（mm）和旋转矩阵（XYZ内旋欧拉角）
- **误差分析**：用 `T_base_target = T_g2b × T_c2g × T_board2cam` 检验各姿态下棋盘格在基座坐标系的位置一致性
- **多方法对比表**（使用 `--compare-methods` 时）：排名、方法名、误差、质量
- **逐点误差表**：各位姿的误差，标记异常点
- **异常点剔除摘要**（使用 `--outlier-rejection` 时）

标定质量参考标准：

| 平均误差 | 等级 |
|----------|------|
| < 2 mm | 优秀 |
| < 5 mm | 良好 |
| < 10 mm | 一般 |
| ≥ 10 mm | 较差（建议重新采集）|

结果保存为 `Handeyecalib/hand_eye_result.txt`（4×4 齐次变换矩阵）。

## 自动手眼采样

`auto_handeye_collect.py` 现在不再依赖写死的绝对轨迹点，而是围绕一个给定的 TCP 中心位姿自动生成采样点。

```bash
# 用当前 TCP 作为采样中心
python scripts/auto_handeye_collect.py --camera hikvision_normal --robot-driver modbus

# 显式指定采样中心
python scripts/auto_handeye_collect.py \
    --camera hikvision_normal \
    --robot-driver modbus \
    --start-pose -350 750 350 -177.5 -1.6 89.5

# 先看采样点，不实际运动
python scripts/auto_handeye_collect.py \
    --camera hikvision_normal \
    --robot-driver modbus \
    --start-pose -350 750 350 -177.5 -1.6 89.5 \
    --dry-run
```

默认采样策略：

- 空间范围：围绕中心位姿形成约 `200mm x 200mm x 130mm` 的采样块
- 位置模板：中心点、6 个轴点、4 个 XY 角点、4 个补点
- 姿态模板：原姿态、`Rx+`、`Rx-`、`Ry+`、`Rz-`
- 总采样点数：默认 51 个

## 精度验证

### 离线一：`verify_aruco_motion.py` — ArUco 检测精度（与手眼解耦）

自动从 TCP 数据中检测**纯平移段**（角度不变）和**纯旋转段**（位置不变），
每段内部做 C(n,2) 全配对比较。不同段之间不跨组配对。
**与手眼标定完全解耦**，单独反映 ArUco 检测精度。

支持混合数据：同一文件中既有纯平移数据也有纯旋转数据，甚至多组不同角度下的纯平移，
脚本会自动识别并分组独立验证。

```bash
python scripts/verify_aruco_motion.py
python scripts/verify_aruco_motion.py \
    --aruco-file aruco_pose_id0.txt \
    --tcp-file robot_tcp_id0.txt -v
```

| 参数 | 说明 |
|------|------|
| `--aruco-file` | ArUco 位姿文件（默认 `aruco_pose_id0.txt`）|
| `--tcp-file` | 与 aruco-file 行对行对齐的 TCP 文件（默认 `robot_tcp_id0.txt`）|
| `--angle-tol` | 角度变化容差（度，默认 0.05），相邻帧角度差 < 此值视为纯平移 |
| `--pos-tol` | 位置变化容差（米，默认 0.0005），相邻帧位移 < 此值视为纯旋转 |
| `--marker-id` | Marker ID，仅用于日志提示（默认 0）|
| `-v` | 打印每对配对的详细信息 |

误差定义：
- **平移**：`|d_tool - d_cam|`（mm），C(n,2) 全配对
- **旋转**：`|theta_tool - theta_cam|`（°），C(n,2) 全配对

### 离线二：`verify_pose_chain.py` — 全链路联合精度

利用配对数据 `(aruco_i, tcp_i)` 和手眼标定结果预测 `tcp_j`，与真值对比。
反映**手眼标定 + ArUco 检测的联合误差**。

```bash
python scripts/verify_pose_chain.py
python scripts/verify_pose_chain.py \
    --aruco-file aruco_pose_id0.txt \
    --tcp-file robot_tcp_id0.txt \
    --hand-eye Handeyecalib/hand_eye_result.txt
```

| 参数 | 说明 |
|------|------|
| `--aruco-file` | ArUco 位姿文件（默认 `aruco_pose_id0.txt`）|
| `--tcp-file` | 与 aruco-file 行对行对齐的 TCP 文件（默认 `robot_tcp_id0.txt`）|
| `--hand-eye` | 手眼标定结果（默认 `Handeyecalib/hand_eye_result.txt`）|
| `--marker-id` | Marker ID，仅用于日志提示（默认 0）|

配对方式：全组合 N*(N-1)/2 对（充分利用所有数据）。

## 项目结构

```
robovision/          # 核心库
├── cameras/         # 相机抽象层
├── detection/       # ArUco / 棋盘格 / 黑色方块检测
├── tracking/        # 卡尔曼滤波 / 位姿平滑
├── geometry/        # 坐标变换工具
├── calibration/     # 内参 / 手眼标定算法
├── robot/           # 机械臂接口（JAKA / Modbus）+ build_robot() 工厂
├── servo/           # PBVS 视觉伺服（步进计算 / 会话管理 / OSD 显示）
├── network.py       # 网络接口自动配置
├── io/              # 文件读写工具
├── visualization/   # 位姿可视化
└── config/          # 配置加载器

config/              # YAML 配置（相机内参、标记尺寸、检测参数）
scripts/             # 入口脚本（替代原有各独立脚本）

```

## 数据文件格式

以下格式完全向后兼容，已有数据集可直接使用：

| 文件 | 格式 |
|------|------|
| `intrinsic_calib/intrinsic.txt` | Tab分隔，两组相机参数 |
| `robot_tcp.txt` | CSV：`x,y,z,rx,ry,rz`（米，度，XYZ内旋）|
| `aruco_pose_id{N}.txt` | CSV：`x,y,z,rx,ry,rz`（米，度，XYZ内旋）|
| `Handeyecalib/hand_eye_result.txt` | 手眼标定结果 |
| `Handeyecalib/board_to_cam.txt` | 标定板到相机位姿 |

## 坐标约定

- **机械臂 TCP 位姿**：`[x(m), y(m), z(m), rx(°), ry(°), rz(°)]`，XYZ内旋（JAKA格式）
- **相机位姿**：`[tx(m), ty(m), tz(m), rx(°), ry(°), rz(°)]`，XYZ内旋
- **手眼标定输出**：`cam2gripper`（相机到机械臂末端的变换）

## 依赖环境
- **mech相机**：mecheye 
    - 安装方式：pip install MechEyeApi
    - 注意不要加sudo，否则会强制安装到系统Python环境中，而不是当前的conda环境；此外MechEyeApi包只适用python版本为3.7至3.11