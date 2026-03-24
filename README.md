# ArucoPoseEstimation

## 免责声明

本分支提供一个极简版 ArUco 目标 TCP 位姿计算逻辑参考，旨在帮助将算法迁移到 C++ 实现。当前代码不可直接用于生产或精确控制：

- 将产生错误的结果（示例中 `tcp_cur` 设为全零，未连接实际机械臂姿态）
- 相机内参 `CAMERA_MATRIX` / 畸变 `DIST_COEFFS` / 分辨率 / 手眼标定矩阵 `T_C2G` 均硬编码为 Mech 相机与当前手眼标定结果
- 仅读取单帧图像（No 循环视频输入、No 机械臂运动命令输出、No 视觉可视化）

本分支主要用于方案验证和 C++ 移植参考，尤其面向 JAKA 机械臂的「Ar 码取枪」流程。

## 核心逻辑梳理

1. `target_tcp_pose_clean.py`:
   - 入口脚本，接受 `--image` 参数及可选 `--raw`（原生 OpenCV ArUco 识别）
   - 读取图像并查找 `TARGET_MARKER`（默认 0），命中后得到 marker->camera 变换
   - `T_aruco2cam_ref` 由预设参考 Marker 全局位姿 `REF_POSE_XYZRPY` 生成
   - 计算当前相机到 marker 的变换 `T_aruco2cam_cur`
   - 计算机器人当前 TCP 位姿 `T_g2b_cur`（示例中全零，实际需从 JAKA 读）
   - 基于手眼标定矩阵 `T_C2G` 和参考姿态计算目标机器人位姿 `T_g2b_target`
   - 输出 `final_pose`（`[x,y,z,rx,ry,rz]`，单位 mm / deg）

2. `robovision/detection/aruco.py`:
   - 通过 `ArucoDetector` 实现鲁棒检测（CLAHE、形态学、子像素、卡尔曼、平滑等）
   - 返回每个 marker 的 `R_m2c`、`tvec_m2c`、`reproj_err` 等位姿信息

3. `robovision/geometry/transforms.py`:
   - 位姿与矩阵互转：`pose_to_matrix`, `matrix_to_pose`
   - 计算手眼变换：`compute_new_tool_pose`

## C++ 移植要点

- 输入：单帧图像 + 相机内参 + 手眼标定矩阵（可由标定程序获取）
- 先 detect ArUco, 再 solvePnP（或纯 `ArucoDetector`）获得 `T_aruco2cam_cur`
- 计算 `base_from_target = base_from_tool_old * tool_from_cam * cam_from_target_old`
- 目标位姿：`T_g2b_target = base_from_target * inv(cam_from_target_new) * inv(tool_from_cam)`
- 输出 `final_pose` 供 JAKA 指令系统执行

## 适用场景

- JAKA 机械臂进行 ArUco 取枪（目标 marker 姿态约定明确定义）
- 预研算法与过程验证，非线上控制最终产线


> 注意：运行仅用于算法调试，结果仅供参考，必须结合实机调试与标定修正。
> 注意：本代码计算得到的 target_tcp 中的rx ry rz均为角度，后续jaka SDK api需求的是弧度，在调用前需进行角度->弧度转换。