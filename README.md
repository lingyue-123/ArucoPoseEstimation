# ArucoPoseEstimation

通过手眼标定和相机位姿估计，实现从ArUco码在相机坐标系下的位姿转换到机器人基坐标系下的位姿。

## 主要功能

### 1. 手眼标定矩阵计算
- **get_eye2base_matrix**: 计算相机（眼）在机器人基坐标系下的4×4齐次矩阵
- 输入参数：
  - `hand_xyzrpy`: 机械手在基坐标系下的位姿 [x, y, z, rx, ry, rz]
  - `eye2hand`: 手眼标定矩阵（眼到手的转移矩阵）
- 输出：相机在基坐标系下的齐次矩阵

### 2. ArUco码位姿转换
- **get_ar2base_matrix**: 计算ArUco码在机器人基坐标系下的4×4齐次矩阵
- 输入参数：
  - `ar2cam_xyzrpy`: ArUco码在相机坐标系下的位姿 [x, y, z, rx, ry, rz]，描述ArUco码坐标系到相机坐标系的转换
  - `cam2base`: 相机在基坐标系下的4×4齐次矩阵
- 输出：ArUco码在基坐标系下的齐次矩阵

## 关键概念说明

- **手眼标定矩阵**: 是眼到手的转移矩阵，表示相机坐标系相对于机械手坐标系的变换关系
- **ar2cam**: ArUco码在相机坐标系下的位姿，描述ArUco码坐标系到相机坐标系的转换关系

## 依赖库

- numpy
- scipy

## 使用方法

运行 `python aruco2base.py` 查看示例计算结果。