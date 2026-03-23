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

- **手眼标定矩阵**: 是眼到手的转移矩阵，表示相机坐标系相对于末端法兰坐标系的变换关系
- **ar2cam**: ArUco码在相机坐标系下的位姿，描述ArUco码坐标系到相机坐标系的转换关系

## Rotation包欧拉角变换说明

项目中使用scipy.spatial.transform.Rotation包进行旋转矩阵的计算和转换。以下是对R.from_euler()函数及其欧拉角变换的详细说明：

### R.from_euler()函数

`R.from_euler(seq, angles, degrees=False)` 用于从欧拉角创建旋转对象，其中：
- `seq`: 旋转顺序字符串，如'xyz'、'ZYX'等，表示旋转轴的顺序，小写表示内旋/大写表示外旋
- `angles`: 旋转角度数组，对应seq中每个轴的角度
- `degrees`: 是否以度为单位（默认False，使用弧度）

scipy中的R.from_euler()默认使用**外旋（extrinsic）**约定，即每次旋转都相对于固定坐标系进行。


### 欧拉角的对偶性

欧拉角具有对偶性，即**内旋'xyz'等价于外旋'zyx'**（对于相同的角度值）。这意味着：
- 使用内旋'xyz'的旋转等价于使用外旋'zyx'的旋转
- 这种对偶性源于旋转群的性质，可以通过改变旋转顺序和内/外旋约定来实现等价变换

在实际应用中，根据具体需求选择合适的旋转约定，以确保位姿计算的正确性。

## 依赖库

- numpy
- scipy

## 使用方法

运行 `python aruco2base.py` 查看示例计算结果。