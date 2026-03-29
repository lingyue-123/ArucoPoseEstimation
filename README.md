# 相机内参标定

这是一个相机标定的分支，基于OpenCalib的内参标定工具。

## 编译

```bash
mkdir build
cd build
cmake ..
make -j8
```

## 使用

```bash
./bin/run_intrinsic_calibration xxx
```

其中 `xxx` 为相机拍摄图像的路径。