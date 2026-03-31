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
