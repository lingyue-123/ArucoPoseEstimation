================================================================================
  CrobotpOS SDK — Python Wrapper 实现总结
  日期：2026-02-27
================================================================================

一、原始 SDK 实现了什么功能
────────────────────────────────────────────────────────────────────────────────

原始 SDK（成都柔博 CrobotpOS）以一个共享库 libRobotService.so 的形式发布，
内部通过 C++ 纯虚接口（vtable-based）暴露五大服务：

1. IRobotService（机器人基础服务）
   - 连接 / 断开控制柜 (connect / disconnect)
   - 伺服上下电 (servoPowerOn / servoPowerOff)
   - 钥匙档位 / 坐标系 / 速度倍率读写
   - 获取当前关节位置与笛卡尔位置
   - 启动 / 停止 / 恢复程序
   - 报警查询与清除，软急停
   - 全局变量：GI（整型）、GR（浮点型）、UI（短整型）
   - 位置变量：GP（笛卡尔）、GJ（关节）
   - 手动点动 (jogMoveJ / jogMoveL)、MoveJ / MoveL（手动模式）
   - 协作模式开关 (enableCobotMode)
   - 试运行模式、自动运行模式

2. IIOService（IO 读写服务）
   - 数字输入 X / 数字输出 Y / 辅助继电器 M
   - 系统 IO：SX / SY / SM
   - 模拟输入 AIN / 模拟输出 AOT
   - 组输入 GIN / 组输出 GOT

3. IMotionService（引导运动服务）
   - 路径引导（位置流，每 2ms 一帧）：sendPath(joint) / sendPath(cartesian)
   - 指令引导：MoveAbsJ / MoveJ / MoveL / MoveC / MoveJump
   - 缓冲区管理 (getMaxPathBufferSize / getAvailPathBufferSize)
   - finalize() 结束引导，切换到程序下一行

4. IModelService（运动学服务）
   - 正解 FKine（离线，不含工具/用户偏置）
   - 反解 IKine（离线）
   - joint2Position / position2Joint（在线，需控制器联网）
   - convertCoordSys（坐标系间转换）
   - calcCfg（计算 CFG 构型配置）

5. IFileService（文件服务）
   - 上传 / 下载程序文件
   - mkdir / rmdir / rename / copy / remove / exists

SDK 的加载器 CSDKLoader 通过 dlopen + createInterface("UUID") 工厂模式
动态创建上述接口对象，返回 C++ 虚类指针。

================================================================================

二、Python Wrapper 的设计思路
────────────────────────────────────────────────────────────────────────────────

Python 无法直接调用 C++ 虚函数（vtable 调用约定、名称修饰等问题），
因此采用三层架构：

    libRobotService.so        (原始 SDK，C++ 虚接口)
           ↓
    crp_robot/robot_bridge.so (C 桥接层，本次编译)
           ↓
    crp_robot/ Python 包      (ctypes 调用桥接层)
           ↓
    用户 Python 代码

──── 第一层：robot_bridge.cpp ────

将所有 C++ 虚方法包装成 extern "C" 的普通 C 函数，命名规则：
  crp_<服务>_<方法>

核心 Context 结构体保存所有服务指针：

  struct BridgeCtx {
      Crp::CSDKLoader*     loader;
      Crp::IRobotService*  robot;
      Crp::IIOService*     io;
      Crp::IMotionService* motion;
      Crp::IModelService*  model;
      Crp::IFileService*   file;
  };

生命周期管理：
  void* crp_create(const char* so_path)  → new BridgeCtx + loader.initialize()
  void  crp_destroy(void* ctx)           → loader.deinitialize() + delete

结构体扁平化（Python 友好）：
  SRobotPosition (x,y,z,Rx,Ry,Rz + extJoint[6] + cfg[4])
    → 拆成  double pos6[6]  + double ext[6]  + int cfg[4]

  SJointPosition (body[6] + ext[6] + cfg[4])
    → 拆成  double body[6]  + double ext[6]  + int cfg[4]

  SStdDHParam (a1..a7, d1..d7)
    → 拆成  double dh14[14]

共导出 111 个 C 符号。

──── 第二层：_lib.py ────

使用 ctypes.CDLL 加载 robot_bridge.so，并为每个函数声明
  .argtypes / .restype
确保 Python 调用时类型安全。

辅助函数：d6() / i4() / d14() / dn() 创建对应 ctypes 数组。

──── 第三层：Python 类 ────

  _types.py    → RobotPosition / JointPosition / MotionParam / DHParam (dataclass)
  _enums.py    → RobotMode / CoordSystem / ProgramStatus / MotionType 等 (IntEnum)
  robot.py     → Robot 类（顶层入口，context manager 支持）
  io_service.py      → IOService
  motion_service.py  → MotionService
  model_service.py   → ModelService
  file_service.py    → FileService

Robot 类通过懒加载属性暴露子服务：

  robot.io.set_y(0, True)
  robot.motion.move_abs_j(0, joint, param)
  robot.model.fkine(dh, joints)
  robot.file.upload("/local/prog.pro", "/robot/program/prog.pro")

================================================================================

三、过程中遇到的问题及解决方案
────────────────────────────────────────────────────────────────────────────────

──── 问题 1：libRobotService.so 无法被 dlopen 加载 ────

现象：
  OSError: undefined symbol: __libc_single_threaded

原因：
  系统 glibc 版本为 2.31（Ubuntu 20.04，ARM Tegra）。
  libRobotService.so 编译时依赖 glibc 2.32+ 引入的
  __libc_single_threaded 全局变量（pthread 优化用）。
  dlopen 时符号解析失败，即使使用 RTLD_LAZY 也不例外
  （该符号在库的静态初始化阶段即被引用）。

奇怪现象：
  SDK 自带的示例二进制（如 it_robotpos）可以正常运行，
  因为它们同样通过 dlopen 加载 SDK，但这些二进制在编译时
  已经由链接器注入了这个符号的定义（或通过其他方式提供）。

解决方案（两步）：

  步骤 1 — 在 robot_bridge.cpp 中提供弱符号定义：
    extern "C" {
        int __libc_single_threaded __attribute__((weak)) = 0;
    }

  步骤 2 — 在 _lib.py 中以 RTLD_GLOBAL 模式加载桥接库：
    lib = ctypes.CDLL(bridge_path, mode=ctypes.RTLD_GLOBAL)

  原理：
    - RTLD_GLOBAL 将 robot_bridge.so 的符号导出到进程全局符号表。
    - 当 CSDKLoader 随后调用 dlopen(libRobotService.so, RTLD_LAZY) 时，
      动态链接器在全局符号表中找到了 __libc_single_threaded，
      libRobotService.so 加载成功。

验证：
  [info]  create service `A5236E6F-...`  (IRobotService)
  [warning] unlicensed                   (硬件 License 检查，非阻塞)
  Robot created: <Robot disconnected>    ← 成功创建

──── 问题 2：Python 类型注解语法不兼容 ────

现象：
  TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

原因：
  X | None 联合类型注解语法（PEP 604）是 Python 3.10 引入的特性。
  本机运行的是 PyPy 3.9，不支持该语法（即使有 from __future__ import annotations
  也无法在运行时使用 | 操作符）。

解决方案：
  将所有 X | None 改为 Optional[X]，并在相应文件中添加：
    from typing import Optional

  涉及文件：
    - crp_robot/motion_service.py（5 处方法签名）
    - crp_robot/_types.py（4 处静态方法参数）

──── 问题 3：SDK 计划书中的数组大小标注错误 ────

计划书中部分函数签名写的是 double pos_out[10]，
实际上 SRobotPosition 的位置分量只有 6 个 double（x,y,z,Rx,Ry,Rz）。
实现时以头文件定义为准，统一使用 double pos6[6] + double ext[6] + int cfg[4]，
这样 Python 侧的内存布局清晰、不易出错。

================================================================================

四、最终验证结果
────────────────────────────────────────────────────────────────────────────────

编译：
  $ make -C crp_robot
  → robot_bridge.so  (132 KB, aarch64, 111 个导出符号)

导入测试（无硬件）：
  $ python3 -c "from crp_robot import Robot; ..."
  → Robot created: <Robot disconnected>   ✓
  → is_connected: False                   ✓
  → has_error: False                      ✓
  → IO X count: -1   (未连接时返回 -1，正常)
  → Motion available: False               ✓

连接到真实机器人后的完整使用示例：
  from crp_robot import Robot
  from crp_robot import RobotMode, CoordSystem, MotionParam

  with Robot("CrobotpOSSDK/bin/libRobotService.so") as robot:
      robot.connect("192.168.1.1")
      robot.servo_on()
      pos    = robot.get_position()           # → RobotPosition
      joints = robot.get_joint()              # → JointPosition
      robot.io.set_y(0, True)                 # 数字输出
      robot.motion.is_available()             # 引导运动就绪检查
      robot.file.upload("prog.pro", "/robot/program/prog.pro")

================================================================================

五、文件清单
────────────────────────────────────────────────────────────────────────────────

  crp_robot/
  ├── robot_bridge.cpp   (832 行 C++ 桥接代码)
  ├── Makefile
  ├── robot_bridge.so    (编译产物，可用 make 重新生成)
  ├── __init__.py        (65 行，公开导出)
  ├── _types.py          (90 行，数据类)
  ├── _enums.py          (78 行，枚举)
  ├── _lib.py            (336 行，ctypes 签名声明)
  ├── robot.py           (344 行，Robot 主类)
  ├── io_service.py      (127 行)
  ├── motion_service.py  (166 行)
  ├── model_service.py   (135 行)
  └── file_service.py    (43 行)

================================================================================
