# force_control_crp

## 1. 模块定位

`force_control_crp` 是 `scripts/oneshot_move_jaka_insert_v4.3.py` 使用的新力控模块。

它的目标不是替换 `scripts/bin` 下的老版本文件，而是：

- 保留 `v4.2` 里 `e / d / s` 这套操作习惯
- 不修改 `scripts/bin/` 原始文件
- 将力控源码、构建脚本、运行依赖统一整理到 `third_party/force_control_crp/`
- 为后续“主流程机器人会话”和“力控会话”共享同一套 CRP 底层上下文做准备

---

## 2. 源码来源

当前目录中的 native 力控代码来自以下历史文件的副本：

- `scripts/bin/it_robotposTEST8.cpp`
- `scripts/bin/ForceControl.c`
- `scripts/bin/IdentifyCommon.c`
- `scripts/bin/OnLineFit.c`

同时复制了原来运行时依赖的资源：

- `simple_test`
- `data/guidanceInst.pro`
- `data/guidancePos.pro`
- `libRobotService.so`
- `license.key`

这样做的目的，是让新模块可以完全独立于 `scripts/bin/` 工作，后续所有调整都只发生在 `third_party/force_control_crp/` 下。

---

## 3. 这次实际改了哪些地方

### 3.1 新增了独立模块骨架

新增文件：

- `controller.py`
- `shared_crp_robot.py`
- `Makefile`
- `src/*.c / *.cpp / *.h`
- `data/guidanceInst.pro`
- `data/guidancePos.pro`

其中：

- `controller.py` 提供 Python 层 `RobotController` 封装，尽量保持旧接口兼容
- `shared_crp_robot.py` 提供一个基于 `third_party/crp_robot_sdk` 的机器人适配层
- `Makefile` 负责构建 `libforcecontrol_crp.so`

### 3.2 native 层增加了“外部服务接入”能力

在 `src/it_robotposTEST8.cpp` 中新增了两个导出接口：

- `AttachExternalCrpServices(void *robot, void *motion, void *file)`
- `DetachExternalCrpServices()`

这两个接口的作用是：

- 允许 Python 主流程把已经建立好的 CRP service 指针传进力控 `.so`
- 让力控逻辑不必总是自己重新创建一套 `CSDKLoader + IRobotService + IMotionService`

### 3.3 增加了共享服务检测逻辑

在 `src/it_robotposTEST8.cpp` 中新增了：

- `ensure_sdk_services()`
- `ensure_robot_session(...)`
- `g_external_services`

思路是：

1. 如果没有外部 service，就按模块内部自己的逻辑初始化
2. 如果已经 attach 了外部 service，就优先复用外部的 `robot / motion / file`
3. 这样 `PowerOn_PPmode / ChargeIn / PoseAdjust / ChargeOut` 都可以走同一套 service 指针

### 3.4 去掉了力控结束时的强制下电 / 断连

原始版本力控结束时会执行：

- `servoPowerOff()`
- `disconnect()`

这会和上层业务脚本已经连接好的机器人主流程发生冲突。

因此，这次在 `force_control_crp` 的副本里，把这些“结束时强制下电 / 断连”的逻辑去掉了，避免：

- 主流程还在用机器人
- 力控一结束就把伺服关掉
- 后续普通运动或状态读取立即失效

### 3.5 增加了 Python 层 attach 接口

在 `controller.py` 中新增：

- `attach_crp_robot(...)`
- `detach_crp_robot()`

作用：

- 从 `third_party/crp_robot_sdk` 的 bridge `Robot` 对象里取出底层 service 指针
- 调用 native 层的 `AttachExternalCrpServices(...)`
- 让 Python 主流程和力控 `.so` 指向同一份 CRP service

### 3.6 新增共享机器人适配器

新增：

- `shared_crp_robot.py`

这个文件里的 `BridgeCRobotAdapter` 负责：

- 用 `third_party/crp_robot_sdk` 建立主机器人连接
- 对外提供和旧 `CRobot` 相近的方法名
- 让 `scripts/oneshot_move_jaka_insert_v4.3.py` 尽量少改业务逻辑

---

## 4. 改动思路

### 4.1 为什么不能继续直接用旧 `.so`

原始 `scripts/bin/librobotcontrol.so` 的问题不是“只有路径乱”，而是架构上把两件事耦合在了一起：

1. 力控算法
2. 机器人连接 / 上电 / 运动服务初始化

这会导致：

- 主流程自己连了一次机器人
- 力控 `.so` 又自己连一次机器人
- 普通运动和力控运动并不是同一套底层会话

最终表现就是：

- 重复 connect
- 重复 servo on/off
- `motion service` 认为自己没连上机器人
- 普通运动和力控互相干扰

### 4.2 为什么需要“共享 CRP 底层句柄”

这里说的“CRP 底层句柄”，本质上就是 native C/C++ 层真正控制机器人的那几个对象引用，例如：

- `IRobotService*`
- `IMotionService*`
- `IFileService*`

只要普通运动和力控使用的不是同一组 service 指针，就仍然是“两套会话”。

所以推荐方案不是“减少重复 connect 调用次数”，而是：

- 主流程建立唯一机器人上下文
- 力控模块 attach 到这同一份上下文
- 普通运动、程序控制、路径引导、力控路径发送都走同一套底层 service

### 4.3 为什么要单独做 `shared_crp_robot.py`

因为项目原来 `scripts/crobot_driver_interface.py` 走的是 `crobotsdk` 那套 Python 扩展，
它在 Python 层没有直接暴露出可复用的 raw context / service 指针。

而 `third_party/crp_robot_sdk` 这条 bridge 路线内部有：

- `Robot._ctx`
- `get_service_ptrs()`

所以这次选择在 `v4.3` 中将主机器人切换到 bridge 这条链，再把同一套 service 指针传给力控 `.so`。

### 4.4 这次如何解决“反复上下电”

这次处理“反复上下电”的核心思路，不是继续容忍“主流程一套机器人连接、力控 `.so` 再自己连一套”的双会话结构，而是把普通运动、程序启动、力控路径发送统一收口到同一份 CRP 底层上下文上。具体措施包括：一是在 native 层增加 `AttachExternalCrpServices(...)`，让力控模块直接复用主流程已经建立好的 `IRobotService / IMotionService / IFileService`；二是移除力控流程结束时内部那套强制 `servoPowerOff()` 和 `disconnect()` 的收尾逻辑，避免力控结束后把主流程仍在使用的机器人会话关掉；三是在程序切模式、重启 `guidanceInst.pro / guidancePos.pro` 前增加伺服状态确认与必要时补上电，避免模式切换后因为伺服状态丢失导致再次启动失败；四是在每次进入插枪力控前重置力控内部运行状态，避免上一轮残留状态导致看起来像“没有真正进入力控、却又重复触发上下电流程”。这样调整后，力控模块的职责被收敛为“复用主会话并执行力控逻辑”，而不是“自己再建一套机器人连接并独立管理上下电”。

---

## 5. 当前目录和 `crp_robot_sdk` 的关系

这次改动分成两半：

### `force_control_crp` 负责

- 保存力控 native 源码副本
- 构建 `libforcecontrol_crp.so`
- 暴露 Python `RobotController`
- 接收外部 CRP service 指针

### `crp_robot_sdk` 负责

- 创建 bridge 版本的 CRP 上下文
- 暴露 `_ctx`
- 暴露 `IRobotService / IMotionService / IFileService` 对应的原生指针
- 供 `force_control_crp` attach 复用

两边缺一不可：

- 只有 `force_control_crp`，没有 service 导出能力，仍然无法共享底层上下文
- 只有 `crp_robot_sdk`，没有力控 native 接入点，也无法让力控复用这份上下文

---

## 6. 构建方式

```bash
make -C third_party/force_control_crp
```

会生成：

- `third_party/force_control_crp/libforcecontrol_crp.so`

---

## 7. 已知事项

### 7.1 当前版本的重点是“单会话化”

这次工作的核心目标不是重写力控算法，而是把：

- 普通运动
- 力控运动

统一到一套共享的 CRP 底层 service 上。

### 7.2 仍然需要真机回归

已完成的验证包括：

- `.so` 编译通过
- Python import 通过
- bridge 可以导出 service 指针
- force-control 可以 attach 外部 service

但以下内容仍需要真机验证：

- `e / d / s` 实际运行链路
- `simple_test` 的 sudo 权限
- 控制器当前程序状态与 `guidanceInst.pro / guidancePos.pro` 的现场一致性

---

## 8. 总结

这次对 `force_control_crp` 的改动，核心可以概括为三句话：

1. 先把 `scripts/bin` 里的力控实现完整搬进 `third_party`
2. 再把“力控算法”和“机器人会话管理”从架构上拆开
3. 最后通过共享 CRP 底层 service 指针，让 `v4.3` 的普通运动和力控真正使用同一套底层会话
