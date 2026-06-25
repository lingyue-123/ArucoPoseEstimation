# 细粒度运动分段计时系统

## 1. Python 上下文管理器基础

### 1.1 `with` 语句与 `__enter__` / `__exit__`

```python
class File:
    def __enter__(self):
        print("打开文件")
        return self                     # return self 让 as 接收

    def __exit__(self, exc, exc_val, exc_tb):
        print("关闭文件")               # 即使 with 块抛异常也执行
        return False                    # False = 传播异常

with File() as f:
    print("读写文件")
# 输出:
# 打开文件
# 读写文件
# 关闭文件
```

**机制**：`with` 进入时自动调用 `__enter__`，离开时自动调用 `__exit__`。无论 `with` 块是正常结束还是抛异常，`__exit__` 都会执行——这就是为什么文件句柄、锁等资源可以用 `with` 安全释放。

### 1.2 `contextlib.contextmanager` + `yield`

手写 `__enter__`/`__exit__` 每次都要定义类，较繁琐。Python 提供了 `@contextmanager` 装饰器，把**生成器函数**转为上下文管理器：

```python
from contextlib import contextmanager

@contextmanager
def timer(label):
    """yield 之前 = __enter__  ||  yield 之后 = __exit__"""
    t0 = time.time()
    print(f"{label} 开始")
    yield                                 # ← 分界线：with 块在此执行
    dt = time.time() - t0
    print(f"{label} 结束 耗时={dt:.3f}s")

with timer("test"):
    time.sleep(1)
# 输出:
# test 开始
# test 结束 耗时=1.003s
```

**`yield` 的角色**：生成器执行到 `yield` 时暂停，控制权交还给 `with` 块。`with` 块执行完后，生成器从 `yield` 之后继续执行。

**执行时序图**：

```
  with timer("X"):
      time.sleep(2)
  print("done")

  → time0 = time.time()
  → print("开始")
  → yield (暂停)
  →     time.sleep(2)        ← with 块内
  → dt = time.time() - t0   ← 恢复
  → print("结束 耗时=2.003s")
  → print("done")
```

**对比两种写法**：

| | `__enter__/__exit__` 类 | `@contextmanager` 函数 |
|---|---|---|
| 代码量 | 多（定义类 + 2 个方法） | 少（一个函数） |
| 适用场景 | 复杂状态管理、需要复用对象 | before/after 做简单操作 |
| 本项目选择 | — | `MotionSegmentTimer.segment()` |

---

## 2. 计时系统架构

### 2.1 两层计时叠加

```
┌───────────────────────────────────────────────────────┐
│  _execute_auto_step                                  │
│  _step_t0 = time.time()                              │
│    ┌───────────────────────────────────────────────┐ │
│    │  timer.set_step("STATE_1")                    │ │
│    │                                               │ │
│    │  with timer.segment("夹爪+机械臂初始化"):     │ │
│    │      t0 = time.time()  ← 段级计时开始         │ │
│    │      asyncio.run(gather(...))                 │ │
│    │      dt = time.time() - t0  ← 段级计时结束    │ │
│    │                                               │ │
│    │  with timer.segment("双目粗定位运动"):         │ │
│    │      t0 = time.time()                         │ │
│    │      robot.move_by_joint_list(...)            │ │
│    │      dt = time.time() - t0                    │ │
│    └───────────────────────────────────────────────┘ │
│  _step_dt = time.time() - _step_t0                   │
│  logger.info("[计时] STATE_1 耗时 %.3f 秒")          │
└───────────────────────────────────────────────────────┘
```

- **Step 级**：整步总耗时，由 `_execute_auto_step` 记录，输出 `[计时] STATE_X 耗时 x.xxxs`
- **段 级**：步内每段运动的耗时 + TCP 位姿，由 `MotionSegmentTimer.segment()` 记录，汇总到 txt

两者不冲突——Step 级是外层总包，段级是内层拆分明细。

---

## 3. `MotionSegmentTimer` 核心实现

> 代码位置：`scripts/oneshot_move_crobot_insert_auto_brightness_v4.6.py:272-398`

### 3.1 初始化

```python
class MotionSegmentTimer:
    def __init__(self, robot, robot_cfg, robot_connected, output_dir="data"):
        self._segments = []           # 累积累积所有运动段的计时记录
        self._robot = robot           # 用于读取 TCP
        self._robot_cfg = robot_cfg   # 用于 TCP 单位转换
        self._robot_connected = robot_connected
        self._output_dir = output_dir
        self._step_name = "UNKNOWN"   # 当前 Step 名
        self._seg_idx = 0             # 当前 Step 内段号
        self._start_time = time.time() # 计时器创建时刻（用于 Total）
```

### 3.2 `segment()` — 核心计时方法

```python
@contextmanager
def segment(self, label):
    seg_idx = self._seg_idx + 1
    self._seg_idx = seg_idx

    # ── __enter__ 阶段 ──
    tcp_before = self._read_tcp()          # 读运动前 TCP
    logger.info("[计时] %s | #%d %s 开始", ...)   # 日志（不计入耗时）
    if tcp_before is not None:
        logger.info("[计时]  ... 位姿(前): ...")

    t0 = time.time()                        # ← 计时起点紧贴 yield
    yield                                    # ← 分界线，with 块在此执行

    # ── __exit__ 阶段 ──
    dt = time.time() - t0                   # 纯运动耗时
    tcp_after = self._read_tcp()            # 读运动后 TCP
    logger.info("[计时] %s | #%d %s 完成 耗时=%.3fs", ...)
    if tcp_after is not None:
        logger.info("[计时]  ... 位姿(后): ...")

    self._segments.append({                 # 存入列表等待导出
        'step': self._step_name,
        'idx': seg_idx,
        'label': label,
        'dt': dt,
        'tcp_before': ...,
        'tcp_after': ...,
    })
```

**关键设计**：`t0 = time.time()` 放在 `yield` 紧前面，所有 logger.info 调用在 `t0` 之前，确保日志本身不计入运动耗时。

### 3.3 `_read_tcp()` — 读取机器人位姿

```python
def _read_tcp(self):
    if not self._robot_connected or self._robot is None:
        return None
    tcp = self._robot.get_tcp_pose()      # 读原始值
    if tcp is None: return None
    pose = [float(v) for v in tcp]
    unit = getattr(self._robot_cfg, 'tcp_position_unit', 'mm')
    if unit == 'm':
        pose[:3] = [v * 1000.0 for v in pose[:3]]  # 统一为 mm
    return pose
```

主脚本中 `flow.arm = robot = bridge_robot` 三者指向同一对象，所以 `timer` 无论由主脚本还是 `CoverActionFlow` 触发，读到的都是同一个机器人。

### 3.4 `export_txt()` — 导出汇总

按 Step 分组输出到 `data/timing_log_YYYYmmdd_HHMMSS.txt`：

```
================================================================
  Step: STATE_1 | 耗时: 5.812s | 2段
----------------------------------------------------------------
  #1  夹爪(45)+机械臂(INIT_JOINT)(异步)          3.245s
      前: 100.00, 120.00, 300.00, 0.00, 0.00, 0.00
      后: 112.20, 99.88, 310.00, -56.13, 128.45, -5.91
  #2  双目粗定位运动                           2.567s
      前: 112.20, 99.88, 310.00, -56.13, 128.45, -5.91
      后: 115.50, 95.30, 305.00, -55.00, 130.00, -4.50
...
================================================================
  Step 耗时汇总
================================================================
  STATE_1     : 5.812s (2段)
  STATE_2     : 1.890s (1段)
  ...
  Total       : 185.234s (34段)
```

**注意**：`Total` 是 `time.time() - self._start_time`（即 `__init__` 到 `export_txt()` 的**墙钟时间**），包含 `time.sleep(12)` 等非运动等待。运动段 `dt` 求和 < `Total`。

---

## 4. 主流程中的调用模式

### 4.1 创建与绑定

```python
# main() 中，初始化完 robot、camera 之后
timer = MotionSegmentTimer(robot, robot_cfg, robot_connected, output_dir="data")
```

### 4.2 Step 内的直调运动

每个 step 函数内，对运动调用直接用 `timer.segment()`：

```python
def _step_2_force_open():
    timer.set_step("STATE_2")              # 声明当前 Step，段号自动归 0
    ...
    with timer.segment("力控按压开盖"):     # 段号自动 +1
        force_ctrl.run_ForceControl_OpenCover()
```

### 4.3 异步并行运动视为同一段

```python
# STEP_1: 夹爪 + 机械臂通过 asyncio.gather 同时执行
with timer.segment("夹爪(45)+机械臂(INIT_JOINT)(异步)"):
    async def parallel_task():
        await asyncio.gather(
            to_thread(flow.gripper.set_position, 45),
            to_thread(robot.move_joint, INIT_JOINT, 60))
    asyncio.run(parallel_task())
```

### 4.4 自动对准的迭代级计时

```python
# _execute_auto_align 内部
for attempt in range(1, args.align_max_attempts + 1):
    ...
    with timer.segment(f"对准插枪 #{attempt}/{args.align_max_attempts}"):
        ok = execute_move(robot, ctx["target_pose"], timeout=...)
```

每轮迭代是一条独立记录，而非整个对齐循环算一次。

### 4.5 flow.run() 的跨文件传递

主脚本通过 `flow.run(N, timer=timer)` 传递计时器。`CoverActionFlow` 中使用 `_timed()` 包装：

```python
# cover_main_bak_run.py
class _NullContext:
    def __enter__(self): return None
    def __exit__(self, *a): return False

def _timed(timer, label):
    if timer is None:              # 无计时器 → 空操作（不报错）
        return _NullContext()
    return timer.segment(label)    # 有计时器 → 正常记录

# 函数签名
def run(self, mode, pose=None, timer=None):
    ...
    self.run_screw_cover(timer=timer)

# 方法内使用
def run_screw_cover(self, timer=None):
    with _timed(timer, "旋盖-伺服轨迹运动"):
        self.arm.plan_and_move_position(...)

    with _timed(timer, "模式切换(CSP→PP)"):
        self.arm.switch_motion_model()

    with _timed(timer, "旋盖-夹爪张开"):
        self.gripper.set_position(25)
```

`_timed` 利用 NullObject 模式：当 `timer=None` 时返回一个 `__enter__`/`__exit__` 都是空操作的假上下文管理器，方法代码无需 `if timer:` 分支。

### 4.6 流程结束时导出

```python
# main() 主循环
if auto_state == AutoState.IDLE:
    if auto_flow_start_time is not None:
        logger.info("[计时] 总耗时 ...")
    timer.export_txt()    # ← 全部 Step 完成后导出 txt
```

---

## 5. 运动段划分总览

| Step | 段数 | 运动内容 |
|------|------|---------|
| 1 | 2 | ①夹爪+机械臂初始化(异步) ②双目粗定位运动 |
| 2 | 1 | 力控按压开盖 |
| 3 | 3 | flow.run(1): ①旋盖-伺服轨迹 ②旋盖-夹爪张开 ③旋盖-调整位姿 |
| 4 | N | 对准插枪 #1/N ... #N/N（每轮迭代 1 段） |
| 5 | 1 | 固定偏移运动(两点轨迹) |
| 6 | 5 | flow.run(2): ①夹爪张开(52) ②轨迹下沉 ③夹爪张开(25) ④后退30mm ⑤闭合+取枪点(异步) |
| 7 | N | 对准取枪 #1/N ... |
| 8 | 4 | ①固定偏移 ②沿z前进122mm ③夹爪闭合 ④舵机按压 |
| 9 | 1 | 沿z退出100mm |
| 10 | 2 | flow.run(3): ①舵机复位+关节(异步) ②前进100mm |
| 11 | 1 | 力控插枪 |
| 12 | 2 | ①舵机按压 ②力控拔枪 |
| 13 | 3 | flow.run(4): ①舵机复位+关节(异步) ②夹爪张开 ③pose后退150mm |
| 14 | 1 | flow.run(5): 夹爪+关节移动(异步) |
| 15 | N | 对准取小盖 #1/N ... |
| 16 | 1 | 固定偏移运动(两点轨迹) |
| 17 | 3 | flow.run(7): ①夹爪闭合(52) ②伺服轨迹推盖 ③夹爪张开(25); flow.run(6): 回终点位 |

此外还有 **6 段模式切换**计时（`模式切换(PP→CSP)` / `模式切换(CSP→PP)`），由 `switch_motion_model()` 调用触发。

---

## 6. 关键设计要点

| 要点 | 实现 |
|------|------|
| 计时精度 | `t0 = time.time()` 紧贴 `yield`，日志在 `t0` 之前 |
| NullObject | `_timed(None, ...)` 返回空壳，避免 `if timer:` 分支 |
| 异步归一段 | `asyncio.gather` 包在同一 `with` 内 |
| flow.run 黑盒拆解 | timer 透传到每个 helper 方法，内部逐段包装 |
| 单位统一 | `_read_tcp()` 中 m→mm 转换 |
| 多次运行 | 每次导出到独立文件 `timing_log_{timestamp}.txt` |
