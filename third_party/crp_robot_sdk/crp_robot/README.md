# crp_robot

Python wrapper for the **CrobotpOS SDK** (成都柔博机器人).

Exposes the C++ virtual-interface SDK as a clean Python API via a thin C bridge layer (`robot_bridge.so`).

---

## Architecture

```
libRobotService.so   (CrobotpOS SDK, obtained from manufacturer)
        ↓
robot_bridge.so      (C bridge, compile once with make)
        ↓
crp_robot/           (Python package, ctypes)
        ↓
your Python code
```

---

## Requirements

### Hardware
- CrobotpOS robot controller connected via **Ethernet (RJ45)**
- Upper computer (PC / NVIDIA Jetson) on the same subnet as the controller

### Software
- Linux (tested on Ubuntu 20.04, aarch64 / Jetson)
- Python 3.8+  (CPython or PyPy)
- g++ with C++17 support
- CrobotpOS SDK (`libRobotService.so` + headers) — obtain from manufacturer

---

## Installation

### 1. Obtain the SDK

Get `CrobotpOSSDK` from the robot manufacturer (成都柔博).
Place it so the directory structure looks like:

```
your_project/
├── CrobotpOSSDK/
│   ├── bin/
│   │   └── libRobotService.so
│   └── cpp/
│       └── include/
│           ├── IRobotService.h
│           ├── IIOService.h
│           ├── IMotionService.h
│           ├── IModelService.h
│           ├── IFileService.h
│           ├── CSDKLoader.h
│           └── RobotTypes.h
└── crp_robot/          ← this repo
```

### 2. Compile the C bridge

```bash
make -C crp_robot
```

This produces `crp_robot/robot_bridge.so`.

### 3. Install Python package (optional)

```bash
pip install -e .
```

Or simply add the parent directory to your `PYTHONPATH`.

---

## Quick Start

```python
from crp_robot import Robot

# Path to the SDK .so (obtained from manufacturer)
SO_PATH = "CrobotpOSSDK/bin/libRobotService.so"

with Robot(SO_PATH) as robot:
    robot.connect("192.168.1.10")   # controller IP
    robot.servo_on()

    pos    = robot.get_position()   # → RobotPosition(x, y, z, Rx, Ry, Rz)
    joints = robot.get_joint()      # → JointPosition(body, ext, cfg)
    print(pos)
    print(joints)

    robot.servo_off()
```

---

## API Overview

### Robot (main class)

```python
robot = Robot(so_path, bridge_path=None)   # bridge_path defaults to crp_robot/robot_bridge.so

# Connection
robot.connect(ip, disable_hardware=True)
robot.disconnect()
robot.is_connected()

# Servo
robot.servo_on()
robot.servo_off()
robot.is_servo_on()

# Position
robot.get_position(coord=CoordSystem.Base)  # → RobotPosition
robot.get_joint()                           # → JointPosition

# Manual motion (manual mode only)
robot.jog_joint(joint_index, offset_deg)
robot.jog_linear(axis_index, offset_mm)
robot.move_j(joint_target)
robot.move_l(cart_target)
robot.stop_move()

# Program control
robot.start_program("prog.pro", line=0)
robot.stop_program()
robot.get_program_status()   # → ProgramStatus

# Errors
robot.has_error()
robot.clear_error()
robot.get_errors()           # → [(error_id, message), ...]
robot.emergency_stop(True)

# Global variables
robot.get_gi(index)   # integer
robot.get_gr(index)   # float
robot.set_gi(index, value)
robot.set_gr(index, value)

# Position variables
robot.get_gp(index)   # → RobotPosition
robot.get_gj(index)   # → JointPosition
robot.set_gp(index, pos)
robot.set_gj(index, jp)
```

### Sub-services (lazy-loaded properties)

```python
# IO
robot.io.get_x(index)              # digital input
robot.io.set_y(index, True/False)  # digital output
robot.io.get_ain(index)            # analog input (float)
robot.io.set_aot(index, value)     # analog output
robot.io.get_gin(index)            # group input (uint32)
robot.io.set_got(index, value)     # group output

# Motion guidance (path or instruction mode)
robot.motion.is_available()
robot.motion.move_abs_j(index, joint, param)
robot.motion.move_j(index, target, param)
robot.motion.move_l(index, target, param, strategy)
robot.motion.move_c(index, p2, p3, param)
robot.motion.move_jump(index, target, param, top, up, down)
robot.motion.send_path_joint(joints_list)   # path guidance
robot.motion.send_path_pos(positions_list)
robot.motion.move_path(ratio)
robot.motion.finalize(MotionType.Instruction)

# Kinematics (model service)
robot.model.fkine(dh, joints)         # forward kinematics (offline)
robot.model.ikine(dh, target)         # inverse kinematics (offline)
robot.model.joint2pos(joints, tool, user, coord)   # online
robot.model.pos2joint(pos, tool, user, coord)      # online
robot.model.convert_coord(pos, ...)

# File management
robot.file.upload(local_path, remote_path)
robot.file.download(remote_path, local_path)
robot.file.exists(path)
robot.file.mkdir(path, recursive=True)
robot.file.remove(path)
```

---

## Data Types

```python
from crp_robot import RobotPosition, JointPosition, MotionParam, DHParam

# Cartesian position
pos = RobotPosition(x=482, y=34, z=298, Rx=-177, Ry=0, Rz=0)

# Joint position
jp = JointPosition(body=[0, -30, 90, 0, 90, 0])

# Motion parameters
param = MotionParam(speed=50, pl=0, smooth=0, acc=1, dec=1)

# DH parameters (from teach pendant: menu → parameters → mechanism)
dh = DHParam(a=[0, 0, ...], d=[0, 0, ...])
```

## Enums

```python
from crp_robot import (
    RobotMode,       # Manual, Playing, Remote
    CoordSystem,     # Joint, World, Base, User, Tool
    ProgramStatus,   # Stop, Running, Pause
    MotionType,      # Path, Instruction
    MoveStrategy,    # TimeFirst, DistanceFirst, PoseFirst
    DryRunMode,      # Single, Continuous, Empty
    AutoRunMode,     # SingleLine, SingleLoop, ContinuousLoop
)
```

---

## Network Setup

The controller is connected via a standard **RJ45 Ethernet cable**.
The upper computer can simultaneously use WiFi for internet access — there is no conflict.

```
Upper computer (e.g. Jetson)
├── eth0  192.168.1.100  ──cable──► Controller  192.168.1.10
└── wlan0 192.168.0.xxx  ──WiFi──►  Router ──► Internet
```

Configure the wired interface with a static IP and **no default gateway**:

```bash
sudo nmcli connection add \
  type ethernet ifname eth0 con-name robot-arm \
  ip4 192.168.1.100/24
  # no gw4 — internet traffic stays on WiFi

sudo nmcli connection up robot-arm
ping 192.168.1.10   # verify controller reachable
```

---

## Notes

- The SDK logs `[warning] unlicensed` at startup — this is a hardware licence check from the SDK itself and does **not** block functionality.
- `robot_bridge.so` embeds a weak stub for `__libc_single_threaded` to ensure compatibility with glibc 2.31 (Ubuntu 20.04). The bridge must be loaded with `RTLD_GLOBAL` (handled automatically by `_lib.py`).
- `disableHardware=True` (default) bypasses the teach-pendant safety interlock, which is required for SDK-controlled motion.

---

## License

This wrapper (`crp_robot/`) is provided as-is for research and automation purposes.
The underlying **CrobotpOS SDK** (`libRobotService.so`) is proprietary software owned by
成都柔博机器人科技有限公司 (Chengdu CRP Robot Technology Co., Ltd.) and is **not included** in this repository.
