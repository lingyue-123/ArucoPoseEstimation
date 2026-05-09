# AGENTS.md

## Project & Hardware

Robotic arm visual servo for charging gun insert/extract via magnetic suction. Eye-in-Hand: camera mounted on robot end-effector, ArUco marker on target. Robots: JAKA (jkrc SDK), KEBA (Modbus), CRP (CrobotpOS SDK). Platform: NVIDIA Jetson Linux.

## How to Run

All scripts run from the **project root**. There is no installable package — `robovision` is importable because every script injects the repo root into `sys.path`.

```bash
# Visual servo (key-step: m=move, r=set reference)
python scripts/visual_servo.py --camera mecheye --robot-driver jaka

# Dry-run (no robot connection) for testing calculations
python scripts/visual_servo.py --camera mecheye --no-robot

# One-shot move variants
python scripts/oneshot_move_jaka_insert_v2.py --camera mecheye
python scripts/oneshot_move_keba_insert_v3.py --camera mecheye --robot-driver modbus

# Hand-eye calibration
python scripts/collect_handeye_data.py
python scripts/run_handeye_calib.py
```

## Critical Gotchas

### JAKA SDK os.execv restart
When `JAKARobot.__init__` first runs, `_ensure_sdk_path()` modifies `LD_LIBRARY_PATH` then calls `os.execv()` to restart the Python process. This means the first invocation of any script that uses JAKA will restart. Do NOT try to `import jkrc` before `JAKARobot` is constructed — let the `build_robot` factory handle it. See `robovision/robot/jaka.py:23-31`.

### MechEye API: no sudo
```bash
pip install MechEyeApi   # NO sudo — forces system Python, breaks conda
```
Only works on Python 3.7–3.11.

### Network auto-config runs sudo ip commands
Both `build_camera()` and `build_robot()` may call `ensure_interface()` from `robovision/network.py`, which runs `sudo ip link/addr/route`. This requires passwordless sudo or user interaction. The `--no-robot` flag skips robot connection but not camera network config.

### No test/lint/typecheck infrastructure
There is no `setup.py`, `pyproject.toml`, CI workflow, pre-commit config, or test suite. All verification is done by running scripts against real hardware.

## Coordinate & Unit Conventions (NEVER change these)

| Convention | Value |
|---|---|
| Euler order | **ZYX intrinsic** (RPY) — `Rz(rz) @ Ry(ry) @ Rx(rx)` |
| Angle unit | **degrees** everywhere |
| Translation unit | **millimeters** (mm) everywhere |
| Pose vector format | `[x, y, z, rx, ry, rz]` |
| Tool-axis movement | **Right-multiply**: `T_new = T_current @ T_z` |

`RobotBase.get_tcp_pose()` returns `(x, y, z, rx, ry, rz)` in mm/deg regardless of driver.

## Architecture: What to Import Where

### Factory functions — ALWAYS use these, NEVER import concrete classes directly

```python
from robovision.cameras import build_camera      # returns CameraInterface
from robovision.robot import build_robot          # returns RobotBase
```

They handle network config, JAKA SDK restart, and driver dispatch automatically.

### Config singleton

```python
from robovision.config.loader import get_config
cfg = get_config()                         # loads config/*.yaml lazily
cam = cfg.get_camera('mecheye')            # CameraConfig dataclass
robot_cfg = cfg.get_robot(driver='jaka')   # RobotConfig dataclass
marker = cfg.get_marker()                  # MarkerConfig
det = cfg.get_detection()                  # DetectionConfig
```

### ServoSession — the main entry point

```python
from robovision.servo.session import ServoSession
session = ServoSession(args)
session.setup()
frame, result, target_data, tcp, T_g2b_cur = session.read_and_detect()
```

Wires together camera + robot + detector + hand-eye. Then use `session.compute_target()` and servo core helpers (`compute_step_pose`, `check_step_safety`, `execute_move`).

### Geometry transforms

```python
from robovision.geometry.transforms import (
    pose_to_matrix, matrix_to_pose,       # 6D vector ↔ 4x4 matrix
    euler_to_rotmat, rotmat_to_euler,     # euler ↔ rotation matrix
    transform_pose,                       # apply transform
    offset_pose_along_tool_axis,          # T @ T_axis
)
```

## Configuration

Four YAML files in `config/`:
- `cameras.yaml` — camera type, intrinsics, connection/network params
- `robot.yaml` — multi-driver (`drivers.jaka`, `drivers.modbus`, `drivers.crp`), `default_driver: jaka`
- `markers.yaml` — ArUco dictionary, valid IDs, physical marker sizes (mm)
- `detection.yaml` — Kalman, pose smoother, anomaly gate thresholds

Override robot IP on command line: `--robot-ip 192.168.1.100`

## Pose Files

Format: CSV, one line per pose, 6 columns: `x,y,z,rx,ry,rz`. Read/write via `robovision.io.pose_file` (`load_pose_file`, `save_pose_file`). Key files in `data/`:
- `data/handeye/hand_eye_result.txt` — 4x4 matrix (default `--hand-eye` path)
- `data/aruco/aruco_pose_ref.txt` — reference ArUco pose
- `data/aruco/tcp_ref.txt` — reference TCP pose (derived from aruco_ref path)
