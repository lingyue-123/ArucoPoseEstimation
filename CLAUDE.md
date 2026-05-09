# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotic arm visual calibration and servo system for a "charging gun unplug/insert" application using magnetic suction. The system uses ArUco marker detection with a camera mounted on the robot end-effector (Eye-in-Hand) to guide the robot to a reference pose, then perform insertion/extraction along the tool Z-axis.

## Hardware

- Robots: JAKA (jkrc SDK), KEBA (Modbus TCP + custom C kinematics library), CRP (CrobotpOS SDK)
- Cameras: Hikvision industrial, Mech-Eye 3D, USB, RTSP
- Platform: Linux (NVIDIA Jetson, tegra kernel)

## Running Scripts

All scripts run from the project root. The `robovision` package must be importable (scripts add the project root to `sys.path`).

```bash
# Visual servo - key-step mode (press m to move, r to set reference)
python scripts/visual_servo.py --camera mecheye --robot-driver jaka

# One-shot move (full distance in one motion, with insert/extract)
python scripts/oneshot_move_jaka_insert_v2.py --camera mecheye

# KEBA-specific one-shot with full insert/take workflow
python scripts/oneshot_move_keba_insert_v3.py --camera mecheye --robot-driver modbus

# Force control insertion along Z axis
python scripts/force_insert_z.py

# Collect hand-eye calibration data
python scripts/collect_handeye_data.py

# Run hand-eye calibration
python scripts/run_handeye_calib.py

# Dry-run (no robot connection) for testing calculations
python scripts/visual_servo.py --camera mecheye --no-robot
```

## Architecture

```
robovision/
├── cameras/       # Camera abstraction: base + hikvision/usb/rtsp/mecheye
│                   #   build_camera() factory in __init__.py
├── robot/         # Robot abstraction: base + jaka/keba(modbus)/crp
│                   #   build_robot() factory in __init__.py
├── detection/     # ArucoDetector: Kalman-filtered corners + multi-method PnP + pose smoothing
│                   #   pnp.py: solve_pnp_best (IPPE_SQUARE → ITERATIVE fallback)
├── servo/         # Visual servo: core step computation, ServoSession orchestration
│                   #   runner.py: shared helpers (build_frame_state, run_pbvs_step, render_frame)
├── geometry/      # transforms.py: euler↔matrix↔quat, SLERP, pose chain math
│                   #   All angles are ZYX intrinsic Euler (RPY), degrees
├── calibration/   # Camera intrinsic (intrinsic.py) + hand-eye (hand_eye.py)
│                   #   Eye-in-Hand: T_cam2gripper
├── tracking/      # kalman.py: Kalman filter for ArUco corners
│                   #   pose_smoother.py: SLERP rot + EMA trans + anomaly gate
├── config/        # loader.py: YAML config → typed dataclasses (Config singleton)
├── io/            # pose_file.py: CSV read/write for 6-DOF poses
├── visualization/ # aruco_overlay.py: draw axes/corners/pose text on frame
└── network.py     # Auto-configure Ethernet interfaces (ip link/addr/route via sudo)
```

### Coordinate Conventions (Critical)

- **ZYX intrinsic Euler angles** (RPY) everywhere — rotation order `Rz(rz) @ Ry(ry) @ Rx(rx)`
- **RobotBase.get_tcp_pose()** returns `(x, y, z, rx, ry, rz)` — JAKA uses mm/deg, KEBA uses m/deg internally but the interface aims for mm/deg
- **Tool Z-axis insertion**: `T_new = T_current @ T_z` (right-multiply — move along local tool axis)
- Pose vectors in scripts are `[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]`

### Key Design Patterns

- **Factory functions** (`build_camera`, `build_robot`) take config dataclasses and return interface implementations — scripts never import concrete classes directly
- **ServoSession** (servo/session.py) encapsulates the entire pipeline: camera + robot + detector + hand-eye → single `read_and_detect()` call
- **Config singleton** (`config/loader.py`): `get_config()` lazily loads YAML files from `config/`
- **Modbus robot** (KEBA): reads joint angles via Modbus TCP, computes TCP pose via a C shared library (`JnttoZYX`), inverse kinematics via `ZYXtoJntAngle`

## Configuration

Four YAML files in `config/`:
- `cameras.yaml` — camera type, intrinsics, connection params, optional network config
- `robot.yaml` — multi-driver: `drivers.jaka`, `drivers.modbus`, `drivers.crp`, each with IP/SDK path/network
- `markers.yaml` — ArUco dictionary, valid IDs, physical marker sizes (mm)
- `detection.yaml` — Kalman params, pose smoother params, anomaly gate thresholds

## Third-Party Dependencies

- `third_party/jaka-python-sdk/` — JAKA jkrc.so + libjakaAPI.so (requires LD_LIBRARY_PATH and sys.path setup)
- `third_party/robot_driver/` — CartesianPose dataclass + shared robot interface helpers
- `third_party/crp_robot_sdk/` — CRP robot C SDK
- KEBA kinematics `.so` — loaded via ctypes in `robot/keba.py` (`_load_kinematics_library`)
- MechEyeApi — `pip install MechEyeApi` (Python 3.7-3.11 only, do NOT use sudo)

## Data Files

- `data/handeye/` — calibration input (robot_tcp + board_to_cam) and result (hand_eye_result_*.txt as 4x4 matrix)
- `data/aruco/` — reference ArUco poses and TCP references saved at runtime
