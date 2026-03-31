"""
Robot — top-level class wrapping IRobotService and providing lazy access to
IO, Motion, Model, and File sub-services.
"""
import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ._lib import load_bridge, d6, i4
from ._types import RobotPosition, JointPosition
from ._enums import (CoordSystem, RobotMode, ProgramStatus,
                     DryRunMode, AutoRunMode)
from .io_service     import IOService
from .motion_service import MotionService
from .model_service  import ModelService
from .file_service   import FileService


class Robot:
    """
    Entry point for the CrobotpOS Python SDK.

    Parameters
    ----------
    so_path : str
        Path to libRobotService.so (the original SDK library).
    bridge_path : str
        Path to robot_bridge.so (the compiled C bridge).
        Defaults to <directory of robot.py>/robot_bridge.so.

    Example
    -------
    >>> with Robot("/opt/crp/libRobotService.so") as robot:
    ...     robot.connect("192.168.1.1")
    ...     robot.servo_on()
    ...     print(robot.get_position())
    """

    def __init__(self,
                 so_path: str,
                 bridge_path: Optional[str] = None):
        if bridge_path is None:
            bridge_path = str(Path(__file__).parent / "robot_bridge.so")
        self._lib = load_bridge(bridge_path)
        self._ctx = self._lib.crp_create(so_path.encode())
        if not self._ctx:
            raise RuntimeError(
                f"crp_create failed — could not load '{so_path}'. "
                "Check that the path is correct and libRobotService.so "
                "dependencies are resolvable.")
        self._io:     Optional[IOService]     = None
        self._motion: Optional[MotionService] = None
        self._model:  Optional[ModelService]  = None
        self._file:   Optional[FileService]   = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._ctx:
            self._lib.crp_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    # ── Sub-service properties (lazy) ─────────────────────────────────────────

    @property
    def io(self) -> IOService:
        if self._io is None:
            self._io = IOService(self._lib, self._ctx)
        return self._io

    @property
    def motion(self) -> MotionService:
        if self._motion is None:
            self._motion = MotionService(self._lib, self._ctx)
        return self._motion

    @property
    def model(self) -> ModelService:
        if self._model is None:
            self._model = ModelService(self._lib, self._ctx)
        return self._model

    @property
    def file(self) -> FileService:
        if self._file is None:
            self._file = FileService(self._lib, self._ctx)
        return self._file

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, ip: str, disable_hardware: bool = True) -> bool:
        """Connect to the robot controller."""
        return bool(self._lib.crp_connect(self._ctx, ip.encode(), disable_hardware))

    def disconnect(self) -> bool:
        return bool(self._lib.crp_disconnect(self._ctx))

    def is_connected(self) -> bool:
        return bool(self._lib.crp_is_connected(self._ctx))

    # ── Servo power ───────────────────────────────────────────────────────────

    def servo_on(self) -> bool:
        """Power on / servo-ready."""
        return bool(self._lib.crp_servo_on(self._ctx))

    def servo_off(self) -> bool:
        return bool(self._lib.crp_servo_off(self._ctx))

    def is_servo_on(self) -> bool:
        return bool(self._lib.crp_is_servo_on(self._ctx))

    # ── Mode / speed ──────────────────────────────────────────────────────────

    def get_work_mode(self) -> RobotMode:
        return RobotMode(self._lib.crp_get_work_mode(self._ctx))

    def set_work_mode(self, mode: RobotMode) -> bool:
        return bool(self._lib.crp_set_work_mode(self._ctx, int(mode)))

    def get_speed_ratio(self) -> int:
        return self._lib.crp_get_speed_ratio(self._ctx)

    def set_speed_ratio(self, ratio: int) -> bool:
        """Set speed ratio 0–100 (%)."""
        return bool(self._lib.crp_set_speed_ratio(self._ctx, ratio))

    def get_coord_sys(self) -> CoordSystem:
        return CoordSystem(self._lib.crp_get_coord_sys(self._ctx))

    def set_coord_sys(self, coord: CoordSystem) -> bool:
        return bool(self._lib.crp_set_coord_sys(self._ctx, int(coord)))

    # ── Position ──────────────────────────────────────────────────────────────

    def get_position(self, coord: CoordSystem = CoordSystem.Base) -> RobotPosition:
        """
        Get current Cartesian position in the specified coordinate system.
        For joint coordinates use get_joint() instead.
        """
        buf = (ctypes.c_double * 6)()
        ok = self._lib.crp_get_position(self._ctx, int(coord), buf, 6)
        if not ok:
            raise RuntimeError("crp_get_position failed")
        return RobotPosition(x=buf[0], y=buf[1], z=buf[2],
                             Rx=buf[3], Ry=buf[4], Rz=buf[5])

    def get_joint(self) -> JointPosition:
        """Get current joint angles and cfg."""
        body = d6(); ext = d6(); cfg = i4()
        ok = self._lib.crp_get_joint(self._ctx, body, ext, cfg)
        if not ok:
            raise RuntimeError("crp_get_joint failed")
        return JointPosition(list(body), list(ext), list(cfg))

    def get_joint_raw(self) -> List[float]:
        """
        Get raw joint + ext buffer (12 doubles: J1..J6, ext1..ext6).
        Uses getCurrentPosition with CS_Joint.
        """
        buf = (ctypes.c_double * 12)()
        ok = self._lib.crp_get_position(self._ctx, int(CoordSystem.Joint), buf, 12)
        if not ok:
            raise RuntimeError("crp_get_position(CS_Joint) failed")
        return list(buf)

    # ── Manual motion ─────────────────────────────────────────────────────────

    def jog_joint(self, joint_index: int, offset: float) -> bool:
        """Jog a single joint by offset degrees (manual mode)."""
        return bool(self._lib.crp_jog_move_j(self._ctx, joint_index, offset))

    def jog_linear(self, axis_index: int, offset: float) -> bool:
        """Jog along a Cartesian axis (0=X..5=Rz) by offset mm/deg."""
        return bool(self._lib.crp_jog_move_l(self._ctx, axis_index, offset))

    def move_j(self, target: JointPosition) -> bool:
        """Move to joint target (manual mode, sets GJ999)."""
        body = d6(target.body); ext = d6(target.ext); cfg = i4(target.cfg)
        return bool(self._lib.crp_move_j_joint(self._ctx, body, ext, cfg))

    def move_l(self, target: RobotPosition) -> bool:
        """Move to Cartesian target (manual mode, sets GP999)."""
        ext = d6(target.ext_joint); cfg = i4(target.cfg)
        return bool(self._lib.crp_move_l_cart(
            self._ctx,
            target.x, target.y, target.z, target.Rx, target.Ry, target.Rz,
            ext, cfg))

    def stop_move(self) -> bool:
        return bool(self._lib.crp_stop_move(self._ctx))

    def is_moving(self) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_is_moving(self._ctx, ctypes.byref(val))
        return val.value

    # ── Collaborative mode ────────────────────────────────────────────────────

    def enable_cobot_mode(self, enable: bool) -> bool:
        return bool(self._lib.crp_enable_cobot_mode(self._ctx, enable))

    def is_cobot_mode_enabled(self) -> bool:
        return bool(self._lib.crp_is_cobot_mode_enabled(self._ctx))

    # ── Run modes ─────────────────────────────────────────────────────────────

    def get_auto_run_mode(self) -> AutoRunMode:
        return AutoRunMode(self._lib.crp_get_auto_run_mode(self._ctx))

    def set_auto_run_mode(self, mode: AutoRunMode) -> bool:
        return bool(self._lib.crp_set_auto_run_mode(self._ctx, int(mode)))

    def get_dry_run_mode(self) -> DryRunMode:
        return DryRunMode(self._lib.crp_get_dry_run_mode(self._ctx))

    def set_dry_run_mode(self, mode: DryRunMode) -> bool:
        return bool(self._lib.crp_set_dry_run_mode(self._ctx, int(mode)))

    # ── Program control ───────────────────────────────────────────────────────

    def start_program(self, program: str, line: int = 0) -> bool:
        """Start a robot program. program = path like 'prog.pro' or full path."""
        return bool(self._lib.crp_start_program(self._ctx, program.encode(), line))

    def stop_program(self) -> bool:
        return bool(self._lib.crp_stop_program(self._ctx))

    def resume_program(self, program: str) -> bool:
        return bool(self._lib.crp_resume_program(self._ctx, program.encode()))

    def get_program_status(self) -> ProgramStatus:
        return ProgramStatus(self._lib.crp_get_program_status(self._ctx))

    def get_program_line(self) -> int:
        return self._lib.crp_get_program_line(self._ctx)

    def get_program_path(self) -> Optional[str]:
        r = self._lib.crp_get_program_path(self._ctx)
        return r.decode() if r else None

    # ── Errors / E-stop ───────────────────────────────────────────────────────

    def has_error(self) -> bool:
        return bool(self._lib.crp_has_error(self._ctx))

    def clear_error(self) -> bool:
        return bool(self._lib.crp_clear_error(self._ctx))

    def has_emergency_error(self) -> bool:
        return bool(self._lib.crp_has_emergency_error(self._ctx))

    def emergency_stop(self, enable: bool) -> bool:
        """Trigger (enable=True) or release (enable=False) software e-stop."""
        return bool(self._lib.crp_emergency_stop(self._ctx, enable))

    def get_errors(self) -> List[Tuple[int, str]]:
        """Return list of (error_id, message) for all current errors."""
        count = ctypes.c_size_t(0)
        if not self._lib.crp_get_error_count(self._ctx, ctypes.byref(count)):
            return []
        errors = []
        buf = ctypes.create_string_buffer(512)
        for i in range(count.value):
            eid = ctypes.c_uint32(0)
            self._lib.crp_get_error_id(self._ctx, i, ctypes.byref(eid))
            self._lib.crp_get_error_message(self._ctx, i, buf, 512)
            errors.append((eid.value, buf.value.decode(errors='replace')))
        return errors

    # ── Global variables: GI / GR / UI ───────────────────────────────────────

    def get_gi(self, index: int) -> int:
        arr = (ctypes.c_int32 * 1)()
        self._lib.crp_get_gi(self._ctx, index, arr, 1)
        return arr[0]

    def set_gi(self, index: int, value: int) -> bool:
        arr = (ctypes.c_int32 * 1)(value)
        return bool(self._lib.crp_set_gi(self._ctx, index, arr, 1))

    def get_gr(self, index: int) -> float:
        arr = (ctypes.c_double * 1)()
        self._lib.crp_get_gr(self._ctx, index, arr, 1)
        return arr[0]

    def set_gr(self, index: int, value: float) -> bool:
        arr = (ctypes.c_double * 1)(value)
        return bool(self._lib.crp_set_gr(self._ctx, index, arr, 1))

    def get_ui(self, index: int) -> int:
        arr = (ctypes.c_int16 * 1)()
        self._lib.crp_get_ui(self._ctx, index, arr, 1)
        return arr[0]

    def set_ui(self, index: int, value: int) -> bool:
        arr = (ctypes.c_int16 * 1)(value)
        return bool(self._lib.crp_set_ui(self._ctx, index, arr, 1))

    # ── Position variables: GP / GJ ───────────────────────────────────────────

    def get_gp(self, index: int) -> RobotPosition:
        pos6 = d6(); ext = d6(); cfg = i4()
        ok = self._lib.crp_get_gp(self._ctx, index, pos6, ext, cfg)
        if not ok:
            raise RuntimeError(f"get_gp({index}) failed")
        return RobotPosition.from_flat(list(pos6), list(ext), list(cfg))

    def set_gp(self, index: int, pos: RobotPosition) -> bool:
        pos6 = d6(pos.to_pos6()); ext = d6(pos.ext_joint); cfg = i4(pos.cfg)
        return bool(self._lib.crp_set_gp(self._ctx, index, pos6, ext, cfg))

    def clear_gp(self) -> bool:
        return bool(self._lib.crp_clear_gp(self._ctx))

    def get_gj(self, index: int) -> JointPosition:
        body = d6(); ext = d6(); cfg = i4()
        ok = self._lib.crp_get_gj(self._ctx, index, body, ext, cfg)
        if not ok:
            raise RuntimeError(f"get_gj({index}) failed")
        return JointPosition(list(body), list(ext), list(cfg))

    def set_gj(self, index: int, jp: JointPosition) -> bool:
        body = d6(jp.body); ext = d6(jp.ext); cfg = i4(jp.cfg)
        return bool(self._lib.crp_set_gj(self._ctx, index, body, ext, cfg))

    def clear_gj(self) -> bool:
        return bool(self._lib.crp_clear_gj(self._ctx))

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        connected = "connected" if self.is_connected() else "disconnected"
        return f"<Robot {connected}>"
