"""
crp_robot — Python wrapper for the CrobotpOS SDK.

Quick start
-----------
    from crp_robot import Robot
    from crp_robot import RobotPosition, JointPosition, MotionParam, DHParam
    from crp_robot import (RobotMode, CoordSystem, ProgramStatus,
                           DryRunMode, AutoRunMode, MotionType,
                           MovePathResult, MoveStrategy, RobotModel, ErrorCode)

    with Robot("libRobotService.so") as robot:
        robot.connect("192.168.1.1")
        robot.servo_on()
        pos    = robot.get_position()
        joints = robot.get_joint()
        robot.io.set_y(0, True)
        robot.motion.move_abs_j(0, joints)
        robot.file.upload("/local/prog.pro", "/robot/program/prog.pro")
"""

from .robot          import Robot
from .io_service     import IOService
from .motion_service import MotionService
from .model_service  import ModelService
from .file_service   import FileService

from ._types import RobotPosition, JointPosition, MotionParam, DHParam

from ._enums import (
    RobotMode,
    CoordSystem,
    ProgramStatus,
    DryRunMode,
    AutoRunMode,
    MotionType,
    MovePathResult,
    MoveStrategy,
    RobotModel,
    ErrorCode,
)

__all__ = [
    "Robot",
    "IOService",
    "MotionService",
    "ModelService",
    "FileService",
    # types
    "RobotPosition",
    "JointPosition",
    "MotionParam",
    "DHParam",
    # enums
    "RobotMode",
    "CoordSystem",
    "ProgramStatus",
    "DryRunMode",
    "AutoRunMode",
    "MotionType",
    "MovePathResult",
    "MoveStrategy",
    "RobotModel",
    "ErrorCode",
]
