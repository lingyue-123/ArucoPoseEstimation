"""
Python enums matching CrobotpOS C++ enumerations.
"""
from enum import IntEnum


class RobotMode(IntEnum):
    """ERobotMode — key-switch position."""
    Manual  = 0   # 手动
    Playing = 1   # 自动
    Remote  = 2   # 远程


class CoordSystem(IntEnum):
    """ECoordinateSystem — coordinate frame selector."""
    Joint = 0   # 关节坐标系  (12 values: J1..J6 + ext1..ext6)
    World = 1   # 世界坐标系
    Base  = 2   # 基坐标系
    User  = 3   # 用户坐标系
    Tool  = 4   # 工具坐标系


class ProgramStatus(IntEnum):
    """EProgramStatus — current program execution state."""
    Stop    = 0
    Running = 1
    Pause   = 2


class DryRunMode(IntEnum):
    """EDryRunMode — test-run mode."""
    Single     = 0   # 单行试运行
    Continuous = 1   # 连续试运行
    Empty      = 2   # 空试运行


class AutoRunMode(IntEnum):
    """EAutoRunMode — automatic-run mode."""
    SingleLine     = 0   # 单行
    SingleLoop     = 1   # 单次循环
    ContinuousLoop = 2   # 连续循环


class MotionType(IntEnum):
    """EMotionType — guidance mode."""
    Path        = 0   # 路径引导 (位置流)
    Instruction = 1   # 指令引导


class MovePathResult(IntEnum):
    """EMovePathResult — result of movePath()."""
    Success    = 0
    NotInit    = 1
    BufEmpty   = 2
    IsRunning  = 3
    ParamError = 4


class MoveStrategy(IntEnum):
    """EMoveStrategy — decomposition strategy for MoveL/MoveC/MoveJump."""
    TimeFirst     = 0
    DistanceFirst = 1
    PoseFirst     = 2


class RobotModel(IntEnum):
    """ERobotModel — kinematic model type."""
    Axis6 = 0   # 工业6轴
    Cobot = 1   # 协作6轴


class ErrorCode(IntEnum):
    """EErrorCode — common return codes from model/service calls."""
    OK               = 0x00
    NetworkError     = 0x01
    InvalidParameter = 0x02
    OutOfRange       = 0x03
    NotSupportMethod = 0x06
