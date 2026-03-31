"""
Python dataclasses mirroring CrobotpOS C++ structures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RobotPosition:
    """
    Maps to SRobotPosition.
    pos6 = [x, y, z, Rx, Ry, Rz] (mm / deg)
    ext_joint[6] — external-axis values
    cfg[4]        — [cf1, cf4, cf6, cfx] configuration
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    Rx: float = 0.0
    Ry: float = 0.0
    Rz: float = 0.0
    ext_joint: List[float] = field(default_factory=lambda: [0.0] * 6)
    cfg: List[int]         = field(default_factory=lambda: [0] * 4)

    def to_pos6(self) -> List[float]:
        return [self.x, self.y, self.z, self.Rx, self.Ry, self.Rz]

    @staticmethod
    def from_flat(pos6: List[float],
                  ext: Optional[List[float]] = None,
                  cfg: Optional[List[int]]   = None) -> "RobotPosition":
        return RobotPosition(
            x=pos6[0], y=pos6[1], z=pos6[2],
            Rx=pos6[3], Ry=pos6[4], Rz=pos6[5],
            ext_joint=list(ext) if ext else [0.0]*6,
            cfg=list(cfg) if cfg else [0]*4,
        )


@dataclass
class JointPosition:
    """
    Maps to SJointPosition.
    body[6] — main-axis joint angles (deg)
    ext[6]  — external-axis joint values
    cfg[4]  — configuration (written by robot, ignored on write)
    """
    body: List[float] = field(default_factory=lambda: [0.0] * 6)
    ext:  List[float] = field(default_factory=lambda: [0.0] * 6)
    cfg:  List[int]   = field(default_factory=lambda: [0] * 4)

    @staticmethod
    def from_flat(body: List[float],
                  ext: Optional[List[float]] = None,
                  cfg: Optional[List[int]] = None) -> "JointPosition":
        return JointPosition(
            body=list(body),
            ext=list(ext) if ext else [0.0]*6,
            cfg=list(cfg) if cfg else [0]*4,
        )


@dataclass
class MotionParam:
    """
    Maps to SMotionParam.
    speed  — joint: [1,100] %; cartesian: mm/s
    pl     — smoothing level [0,9]
    smooth — filter level [0,9]
    acc    — acceleration factor [1,20]
    dec    — deceleration factor [1,20]
    """
    speed:  float = 10.0
    pl:     float = 0.0
    smooth: int   = 0
    acc:    int   = 1
    dec:    int   = 1


@dataclass
class DHParam:
    """
    Maps to SStdDHParam (14 values: a1..a7, d1..d7).
    """
    a: List[float] = field(default_factory=lambda: [0.0] * 7)
    d: List[float] = field(default_factory=lambda: [0.0] * 7)

    def to_flat14(self) -> List[float]:
        return list(self.a) + list(self.d)
