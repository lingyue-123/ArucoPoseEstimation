"""
MotionService — wraps IMotionService via the C bridge.

Usage pattern (path guidance):
    ms = robot.motion
    if ms.is_available() and ms.is_ready(MotionType.Path):
        ms.send_path_joint(joints_list)
        ms.move_path(ratio=4)          # 4 × 2 ms = 8 ms interpolation period
        ms.finalize(MotionType.Path)

Usage pattern (instruction guidance):
    ms.move_abs_j(idx, joint, param)
    ms.move_l(idx, target_pos, param)
    ms.finalize(MotionType.Instruction)
"""
import ctypes
from typing import List, Optional, Sequence

from ._lib import d6, i4, dn
from ._types import JointPosition, RobotPosition, MotionParam
from ._enums import MotionType, MoveStrategy, MovePathResult


class MotionService:
    """Wraps IMotionService: path-guidance and instruction-guidance motion."""

    def __init__(self, lib, ctx):
        self._lib = lib
        self._ctx = ctx

    # ── Status ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(self._lib.crp_motion_is_available(self._ctx))

    def is_ready(self, motion_type: MotionType = MotionType.Path) -> bool:
        return bool(self._lib.crp_motion_is_ready(self._ctx, int(motion_type)))

    def get_max_buffer(self) -> int:
        return self._lib.crp_motion_get_max_buf(self._ctx)

    def get_avail_buffer(self) -> int:
        return self._lib.crp_motion_get_avail_buf(self._ctx)

    def current_index(self) -> int:
        return self._lib.crp_motion_current_index(self._ctx)

    # ── Current position ──────────────────────────────────────────────────────

    def current_user_pos(self, tool: int = 0, user: int = 0) -> RobotPosition:
        pos6 = d6(); ext = d6()
        ok = self._lib.crp_motion_current_user_pos(
            self._ctx, pos6, ext, tool, user)
        if not ok:
            raise RuntimeError("crp_motion_current_user_pos failed")
        return RobotPosition.from_flat(list(pos6), list(ext))

    # ── Path guidance ─────────────────────────────────────────────────────────

    def send_path_joint(self, joints: Sequence[JointPosition]) -> bool:
        """
        Send a joint-space path stream.
        Each element is a JointPosition; cfg is ignored.
        flat layout: n × (body[6] + ext[6]) = n*12 doubles
        """
        n = len(joints)
        flat = dn(n * 12)
        for i, jp in enumerate(joints):
            off = i * 12
            for j, v in enumerate(jp.body[:6]): flat[off + j] = v
            for j, v in enumerate(jp.ext[:6]):  flat[off + 6 + j] = v
        return bool(self._lib.crp_motion_send_path_joint(self._ctx, flat, n))

    def send_path_pos(self, positions: Sequence[RobotPosition],
                      tool: int = 0, user: int = 0) -> bool:
        """
        Send a Cartesian-space path stream.
        flat layout: n × (pos6[6] + ext[6]) = n*12 doubles
        """
        n = len(positions)
        flat = dn(n * 12)
        for i, rp in enumerate(positions):
            off = i * 12
            for j, v in enumerate(rp.to_pos6()): flat[off + j] = v
            for j, v in enumerate(rp.ext_joint[:6]): flat[off + 6 + j] = v
        return bool(self._lib.crp_motion_send_path_pos(self._ctx, flat, n, tool, user))

    def move_path(self, ratio: int = 1) -> MovePathResult:
        """
        Start path-guided motion.
        ratio — interpolation multiplier (period = ratio × 2 ms, range 1–50).
        Returns MovePathResult.
        """
        r = self._lib.crp_motion_move_path(self._ctx, ratio)
        return MovePathResult(r)

    # ── Instruction guidance ──────────────────────────────────────────────────

    def move_abs_j(self, index: int, target: JointPosition,
                   param: Optional[MotionParam] = None) -> bool:
        """MoveAbsJ — absolute joint motion."""
        if param is None: param = MotionParam()
        body = d6(target.body); ext = d6(target.ext); cfg = i4(target.cfg)
        return bool(self._lib.crp_motion_move_abs_j(
            self._ctx, index,
            body, ext, cfg,
            param.speed, param.pl, param.smooth, param.acc, param.dec))

    def move_j(self, index: int, target: RobotPosition,
               param: Optional[MotionParam] = None) -> bool:
        """MoveJ — joint-interpolated motion to Cartesian target."""
        if param is None: param = MotionParam()
        ext = d6(target.ext_joint); cfg = i4(target.cfg)
        return bool(self._lib.crp_motion_move_j(
            self._ctx, index,
            target.x, target.y, target.z, target.Rx, target.Ry, target.Rz,
            ext, cfg,
            param.speed, param.pl, param.smooth, param.acc, param.dec))

    def move_l(self, index: int, target: RobotPosition,
               param: Optional[MotionParam] = None,
               strategy: MoveStrategy = MoveStrategy.TimeFirst) -> bool:
        """MoveL — linear Cartesian motion."""
        if param is None: param = MotionParam()
        ext = d6(target.ext_joint); cfg = i4(target.cfg)
        return bool(self._lib.crp_motion_move_l(
            self._ctx, index,
            target.x, target.y, target.z, target.Rx, target.Ry, target.Rz,
            ext, cfg,
            param.speed, param.pl, param.smooth, param.acc, param.dec,
            int(strategy)))

    def move_c(self, index: int,
               p2: RobotPosition, p3: RobotPosition,
               param: Optional[MotionParam] = None,
               strategy: MoveStrategy = MoveStrategy.TimeFirst) -> bool:
        """MoveC — circular arc motion through p2 (intermediate) to p3 (end)."""
        if param is None: param = MotionParam()
        p2pos = d6(p2.to_pos6()); p2ext = d6(p2.ext_joint); p2cfg = i4(p2.cfg)
        p3pos = d6(p3.to_pos6()); p3ext = d6(p3.ext_joint); p3cfg = i4(p3.cfg)
        return bool(self._lib.crp_motion_move_c(
            self._ctx, index,
            p2pos, p2ext, p2cfg,
            p3pos, p3ext, p3cfg,
            param.speed, param.pl, param.smooth, param.acc, param.dec,
            int(strategy)))

    def move_jump(self, index: int, target: RobotPosition,
                  param: Optional[MotionParam] = None,
                  strategy: MoveStrategy = MoveStrategy.TimeFirst,
                  top: float = 100.0, up: float = 50.0, down: float = 50.0) -> bool:
        """MoveJump — arch-shaped motion (pick-and-place)."""
        if param is None: param = MotionParam()
        ext = d6(target.ext_joint); cfg = i4(target.cfg)
        return bool(self._lib.crp_motion_move_jump(
            self._ctx, index,
            target.x, target.y, target.z, target.Rx, target.Ry, target.Rz,
            ext, cfg,
            param.speed, param.pl, param.smooth, param.acc, param.dec,
            int(strategy), top, up, down))

    # ── Finalize ──────────────────────────────────────────────────────────────

    def finalize(self, motion_type: MotionType = MotionType.Instruction) -> bool:
        """Signal end of path/instruction stream; robot advances to next line."""
        return bool(self._lib.crp_motion_finalize(self._ctx, int(motion_type)))
