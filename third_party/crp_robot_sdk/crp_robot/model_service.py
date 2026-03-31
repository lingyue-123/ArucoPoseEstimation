"""
ModelService — wraps IModelService via the C bridge.

Provides offline forward/inverse kinematics, coordinate conversion, and
cfg calculation.  Network round-trips are required for joint2pos / pos2joint
/ convertCoord / calcCfg (these call the controller).
"""
import ctypes
from typing import Tuple

from ._lib import d6, i4, d14
from ._types import JointPosition, RobotPosition, DHParam
from ._enums import RobotModel, CoordSystem, ErrorCode


class ModelService:
    """Wraps IModelService: kinematics and coordinate-system utilities."""

    def __init__(self, lib, ctx):
        self._lib = lib
        self._ctx = ctx

    # ── Forward kinematics (offline) ─────────────────────────────────────────

    def fkine(self, dh: DHParam, joints: JointPosition,
              model: RobotModel = RobotModel.Axis6) -> RobotPosition:
        """
        Forward kinematics (offline — no tool/user offsets).
        Returns RobotPosition with cfg filled.
        Raises RuntimeError on failure.
        """
        dh14 = d14(dh.to_flat14())
        body = d6(joints.body); ext = d6(joints.ext); cfg_in = i4(joints.cfg)
        pos_out = d6(); ext_out = d6(); cfg_out = i4()
        r = self._lib.crp_model_fkine(
            self._ctx, int(model), dh14,
            body, ext, cfg_in,
            pos_out, ext_out, cfg_out)
        if r != 0:
            raise RuntimeError(f"FKine failed: {r}")
        return RobotPosition.from_flat(list(pos_out), list(ext_out), list(cfg_out))

    # ── Inverse kinematics (offline) ─────────────────────────────────────────

    def ikine(self, dh: DHParam, target: RobotPosition,
              model: RobotModel = RobotModel.Axis6) -> JointPosition:
        """
        Inverse kinematics (offline — no tool/user offsets).
        cfg in target must be set correctly.
        Raises RuntimeError on failure.
        """
        dh14 = d14(dh.to_flat14())
        pos = d6(target.to_pos6()); ext = d6(target.ext_joint); cfg = i4(target.cfg)
        body_out = d6(); ext_out = d6(); cfg_out = i4()
        r = self._lib.crp_model_ikine(
            self._ctx, int(model), dh14,
            pos, ext, cfg,
            body_out, ext_out, cfg_out)
        if r != 0:
            raise RuntimeError(f"IKine failed: {r}")
        return JointPosition.from_flat(list(body_out), list(ext_out), list(cfg_out))

    # ── Joint → Cartesian (online, uses controller) ───────────────────────────

    def joint2pos(self, joints: JointPosition,
                  tool: int = 0, user: int = 0,
                  coord: CoordSystem = CoordSystem.Base) -> RobotPosition:
        """
        Convert joint angles to Cartesian position (network call to controller).
        """
        body = d6(joints.body); ext = d6(joints.ext); cfg = i4(joints.cfg)
        pos_out = d6(); ext_out = d6(); cfg_out = i4()
        r = self._lib.crp_model_joint2pos(
            self._ctx, body, ext, cfg,
            pos_out, ext_out, cfg_out,
            tool, user, int(coord))
        if r != 0:
            raise RuntimeError(f"joint2Position failed: {r}")
        return RobotPosition.from_flat(list(pos_out), list(ext_out), list(cfg_out))

    # ── Cartesian → Joint (online) ────────────────────────────────────────────

    def pos2joint(self, pos: RobotPosition,
                  tool: int = 0, user: int = 0,
                  coord: CoordSystem = CoordSystem.Base) -> JointPosition:
        """
        Convert Cartesian position to joint angles (network call to controller).
        cfg must be set for unique solution selection.
        """
        pos6 = d6(pos.to_pos6()); ext = d6(pos.ext_joint); cfg = i4(pos.cfg)
        body_out = d6(); ext_out = d6(); cfg_out = i4()
        r = self._lib.crp_model_pos2joint(
            self._ctx, pos6, ext, cfg,
            tool, user, int(coord),
            body_out, ext_out, cfg_out)
        if r != 0:
            raise RuntimeError(f"position2Joint failed: {r}")
        return JointPosition.from_flat(list(body_out), list(ext_out), list(cfg_out))

    # ── Coordinate-system conversion (online) ────────────────────────────────

    def convert_coord(self, pos: RobotPosition,
                      src_tool: int, src_user: int, src_coord: CoordSystem,
                      dst_tool: int, dst_user: int, dst_coord: CoordSystem) -> RobotPosition:
        """
        Convert a Cartesian position between coordinate frames.
        """
        src_pos = d6(pos.to_pos6()); src_ext = d6(pos.ext_joint); src_cfg = i4(pos.cfg)
        out_pos = d6(); out_ext = d6(); out_cfg = i4()
        r = self._lib.crp_model_convert_coord(
            self._ctx,
            src_pos, src_ext, src_cfg,
            src_tool, src_user, int(src_coord),
            out_pos, out_ext, out_cfg,
            dst_tool, dst_user, int(dst_coord))
        if r != 0:
            raise RuntimeError(f"convertCoordSys failed: {r}")
        return RobotPosition.from_flat(list(out_pos), list(out_ext), list(out_cfg))

    # ── Calc cfg (online) ─────────────────────────────────────────────────────

    def calc_cfg(self, pos: RobotPosition,
                 tool: int = 0, user: int = 0,
                 coord: CoordSystem = CoordSystem.Base) -> RobotPosition:
        """
        Calculate cfg for a Cartesian position relative to the current robot pose.
        Returns a copy of pos with cfg filled in.
        """
        pos6 = d6(pos.to_pos6()); ext = d6(pos.ext_joint); cfg = i4(pos.cfg)
        r = self._lib.crp_model_calc_cfg(
            self._ctx, pos6, ext, cfg,
            tool, user, int(coord))
        if r != 0:
            raise RuntimeError(f"calcCfg failed: {r}")
        return RobotPosition.from_flat(list(pos6), list(ext), list(cfg))
