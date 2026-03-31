"""
IOService — wraps IIOService via the C bridge.
"""
import ctypes
from ._lib import d6, i4


class IOService:
    """Provides access to digital/analog I/O and group I/O."""

    def __init__(self, lib, ctx):
        self._lib = lib
        self._ctx = ctx

    # ── Digital Inputs (X) ───────────────────────────────────────────────────

    def get_x_count(self) -> int:
        return self._lib.crp_io_get_x_count(self._ctx)

    def get_x(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_x(self._ctx, index, ctypes.byref(val))
        return val.value

    # ── Digital Outputs (Y) ──────────────────────────────────────────────────

    def get_y_count(self) -> int:
        return self._lib.crp_io_get_y_count(self._ctx)

    def get_y(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_y(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_y(self, index: int, value: bool) -> bool:
        return bool(self._lib.crp_io_set_y(self._ctx, index, value))

    # ── Auxiliary Relay (M) ───────────────────────────────────────────────────

    def get_m_count(self) -> int:
        return self._lib.crp_io_get_m_count(self._ctx)

    def get_m(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_m(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_m(self, index: int, value: bool) -> bool:
        return bool(self._lib.crp_io_set_m(self._ctx, index, value))

    # ── Analog Input (AIN) ────────────────────────────────────────────────────

    def get_ain_count(self) -> int:
        return self._lib.crp_io_get_ain_count(self._ctx)

    def get_ain(self, index: int) -> float:
        val = ctypes.c_double(0.0)
        self._lib.crp_io_get_ain(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_ain(self, index: int, value: float) -> bool:
        return bool(self._lib.crp_io_set_ain(self._ctx, index, value))

    # ── Analog Output (AOT) ───────────────────────────────────────────────────

    def get_aot_count(self) -> int:
        return self._lib.crp_io_get_aot_count(self._ctx)

    def get_aot(self, index: int) -> float:
        val = ctypes.c_double(0.0)
        self._lib.crp_io_get_aot(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_aot(self, index: int, value: float) -> bool:
        return bool(self._lib.crp_io_set_aot(self._ctx, index, value))

    # ── Group Input (GIN) ─────────────────────────────────────────────────────

    def get_gin_count(self) -> int:
        return self._lib.crp_io_get_gin_count(self._ctx)

    def get_gin(self, index: int) -> int:
        val = ctypes.c_uint32(0)
        self._lib.crp_io_get_gin(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_gin(self, index: int, value: int) -> bool:
        return bool(self._lib.crp_io_set_gin(self._ctx, index, ctypes.c_uint32(value)))

    # ── Group Output (GOT) ────────────────────────────────────────────────────

    def get_got_count(self) -> int:
        return self._lib.crp_io_get_got_count(self._ctx)

    def get_got(self, index: int) -> int:
        val = ctypes.c_uint32(0)
        self._lib.crp_io_get_got(self._ctx, index, ctypes.byref(val))
        return val.value

    def set_got(self, index: int, value: int) -> bool:
        return bool(self._lib.crp_io_set_got(self._ctx, index, ctypes.c_uint32(value)))

    # ── System Inputs/Outputs (SX / SY / SM) — read-only ─────────────────────

    def get_sx_count(self) -> int:
        return self._lib.crp_io_get_sx_count(self._ctx)

    def get_sx(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_sx(self._ctx, index, ctypes.byref(val))
        return val.value

    def get_sy_count(self) -> int:
        return self._lib.crp_io_get_sy_count(self._ctx)

    def get_sy(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_sy(self._ctx, index, ctypes.byref(val))
        return val.value

    def get_sm_count(self) -> int:
        return self._lib.crp_io_get_sm_count(self._ctx)

    def get_sm(self, index: int) -> bool:
        val = ctypes.c_bool(False)
        self._lib.crp_io_get_sm(self._ctx, index, ctypes.byref(val))
        return val.value
