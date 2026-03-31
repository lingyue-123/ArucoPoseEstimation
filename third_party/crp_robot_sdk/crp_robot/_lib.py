"""
ctypes binding: loads robot_bridge.so and declares all function signatures.
"""
import ctypes
import os
from pathlib import Path

# ── Types ─────────────────────────────────────────────────────────────────────

c_bool   = ctypes.c_bool
c_int    = ctypes.c_int
c_int32  = ctypes.c_int32
c_uint32 = ctypes.c_uint32
c_int16  = ctypes.c_int16
c_size_t = ctypes.c_size_t
c_double = ctypes.c_double
c_char_p = ctypes.c_char_p
c_void_p = ctypes.c_void_p

_D6 = ctypes.c_double * 6
_D12 = ctypes.c_double * 12
_D14 = ctypes.c_double * 14
_I4  = ctypes.c_int * 4

# ── Loader ───────────────────────────────────────────────────────────────────

def load_bridge(bridge_path: str) -> ctypes.CDLL:
    """Load robot_bridge.so and set up all argtypes / restypes.

    The bridge is loaded with RTLD_GLOBAL so that the __libc_single_threaded
    stub defined inside it is available in the global symbol table when
    CSDKLoader subsequently calls dlopen(libRobotService.so, RTLD_LAZY).
    """
    bridge_path = str(Path(bridge_path).resolve())
    lib = ctypes.CDLL(bridge_path, mode=ctypes.RTLD_GLOBAL)
    _configure(lib)
    return lib


def _configure(lib: ctypes.CDLL) -> None:
    # ── Lifecycle ─────────────────────────────────────────────────────────────
    lib.crp_create.argtypes  = [c_char_p]
    lib.crp_create.restype   = c_void_p

    lib.crp_destroy.argtypes = [c_void_p]
    lib.crp_destroy.restype  = None

    # ── IRobotService ─────────────────────────────────────────────────────────
    _b1 = [c_void_p]          # bool(ctx)
    _b2s = [c_void_p, c_char_p, c_bool]  # bool(ctx, str, bool)

    lib.crp_connect.argtypes      = [c_void_p, c_char_p, c_bool];   lib.crp_connect.restype      = c_bool
    lib.crp_disconnect.argtypes   = _b1;                             lib.crp_disconnect.restype   = c_bool
    lib.crp_is_connected.argtypes = _b1;                             lib.crp_is_connected.restype = c_bool
    lib.crp_servo_on.argtypes     = _b1;                             lib.crp_servo_on.restype     = c_bool
    lib.crp_servo_off.argtypes    = _b1;                             lib.crp_servo_off.restype    = c_bool
    lib.crp_is_servo_on.argtypes  = _b1;                             lib.crp_is_servo_on.restype  = c_bool

    lib.crp_get_work_mode.argtypes   = _b1;                          lib.crp_get_work_mode.restype   = c_int
    lib.crp_set_work_mode.argtypes   = [c_void_p, c_int];            lib.crp_set_work_mode.restype   = c_bool
    lib.crp_get_speed_ratio.argtypes = _b1;                          lib.crp_get_speed_ratio.restype = c_int
    lib.crp_set_speed_ratio.argtypes = [c_void_p, c_int];            lib.crp_set_speed_ratio.restype = c_bool

    lib.crp_get_position.argtypes = [c_void_p, c_int,
                                      ctypes.POINTER(c_double), c_int]
    lib.crp_get_position.restype  = c_bool

    lib.crp_get_joint.argtypes = [c_void_p,
                                   ctypes.POINTER(c_double),   # body[6]
                                   ctypes.POINTER(c_double),   # ext[6]
                                   ctypes.POINTER(c_int)]      # cfg[4]
    lib.crp_get_joint.restype  = c_bool

    lib.crp_start_program.argtypes  = [c_void_p, c_char_p, c_int]; lib.crp_start_program.restype  = c_bool
    lib.crp_stop_program.argtypes   = _b1;                          lib.crp_stop_program.restype   = c_bool
    lib.crp_resume_program.argtypes = [c_void_p, c_char_p];         lib.crp_resume_program.restype = c_bool

    lib.crp_get_program_status.argtypes = _b1; lib.crp_get_program_status.restype = c_int
    lib.crp_get_program_line.argtypes   = _b1; lib.crp_get_program_line.restype   = c_int
    lib.crp_get_program_path.argtypes   = _b1; lib.crp_get_program_path.restype   = c_char_p

    lib.crp_has_error.argtypes           = _b1; lib.crp_has_error.restype           = c_bool
    lib.crp_clear_error.argtypes         = _b1; lib.crp_clear_error.restype         = c_bool
    lib.crp_has_emergency_error.argtypes = _b1; lib.crp_has_emergency_error.restype = c_bool
    lib.crp_emergency_stop.argtypes      = [c_void_p, c_bool]; lib.crp_emergency_stop.restype = c_bool

    lib.crp_get_error_count.argtypes   = [c_void_p, ctypes.POINTER(c_size_t)];          lib.crp_get_error_count.restype   = c_bool
    lib.crp_get_error_id.argtypes      = [c_void_p, c_size_t, ctypes.POINTER(c_uint32)]; lib.crp_get_error_id.restype      = c_bool
    lib.crp_get_error_message.argtypes = [c_void_p, c_size_t, c_char_p, c_size_t];       lib.crp_get_error_message.restype = c_bool

    # GI / GR / UI
    lib.crp_get_gi_count.argtypes = _b1; lib.crp_get_gi_count.restype = c_int32
    lib.crp_get_gi.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_int32), c_size_t]; lib.crp_get_gi.restype = c_bool
    lib.crp_set_gi.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_int32), c_size_t]; lib.crp_set_gi.restype = c_bool

    lib.crp_get_gr_count.argtypes = _b1; lib.crp_get_gr_count.restype = c_int32
    lib.crp_get_gr.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_double), c_size_t]; lib.crp_get_gr.restype = c_bool
    lib.crp_set_gr.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_double), c_size_t]; lib.crp_set_gr.restype = c_bool

    lib.crp_get_ui_count.argtypes = _b1; lib.crp_get_ui_count.restype = c_int32
    lib.crp_get_ui.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_int16), c_size_t]; lib.crp_get_ui.restype = c_bool
    lib.crp_set_ui.argtypes = [c_void_p, c_size_t, ctypes.POINTER(c_int16), c_size_t]; lib.crp_set_ui.restype = c_bool

    # GP / GJ — pos6[6], ext[6], cfg[4]
    _rpos_args = [c_void_p, c_size_t,
                  ctypes.POINTER(c_double),  # pos6[6]
                  ctypes.POINTER(c_double),  # ext[6]
                  ctypes.POINTER(c_int)]     # cfg[4]
    _jpos_args = [c_void_p, c_size_t,
                  ctypes.POINTER(c_double),  # body[6]
                  ctypes.POINTER(c_double),  # ext[6]
                  ctypes.POINTER(c_int)]     # cfg[4]

    lib.crp_get_gp.argtypes = _rpos_args; lib.crp_get_gp.restype = c_bool
    lib.crp_set_gp.argtypes = _rpos_args; lib.crp_set_gp.restype = c_bool
    lib.crp_clear_gp.argtypes = _b1;      lib.crp_clear_gp.restype = c_bool

    lib.crp_get_gj.argtypes = _jpos_args; lib.crp_get_gj.restype = c_bool
    lib.crp_set_gj.argtypes = _jpos_args; lib.crp_set_gj.restype = c_bool
    lib.crp_clear_gj.argtypes = _b1;      lib.crp_clear_gj.restype = c_bool

    # Jog / manual move
    lib.crp_jog_move_j.argtypes = [c_void_p, c_size_t, c_double]; lib.crp_jog_move_j.restype = c_bool
    lib.crp_jog_move_l.argtypes = [c_void_p, c_size_t, c_double]; lib.crp_jog_move_l.restype = c_bool

    lib.crp_move_j_joint.argtypes = [c_void_p,
                                      ctypes.POINTER(c_double),  # body[6]
                                      ctypes.POINTER(c_double),  # ext[6]
                                      ctypes.POINTER(c_int)]     # cfg[4]
    lib.crp_move_j_joint.restype  = c_bool

    lib.crp_move_l_cart.argtypes = [c_void_p,
                                     c_double, c_double, c_double,  # x y z
                                     c_double, c_double, c_double,  # rx ry rz
                                     ctypes.POINTER(c_double),      # ext[6]
                                     ctypes.POINTER(c_int)]         # cfg[4]
    lib.crp_move_l_cart.restype  = c_bool

    lib.crp_stop_move.argtypes         = _b1; lib.crp_stop_move.restype         = c_bool
    lib.crp_is_moving.argtypes         = [c_void_p, ctypes.POINTER(c_bool)]; lib.crp_is_moving.restype = c_bool
    lib.crp_enable_cobot_mode.argtypes = [c_void_p, c_bool]; lib.crp_enable_cobot_mode.restype = c_bool
    lib.crp_is_cobot_mode_enabled.argtypes = _b1; lib.crp_is_cobot_mode_enabled.restype = c_bool

    lib.crp_get_auto_run_mode.argtypes = _b1;             lib.crp_get_auto_run_mode.restype = c_int
    lib.crp_set_auto_run_mode.argtypes = [c_void_p, c_int]; lib.crp_set_auto_run_mode.restype = c_bool
    lib.crp_get_dry_run_mode.argtypes  = _b1;             lib.crp_get_dry_run_mode.restype  = c_int
    lib.crp_set_dry_run_mode.argtypes  = [c_void_p, c_int]; lib.crp_set_dry_run_mode.restype  = c_bool
    lib.crp_get_coord_sys.argtypes     = _b1;             lib.crp_get_coord_sys.restype     = c_int
    lib.crp_set_coord_sys.argtypes     = [c_void_p, c_int]; lib.crp_set_coord_sys.restype     = c_bool

    # ── IIOService ────────────────────────────────────────────────────────────
    _io_cnt  = _b1               # int32(ctx)
    _io_get  = [c_void_p, c_size_t, ctypes.POINTER(c_bool)]
    _io_getd = [c_void_p, c_size_t, ctypes.POINTER(c_double)]
    _io_getu = [c_void_p, c_size_t, ctypes.POINTER(c_uint32)]

    for name in ('x', 'y', 'm', 'ain', 'aot', 'gin', 'got', 'sx', 'sy', 'sm'):
        fn = getattr(lib, f'crp_io_get_{name}_count')
        fn.argtypes = _io_cnt; fn.restype = c_int32

    for name in ('x', 'y', 'm', 'sx', 'sy', 'sm'):
        fn = getattr(lib, f'crp_io_get_{name}')
        fn.argtypes = _io_get; fn.restype = c_bool

    for name in ('ain', 'aot'):
        fn = getattr(lib, f'crp_io_get_{name}')
        fn.argtypes = _io_getd; fn.restype = c_bool

    for name in ('gin', 'got'):
        fn = getattr(lib, f'crp_io_get_{name}')
        fn.argtypes = _io_getu; fn.restype = c_bool

    lib.crp_io_set_y.argtypes   = [c_void_p, c_size_t, c_bool];   lib.crp_io_set_y.restype   = c_bool
    lib.crp_io_set_m.argtypes   = [c_void_p, c_size_t, c_bool];   lib.crp_io_set_m.restype   = c_bool
    lib.crp_io_set_ain.argtypes = [c_void_p, c_size_t, c_double]; lib.crp_io_set_ain.restype = c_bool
    lib.crp_io_set_aot.argtypes = [c_void_p, c_size_t, c_double]; lib.crp_io_set_aot.restype = c_bool
    lib.crp_io_set_gin.argtypes = [c_void_p, c_size_t, c_uint32]; lib.crp_io_set_gin.restype = c_bool
    lib.crp_io_set_got.argtypes = [c_void_p, c_size_t, c_uint32]; lib.crp_io_set_got.restype = c_bool

    # ── IMotionService ────────────────────────────────────────────────────────
    lib.crp_motion_is_available.argtypes    = _b1; lib.crp_motion_is_available.restype    = c_bool
    lib.crp_motion_is_ready.argtypes        = [c_void_p, c_int]; lib.crp_motion_is_ready.restype = c_bool
    lib.crp_motion_get_max_buf.argtypes     = _b1; lib.crp_motion_get_max_buf.restype     = c_size_t
    lib.crp_motion_get_avail_buf.argtypes   = _b1; lib.crp_motion_get_avail_buf.restype   = c_size_t
    lib.crp_motion_current_index.argtypes   = _b1; lib.crp_motion_current_index.restype   = c_int32

    lib.crp_motion_current_user_pos.argtypes = [c_void_p,
                                                 ctypes.POINTER(c_double),  # pos6[6]
                                                 ctypes.POINTER(c_double),  # ext[6]
                                                 c_int, c_int]
    lib.crp_motion_current_user_pos.restype  = c_bool

    # flat = double*(n*12)
    lib.crp_motion_send_path_joint.argtypes = [c_void_p, ctypes.POINTER(c_double), c_size_t]
    lib.crp_motion_send_path_joint.restype  = c_bool

    lib.crp_motion_send_path_pos.argtypes   = [c_void_p, ctypes.POINTER(c_double), c_size_t, c_int, c_int]
    lib.crp_motion_send_path_pos.restype    = c_bool

    lib.crp_motion_move_path.argtypes = [c_void_p, c_int]; lib.crp_motion_move_path.restype = c_int
    lib.crp_motion_finalize.argtypes  = [c_void_p, c_int]; lib.crp_motion_finalize.restype  = c_bool

    lib.crp_motion_move_abs_j.argtypes = [
        c_void_p, c_int,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),  # body ext cfg
        c_double, c_double, c_int, c_int, c_int,  # speed pl smooth acc dec
    ]; lib.crp_motion_move_abs_j.restype = c_bool

    lib.crp_motion_move_j.argtypes = [
        c_void_p, c_int,
        c_double, c_double, c_double, c_double, c_double, c_double,  # x y z rx ry rz
        ctypes.POINTER(c_double), ctypes.POINTER(c_int),             # ext cfg
        c_double, c_double, c_int, c_int, c_int,                     # speed pl smooth acc dec
    ]; lib.crp_motion_move_j.restype = c_bool

    lib.crp_motion_move_l.argtypes = [
        c_void_p, c_int,
        c_double, c_double, c_double, c_double, c_double, c_double,
        ctypes.POINTER(c_double), ctypes.POINTER(c_int),
        c_double, c_double, c_int, c_int, c_int,
        c_int,  # strategy
    ]; lib.crp_motion_move_l.restype = c_bool

    lib.crp_motion_move_c.argtypes = [
        c_void_p, c_int,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),  # p2
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),  # p3
        c_double, c_double, c_int, c_int, c_int,
        c_int,  # strategy
    ]; lib.crp_motion_move_c.restype = c_bool

    lib.crp_motion_move_jump.argtypes = [
        c_void_p, c_int,
        c_double, c_double, c_double, c_double, c_double, c_double,
        ctypes.POINTER(c_double), ctypes.POINTER(c_int),
        c_double, c_double, c_int, c_int, c_int,
        c_int,   # strategy
        c_double, c_double, c_double,  # top up down
    ]; lib.crp_motion_move_jump.restype = c_bool

    # ── IModelService ─────────────────────────────────────────────────────────
    _model_rpos_io = [
        c_void_p, c_int,                            # ctx, model
        ctypes.POINTER(c_double),                   # dh[14]
        ctypes.POINTER(c_double),                   # body[6]
        ctypes.POINTER(c_double),                   # ext[6]
        ctypes.POINTER(c_int),                      # cfg_in[4]
        ctypes.POINTER(c_double),                   # pos_out[6]
        ctypes.POINTER(c_double),                   # ext_out[6]
        ctypes.POINTER(c_int),                      # cfg_out[4]
    ]
    lib.crp_model_fkine.argtypes = _model_rpos_io; lib.crp_model_fkine.restype = c_int32

    lib.crp_model_ikine.argtypes = [
        c_void_p, c_int,
        ctypes.POINTER(c_double),                   # dh[14]
        ctypes.POINTER(c_double),                   # pos[6]
        ctypes.POINTER(c_double),                   # ext[6]
        ctypes.POINTER(c_int),                      # cfg[4]
        ctypes.POINTER(c_double),                   # body_out[6]
        ctypes.POINTER(c_double),                   # ext_out[6]
        ctypes.POINTER(c_int),                      # cfg_out[4]
    ]; lib.crp_model_ikine.restype = c_int32

    lib.crp_model_joint2pos.argtypes = [
        c_void_p,
        ctypes.POINTER(c_double),                   # body[6]
        ctypes.POINTER(c_double),                   # ext[6]
        ctypes.POINTER(c_int),                      # cfg[4]
        ctypes.POINTER(c_double),                   # pos_out[6]
        ctypes.POINTER(c_double),                   # ext_out[6]
        ctypes.POINTER(c_int),                      # cfg_out[4]
        c_int, c_int, c_int,                        # tool user coord
    ]; lib.crp_model_joint2pos.restype = c_int32

    lib.crp_model_pos2joint.argtypes = [
        c_void_p,
        ctypes.POINTER(c_double),                   # pos[6]
        ctypes.POINTER(c_double),                   # ext[6]
        ctypes.POINTER(c_int),                      # cfg[4]
        c_int, c_int, c_int,                        # tool user coord
        ctypes.POINTER(c_double),                   # body_out[6]
        ctypes.POINTER(c_double),                   # ext_out[6]
        ctypes.POINTER(c_int),                      # cfg_out[4]
    ]; lib.crp_model_pos2joint.restype = c_int32

    lib.crp_model_convert_coord.argtypes = [
        c_void_p,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),
        c_int, c_int, c_int,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),
        c_int, c_int, c_int,
    ]; lib.crp_model_convert_coord.restype = c_int32

    lib.crp_model_calc_cfg.argtypes = [
        c_void_p,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_int),
        c_int, c_int, c_int,
    ]; lib.crp_model_calc_cfg.restype = c_int32

    # ── IFileService ──────────────────────────────────────────────────────────
    lib.crp_file_mkdir.argtypes    = [c_void_p, c_char_p, c_bool]; lib.crp_file_mkdir.restype    = c_bool
    lib.crp_file_rmdir.argtypes    = [c_void_p, c_char_p, c_bool]; lib.crp_file_rmdir.restype    = c_bool
    lib.crp_file_rename.argtypes   = [c_void_p, c_char_p, c_char_p]; lib.crp_file_rename.restype = c_bool
    lib.crp_file_copy.argtypes     = [c_void_p, c_char_p, c_char_p]; lib.crp_file_copy.restype   = c_bool
    lib.crp_file_remove.argtypes   = [c_void_p, c_char_p];           lib.crp_file_remove.restype = c_bool
    lib.crp_file_exists.argtypes   = [c_void_p, c_char_p];           lib.crp_file_exists.restype = c_bool
    lib.crp_file_upload.argtypes   = [c_void_p, c_char_p, c_char_p]; lib.crp_file_upload.restype = c_bool
    lib.crp_file_download.argtypes = [c_void_p, c_char_p, c_char_p]; lib.crp_file_download.restype = c_bool


# ── ctypes helpers ────────────────────────────────────────────────────────────

def d6(vals=None) -> ctypes.Array:
    arr = (ctypes.c_double * 6)()
    if vals:
        for i, v in enumerate(vals[:6]): arr[i] = v
    return arr

def i4(vals=None) -> ctypes.Array:
    arr = (ctypes.c_int * 4)()
    if vals:
        for i, v in enumerate(vals[:4]): arr[i] = v
    return arr

def d14(vals=None) -> ctypes.Array:
    arr = (ctypes.c_double * 14)()
    if vals:
        for i, v in enumerate(vals[:14]): arr[i] = v
    return arr

def dn(n: int, vals=None) -> ctypes.Array:
    arr = (ctypes.c_double * n)()
    if vals:
        for i, v in enumerate(list(vals)[:n]): arr[i] = v
    return arr
