///
/// @file robot_bridge.cpp
/// C bridge layer wrapping CrobotpOS C++ virtual interfaces into plain C functions
/// for Python ctypes consumption.
///

// glibc 2.31 compatibility: provide the __libc_single_threaded symbol that
// libRobotService.so (compiled for glibc 2.32+) references.
// This bridge must be loaded with RTLD_GLOBAL so the symbol becomes visible
// when CSDKLoader calls dlopen(libRobotService.so, RTLD_LAZY).
extern "C" {
    int __libc_single_threaded __attribute__((weak)) = 0;
}

#include <cstring>
#include <cstdint>
#include <cstddef>

#include "CSDKLoader.h"
#include "IRobotService.h"
#include "IIOService.h"
#include "IMotionService.h"
#include "IModelService.h"
#include "IFileService.h"

// ── Context ──────────────────────────────────────────────────────────────────

struct BridgeCtx {
    Crp::CSDKLoader*     loader  = nullptr;
    Crp::IRobotService*  robot   = nullptr;
    Crp::IIOService*     io      = nullptr;
    Crp::IMotionService* motion  = nullptr;
    Crp::IModelService*  model   = nullptr;
    Crp::IFileService*   file    = nullptr;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static inline BridgeCtx* BC(void* ctx) { return static_cast<BridgeCtx*>(ctx); }

static Crp::SRobotPosition make_rpos(const double pos6[6], const double ext[6], const int cfg[4]) {
    Crp::SRobotPosition p;
    p.x  = pos6[0]; p.y  = pos6[1]; p.z  = pos6[2];
    p.Rx = pos6[3]; p.Ry = pos6[4]; p.Rz = pos6[5];
    for (int i = 0; i < 6; ++i) p.extJoint[i] = ext[i];
    for (int i = 0; i < 4; ++i) p.cfg[i] = cfg[i];
    return p;
}

static void unpack_rpos(const Crp::SRobotPosition& p, double pos6[6], double ext[6], int cfg[4]) {
    pos6[0]=p.x; pos6[1]=p.y; pos6[2]=p.z;
    pos6[3]=p.Rx; pos6[4]=p.Ry; pos6[5]=p.Rz;
    if (ext) for (int i=0;i<6;++i) ext[i]=p.extJoint[i];
    if (cfg) for (int i=0;i<4;++i) cfg[i]=p.cfg[i];
}

static Crp::SJointPosition make_jpos(const double body[6], const double ext[6], const int cfg[4]) {
    Crp::SJointPosition j;
    for (int i=0;i<6;++i) j.body[i]=body[i];
    for (int i=0;i<6;++i) j.ext[i]=ext[i];
    if (cfg) for (int i=0;i<4;++i) j.cfg[i]=cfg[i];
    else for (int i=0;i<4;++i) j.cfg[i]=0;
    return j;
}

static void unpack_jpos(const Crp::SJointPosition& j, double body[6], double ext[6], int cfg[4]) {
    for (int i=0;i<6;++i) body[i]=j.body[i];
    if (ext) for (int i=0;i<6;++i) ext[i]=j.ext[i];
    if (cfg) for (int i=0;i<4;++i) cfg[i]=j.cfg[i];
}

// ── Public C API ──────────────────────────────────────────────────────────────

extern "C" {

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void* crp_create(const char* so_path) {
    BridgeCtx* ctx = new BridgeCtx();
    ctx->loader = new Crp::CSDKLoader(so_path ? so_path : ROBOT_SERVICE_NAME);
    if (!ctx->loader->initialize()) {
        delete ctx->loader;
        delete ctx;
        return nullptr;
    }
    ctx->robot  = ctx->loader->getService<Crp::IRobotService>(ID_ROBOT_SERVICE);
    ctx->io     = ctx->loader->getService<Crp::IIOService>(ID_IO_SERVICE);
    ctx->motion = ctx->loader->getService<Crp::IMotionService>(ID_MOTION_SERVICE);
    ctx->model  = ctx->loader->getService<Crp::IModelService>(ID_MODEL_SERVICE);
    ctx->file   = ctx->loader->getService<Crp::IFileService>(ID_FILE_SERVICE);
    return ctx;
}

void crp_destroy(void* ctx) {
    BridgeCtx* bc = BC(ctx);
    if (bc) {
        if (bc->loader) {
            bc->loader->deinitialize();
            delete bc->loader;
        }
        delete bc;
    }
}

// ── IRobotService ─────────────────────────────────────────────────────────────

bool crp_connect(void* ctx, const char* ip, bool disable_hw) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->connect(ip, disable_hw);
}

bool crp_disconnect(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->disconnect();
}

bool crp_is_connected(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->isConnected();
}

bool crp_servo_on(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->servoPowerOn();
}

bool crp_servo_off(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->servoPowerOff();
}

bool crp_is_servo_on(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->isServoOn();
}

int crp_get_work_mode(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getWorkMode();
}

bool crp_set_work_mode(void* ctx, int mode) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setWorkMode(mode);
}

int crp_get_speed_ratio(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getSpeedRatio();
}

bool crp_set_speed_ratio(void* ctx, int ratio) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setSpeedRatio(ratio);
}

// Returns raw position buffer; coord=0→joint(12 doubles), coord=1/2/3→cartesian(6 doubles)
bool crp_get_position(void* ctx, int coord, double* buf, int count) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->getCurrentPosition(coord, buf, (size_t)count);
}

bool crp_get_joint(void* ctx, double body[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SJointPosition jp;
    if (!bc->robot->getCurrentJoint(jp)) return false;
    unpack_jpos(jp, body, ext, cfg);
    return true;
}

bool crp_start_program(void* ctx, const char* prog, int line) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->startProgram(prog, line);
}

bool crp_stop_program(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->stopProgram();
}

bool crp_resume_program(void* ctx, const char* prog) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->resumeProgram(prog);
}

int crp_get_program_status(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getProgramStatus();
}

int crp_get_program_line(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getProgramLine();
}

const char* crp_get_program_path(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return nullptr;
    return bc->robot->getProgramPath();
}

bool crp_has_error(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->hasError();
}

bool crp_clear_error(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->clearError();
}

bool crp_has_emergency_error(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->hasEmergenceError();
}

bool crp_emergency_stop(void* ctx, bool enable) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->emergencyStop(enable);
}

bool crp_get_error_count(void* ctx, size_t* count) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot || !count) return false;
    return bc->robot->getErrorCount(*count);
}

bool crp_get_error_id(void* ctx, size_t idx, uint32_t* eid) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot || !eid) return false;
    return bc->robot->getErrorId(idx, *eid);
}

bool crp_get_error_message(void* ctx, size_t idx, char* buf, size_t len) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot || !buf) return false;
    return bc->robot->getErrorMessage(idx, buf, len);
}

// ── GI / GR / UI ──────────────────────────────────────────────────────────────

int32_t crp_get_gi_count(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getGICount();
}

bool crp_get_gi(void* ctx, size_t idx, int32_t* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->getGI(idx, data, n);
}

bool crp_set_gi(void* ctx, size_t idx, int32_t* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setGI(idx, data, n);
}

int32_t crp_get_gr_count(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getGRCount();
}

bool crp_get_gr(void* ctx, size_t idx, double* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->getGR(idx, data, n);
}

bool crp_set_gr(void* ctx, size_t idx, double* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setGR(idx, data, n);
}

int32_t crp_get_ui_count(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getUICount();
}

bool crp_get_ui(void* ctx, size_t idx, int16_t* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->getUI(idx, data, n);
}

bool crp_set_ui(void* ctx, size_t idx, int16_t* data, size_t n) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setUI(idx, data, n);
}

// ── GP / GJ ──────────────────────────────────────────────────────────────────

// pos6[6] = x,y,z,Rx,Ry,Rz   ext[6] = extJoint   cfg[4] = cfg

bool crp_get_gp(void* ctx, size_t idx, double pos6[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SRobotPosition rp;
    if (!bc->robot->getGP(idx, rp)) return false;
    unpack_rpos(rp, pos6, ext, cfg);
    return true;
}

bool crp_set_gp(void* ctx, size_t idx, double pos6[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SRobotPosition rp = make_rpos(pos6, ext, cfg);
    return bc->robot->setGP(idx, rp);
}

bool crp_clear_gp(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->clearGP();
}

bool crp_get_gj(void* ctx, size_t idx, double body[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SJointPosition jp;
    if (!bc->robot->getGJ(idx, jp)) return false;
    unpack_jpos(jp, body, ext, cfg);
    return true;
}

bool crp_set_gj(void* ctx, size_t idx, double body[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SJointPosition jp = make_jpos(body, ext, cfg);
    return bc->robot->setGJ(idx, jp);
}

bool crp_clear_gj(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->clearGJ();
}

// ── Jog / manual move ────────────────────────────────────────────────────────

bool crp_jog_move_j(void* ctx, size_t ji, double offset) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->jogMoveJ(ji, offset);
}

bool crp_jog_move_l(void* ctx, size_t pi, double offset) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->jogMoveL(pi, offset);
}

bool crp_move_j_joint(void* ctx, double body[6], double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    Crp::SJointPosition jp = make_jpos(body, ext, cfg);
    return bc->robot->moveJ(jp);
}

bool crp_move_l_cart(void* ctx, double x, double y, double z,
                     double rx, double ry, double rz,
                     double ext[6], int cfg[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return false;
    double pos6[6] = {x,y,z,rx,ry,rz};
    Crp::SRobotPosition rp = make_rpos(pos6, ext, cfg);
    return bc->robot->moveL(rp);
}

bool crp_stop_move(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->stopMove();
}

bool crp_is_moving(void* ctx, bool* out) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot || !out) return false;
    return bc->robot->isMoving(*out);
}

bool crp_enable_cobot_mode(void* ctx, bool enable) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->enableCobotMode(enable);
}

bool crp_is_cobot_mode_enabled(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->isCobotModeEnabled();
}

int crp_get_auto_run_mode(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getAutoRunMode();
}

bool crp_set_auto_run_mode(void* ctx, int mode) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setAutoRunMode(mode);
}

int crp_get_dry_run_mode(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getDryRunMode();
}

bool crp_set_dry_run_mode(void* ctx, int mode) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setDryRunMode(mode);
}

int crp_get_coord_sys(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->robot) return -1;
    return bc->robot->getCoordSys();
}

bool crp_set_coord_sys(void* ctx, int coord) {
    auto* bc = BC(ctx);
    return bc && bc->robot && bc->robot->setCoordSys(coord);
}

// ── IIOService ────────────────────────────────────────────────────────────────

int32_t crp_io_get_x_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getXCount();
}
bool crp_io_get_x(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getX(idx, *val);
}

int32_t crp_io_get_y_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getYCount();
}
bool crp_io_get_y(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getY(idx, *val);
}
bool crp_io_set_y(void* ctx, size_t idx, bool val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setY(idx, val);
}

int32_t crp_io_get_m_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getMCount();
}
bool crp_io_get_m(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getM(idx, *val);
}
bool crp_io_set_m(void* ctx, size_t idx, bool val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setM(idx, val);
}

int32_t crp_io_get_ain_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getAINCount();
}
bool crp_io_get_ain(void* ctx, size_t idx, double* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getAIN(idx, *val);
}
bool crp_io_set_ain(void* ctx, size_t idx, double val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setAIN(idx, val);
}

int32_t crp_io_get_aot_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getAOTCount();
}
bool crp_io_get_aot(void* ctx, size_t idx, double* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getAOT(idx, *val);
}
bool crp_io_set_aot(void* ctx, size_t idx, double val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setAOT(idx, val);
}

int32_t crp_io_get_gin_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getGINCount();
}
bool crp_io_get_gin(void* ctx, size_t idx, uint32_t* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getGIN(idx, *val);
}
bool crp_io_set_gin(void* ctx, size_t idx, uint32_t val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setGIN(idx, val);
}

int32_t crp_io_get_got_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getGOTCount();
}
bool crp_io_get_got(void* ctx, size_t idx, uint32_t* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getGOT(idx, *val);
}
bool crp_io_set_got(void* ctx, size_t idx, uint32_t val) {
    auto* bc = BC(ctx);
    return bc && bc->io && bc->io->setGOT(idx, val);
}

int32_t crp_io_get_sx_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getSXCount();
}
bool crp_io_get_sx(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getSX(idx, *val);
}

int32_t crp_io_get_sy_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getSYCount();
}
bool crp_io_get_sy(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getSY(idx, *val);
}

int32_t crp_io_get_sm_count(void* ctx) {
    auto* bc = BC(ctx); if (!bc || !bc->io) return -1;
    return bc->io->getSMCount();
}
bool crp_io_get_sm(void* ctx, size_t idx, bool* val) {
    auto* bc = BC(ctx); if (!bc || !bc->io || !val) return false;
    return bc->io->getSM(idx, *val);
}

// ── IMotionService ────────────────────────────────────────────────────────────

bool crp_motion_is_available(void* ctx) {
    auto* bc = BC(ctx);
    return bc && bc->motion && bc->motion->isAvailable();
}

bool crp_motion_current_user_pos(void* ctx, double pos6[6], double ext[6], int tool, int user) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SRobotPosition rp;
    if (!bc->motion->currentUserPosition(rp, tool, user)) return false;
    unpack_rpos(rp, pos6, ext, nullptr);
    return true;
}

bool crp_motion_is_ready(void* ctx, int type) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    return bc->motion->isReady(static_cast<Crp::EMotionType>(type));
}

size_t crp_motion_get_max_buf(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return 0;
    return bc->motion->getMaxPathBufferSize();
}

size_t crp_motion_get_avail_buf(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return 0;
    return bc->motion->getAvailPathBufferSize();
}

// flat: n × (body[6] + ext[6]) = n*12 doubles; cfg ignored for path
bool crp_motion_send_path_joint(void* ctx, double* flat, size_t n) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion || !flat) return false;
    Crp::SJointPosition* arr = new Crp::SJointPosition[n];
    for (size_t i = 0; i < n; ++i) {
        double* row = flat + i * 12;
        for (int j = 0; j < 6; ++j) arr[i].body[j] = row[j];
        for (int j = 0; j < 6; ++j) arr[i].ext[j]  = row[6+j];
        for (int j = 0; j < 4; ++j) arr[i].cfg[j]   = 0;
    }
    bool ok = bc->motion->sendPath(arr, n);
    delete[] arr;
    return ok;
}

// flat: n × (pos6[6] + ext[6]) = n*12 doubles; cfg ignored for path
bool crp_motion_send_path_pos(void* ctx, double* flat, size_t n, int tool, int user) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion || !flat) return false;
    Crp::SRobotPosition* arr = new Crp::SRobotPosition[n];
    for (size_t i = 0; i < n; ++i) {
        double* row = flat + i * 12;
        arr[i].x = row[0]; arr[i].y = row[1]; arr[i].z  = row[2];
        arr[i].Rx= row[3]; arr[i].Ry= row[4]; arr[i].Rz = row[5];
        for (int j = 0; j < 6; ++j) arr[i].extJoint[j] = row[6+j];
        for (int j = 0; j < 4; ++j) arr[i].cfg[j] = 0;
    }
    bool ok = bc->motion->sendPath(arr, n, tool, user);
    delete[] arr;
    return ok;
}

int crp_motion_move_path(void* ctx, int ratio) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return -1;
    return (int)bc->motion->movePath(ratio);
}

bool crp_motion_move_abs_j(void* ctx, int idx,
                            double body[6], double ext[6], int cfg[4],
                            double speed, double pl, int smooth, int acc, int dec) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SInstMoveAbsJ inst;
    inst.joint = make_jpos(body, ext, cfg);
    inst.param.speed = speed; inst.param.pl = pl;
    inst.param.smooth = smooth; inst.param.acc = acc; inst.param.dec = dec;
    return bc->motion->moveAbsJ(idx, inst);
}

bool crp_motion_move_j(void* ctx, int idx,
                        double x, double y, double z, double rx, double ry, double rz,
                        double ext[6], int cfg[4],
                        double speed, double pl, int smooth, int acc, int dec) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SInstMoveJ inst;
    double pos6[6] = {x,y,z,rx,ry,rz};
    inst.targetPos = make_rpos(pos6, ext, cfg);
    inst.param.speed = speed; inst.param.pl = pl;
    inst.param.smooth = smooth; inst.param.acc = acc; inst.param.dec = dec;
    return bc->motion->moveJ(idx, inst);
}

bool crp_motion_move_l(void* ctx, int idx,
                        double x, double y, double z, double rx, double ry, double rz,
                        double ext[6], int cfg[4],
                        double speed, double pl, int smooth, int acc, int dec, int strategy) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SInstMoveL inst;
    double pos6[6] = {x,y,z,rx,ry,rz};
    inst.targetPos = make_rpos(pos6, ext, cfg);
    inst.param.speed = speed; inst.param.pl = pl;
    inst.param.smooth = smooth; inst.param.acc = acc; inst.param.dec = dec;
    inst.strategy = static_cast<Crp::EMoveStrategy>(strategy);
    return bc->motion->moveL(idx, inst);
}

bool crp_motion_move_c(void* ctx, int idx,
                        double p2[6], double p2ext[6], int p2cfg[4],
                        double p3[6], double p3ext[6], int p3cfg[4],
                        double speed, double pl, int smooth, int acc, int dec, int strategy) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SInstMoveC inst;
    inst.p2 = make_rpos(p2, p2ext, p2cfg);
    inst.p3 = make_rpos(p3, p3ext, p3cfg);
    inst.param.speed = speed; inst.param.pl = pl;
    inst.param.smooth = smooth; inst.param.acc = acc; inst.param.dec = dec;
    inst.strategy = static_cast<Crp::EMoveStrategy>(strategy);
    return bc->motion->moveC(idx, inst);
}

bool crp_motion_move_jump(void* ctx, int idx,
                           double x, double y, double z, double rx, double ry, double rz,
                           double ext[6], int cfg[4],
                           double speed, double pl, int smooth, int acc, int dec, int strategy,
                           double top, double up, double down) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    Crp::SInstMoveJump inst;
    double pos6[6] = {x,y,z,rx,ry,rz};
    inst.targetPos = make_rpos(pos6, ext, cfg);
    inst.param.speed = speed; inst.param.pl = pl;
    inst.param.smooth = smooth; inst.param.acc = acc; inst.param.dec = dec;
    inst.strategy = static_cast<Crp::EMoveStrategy>(strategy);
    inst.top = top; inst.up = up; inst.down = down;
    return bc->motion->moveJump(idx, inst);
}

int32_t crp_motion_current_index(void* ctx) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return -1;
    return bc->motion->currentIndex();
}

bool crp_motion_finalize(void* ctx, int type) {
    auto* bc = BC(ctx);
    if (!bc || !bc->motion) return false;
    return bc->motion->finalize(static_cast<Crp::EMotionType>(type));
}

// ── IModelService ─────────────────────────────────────────────────────────────

static Crp::SStdDHParam make_dh(const double dh14[14]) {
    Crp::SStdDHParam d;
    d.a1=dh14[0]; d.a2=dh14[1]; d.a3=dh14[2]; d.a4=dh14[3];
    d.a5=dh14[4]; d.a6=dh14[5]; d.a7=dh14[6];
    d.d1=dh14[7]; d.d2=dh14[8]; d.d3=dh14[9]; d.d4=dh14[10];
    d.d5=dh14[11]; d.d6=dh14[12]; d.d7=dh14[13];
    return d;
}

int32_t crp_model_fkine(void* ctx, int model, double dh14[14],
                         double body[6], double ext[6], int cfg_in[4],
                         double pos_out[6], double ext_out[6], int cfg_out[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SStdDHParam dh = make_dh(dh14);
    Crp::SJointPosition jp = make_jpos(body, ext, cfg_in);
    Crp::SRobotPosition rp;
    int32_t r = bc->model->FKine(static_cast<Crp::ERobotModel>(model), dh, jp, rp);
    if (r == 0) unpack_rpos(rp, pos_out, ext_out, cfg_out);
    return r;
}

int32_t crp_model_ikine(void* ctx, int model, double dh14[14],
                         double pos[6], double ext[6], int cfg[4],
                         double body_out[6], double ext_out[6], int cfg_out[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SStdDHParam dh = make_dh(dh14);
    Crp::SRobotPosition rp = make_rpos(pos, ext, cfg);
    Crp::SJointPosition jp;
    int32_t r = bc->model->IKine(static_cast<Crp::ERobotModel>(model), dh, rp, jp);
    if (r == 0) unpack_jpos(jp, body_out, ext_out, cfg_out);
    return r;
}

int32_t crp_model_joint2pos(void* ctx,
                             double body[6], double ext[6], int cfg[4],
                             double pos_out[6], double ext_out[6], int cfg_out[4],
                             int tool, int user, int coord) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SJointPosition jp = make_jpos(body, ext, cfg);
    Crp::SRobotPosition rp;
    int32_t r = bc->model->joint2Position(jp, &rp, tool, user, coord);
    if (r == 0) unpack_rpos(rp, pos_out, ext_out, cfg_out);
    return r;
}

int32_t crp_model_pos2joint(void* ctx,
                             double pos[6], double ext[6], int cfg[4],
                             int tool, int user, int coord,
                             double body_out[6], double ext_out[6], int cfg_out[4]) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SRobotPosition rp = make_rpos(pos, ext, cfg);
    Crp::SJointPosition jp;
    int32_t r = bc->model->position2Joint(rp, tool, user, coord, &jp);
    if (r == 0) unpack_jpos(jp, body_out, ext_out, cfg_out);
    return r;
}

int32_t crp_model_convert_coord(void* ctx,
                                 double pos[6], double ext[6], int cfg[4],
                                 int src_tool, int src_user, int src_coord,
                                 double out_pos[6], double out_ext[6], int out_cfg[4],
                                 int dst_tool, int dst_user, int dst_coord) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SRobotPosition src = make_rpos(pos, ext, cfg);
    Crp::SRobotPosition dst;
    int32_t r = bc->model->convertCoordSys(src, src_tool, src_user, src_coord,
                                            &dst, dst_tool, dst_user, dst_coord);
    if (r == 0) unpack_rpos(dst, out_pos, out_ext, out_cfg);
    return r;
}

int32_t crp_model_calc_cfg(void* ctx,
                            double pos[6], double ext[6], int cfg[4],
                            int tool, int user, int coord) {
    auto* bc = BC(ctx);
    if (!bc || !bc->model) return -1;
    Crp::SRobotPosition rp = make_rpos(pos, ext, cfg);
    int32_t r = bc->model->calcCfg(&rp, tool, user, coord);
    if (r == 0) {
        // write cfg back
        for (int i=0;i<4;++i) cfg[i] = rp.cfg[i];
    }
    return r;
}

// ── IFileService ─────────────────────────────────────────────────────────────

bool crp_file_mkdir(void* ctx, const char* path, bool recursive) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->mkdir(path, recursive);
}

bool crp_file_rmdir(void* ctx, const char* path, bool recursive) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->rmdir(path, recursive);
}

bool crp_file_rename(void* ctx, const char* src, const char* dst) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->rename(src, dst);
}

bool crp_file_copy(void* ctx, const char* src, const char* dst) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->copy(src, dst);
}

bool crp_file_remove(void* ctx, const char* path) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->remove(path);
}

bool crp_file_exists(void* ctx, const char* path) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->exists(path);
}

bool crp_file_upload(void* ctx, const char* local, const char* remote) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->upload(local, remote);
}

bool crp_file_download(void* ctx, const char* remote, const char* local) {
    auto* bc = BC(ctx);
    return bc && bc->file && bc->file->download(remote, local);
}

} // extern "C"
