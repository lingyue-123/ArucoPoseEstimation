import ctypes
import os
import threading
from contextlib import contextmanager


class RobotController:
    def __init__(self, so_path=None):
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.so_path = so_path or os.path.join(self.module_dir, "libforcecontrol_crp.so")
        self.lib = None
        self.running = False
        self.current_thread = None
        self.state_thread = None
        self.state_thread_running = False
        self._cwd_lock = threading.Lock()

    @contextmanager
    def _module_cwd(self):
        prev = os.getcwd()
        with self._cwd_lock:
            os.chdir(self.module_dir)
            try:
                yield
            finally:
                os.chdir(prev)

    def load(self):
        try:
            self.lib = ctypes.CDLL(self.so_path, mode=ctypes.RTLD_GLOBAL)

            self.lib.PowerOn_PPmode.argtypes = []
            self.lib.PowerOn_PPmode.restype = ctypes.c_int

            self.lib.Forcecontrol_ChargeIn.argtypes = []
            self.lib.Forcecontrol_ChargeIn.restype = ctypes.c_int

            self.lib.Forcecontrol_ChargeOut.argtypes = []
            self.lib.Forcecontrol_ChargeOut.restype = ctypes.c_int

            self.lib.ForceControl_demo.argtypes = []
            self.lib.ForceControl_demo.restype = ctypes.c_int

            self.lib.GetForcecontrolState.argtypes = []
            self.lib.GetForcecontrolState.restype = ctypes.c_int

            self.lib.ForceControl_OpenCover.argtypes = []
            self.lib.ForceControl_OpenCover.restype = ctypes.c_int

            self.lib.return_the_gun.argtypes = []
            self.lib.return_the_gun.restype = ctypes.c_int

            self.lib.ForceControl_Poseadjust.argtypes = []
            self.lib.ForceControl_Poseadjust.restype = ctypes.c_int

            self.lib.AttachExternalCrpServices.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.AttachExternalCrpServices.restype = None

            self.lib.DetachExternalCrpServices.argtypes = []
            self.lib.DetachExternalCrpServices.restype = None

            return True
        except OSError as e:
            print(f"加载库失败: {e}")
            return False

    def attach_crp_robot(self, bridge_robot):
        if not self.lib:
            raise RuntimeError("force control library not loaded")
        service_ptrs = bridge_robot.get_service_ptrs()
        self.lib.AttachExternalCrpServices(
            ctypes.c_void_p(service_ptrs["robot"]),
            ctypes.c_void_p(service_ptrs["motion"]),
            ctypes.c_void_p(service_ptrs["file"]),
        )

    def detach_crp_robot(self):
        if self.lib and hasattr(self.lib, "DetachExternalCrpServices"):
            self.lib.DetachExternalCrpServices()

    def _call(self, func_name, title):
        if not self.lib:
            print("请先加载库")
            return -1

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        with self._module_cwd():
            result = getattr(self.lib, func_name)()
        print(f"{func_name} 返回值: {result}")
        return result

    def run_power_on_ppmode(self):
        return self._call("PowerOn_PPmode", "运行 PowerOn_PPmode...")

    def run_forcecontrol_charge_in(self):
        return self._call("Forcecontrol_ChargeIn", "运行 Forcecontrol_ChargeIn (插枪)...")

    def run_forcecontrol_pose_adjust(self):
        return self._call("ForceControl_Poseadjust", "运行 ForceControl_Poseadjust (姿态保持)...")

    def run_forcecontrol_charge_out(self):
        return self._call("Forcecontrol_ChargeOut", "运行 Forcecontrol_ChargeOut (拔枪)...")

    def run_forcecontrol_demo(self):
        return self._call("ForceControl_demo", "运行 ForceControl_demo...")

    def run_ForceControl_OpenCover(self):
        return self._call("ForceControl_OpenCover", "运行 ForceControl_OpenCover...")

    def run_ForceControl_return_the_gun(self):
        return self._call("return_the_gun", "运行 ForceControl_return_the_gun...")

    def get_forcecontrol_state(self):
        if not self.lib or not hasattr(self.lib, "GetForcecontrolState"):
            return -1
        with self._module_cwd():
            return self.lib.GetForcecontrolState()
