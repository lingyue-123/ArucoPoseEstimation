# main.py
from sms_sts_driver import SMSSTSController
import time

# 1. 初始化控制器 (单例模式最佳，这里为了演示直接实例化)
arm_controller = SMSSTSController("/dev/ttysWK1")
arm_controller.connect()

try:
    print("\n=== 等待任务触发 ===")
    # 模拟其他模块完成任务，比如视觉识别到了枪
    task_finished = True 
    while True: 
        if task_finished:
            # 2. 调用接口：去按下/取枪位置
            arm_controller.press_trigger()
            print('按下')
            time.sleep(2)
            # 4. 调用接口：回到原点
            arm_controller.reset_position()
            print('回原')
            
        time.sleep(1)

except KeyboardInterrupt:
    print("程序退出")
finally:
    arm_controller.disconnect()