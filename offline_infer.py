import time
from openteach.utils.timer import FrequencyTimer

# 保留真实机器人和相机接口
from openteach.robot.piper import PiperArm

"""
离线机器人操作测试文件,读取保存的txt文件,执行机器人操作
"""

robot = PiperArm()
timer = FrequencyTimer(25)
with open('/home/eigindustry/workspace/Open-Teach/inference/actions/result_policy_last.txt', 'r') as f:
    for line in f.readlines():
        timer.start_loop()
        line = line.strip()
        data = line.split(",")
        position = [float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])]
        robot.move(position[:6])
        robot.set_gripper(position[6])
        timer.end_loop()



# position = [0, 70000, -44000, 0, 44000, 0, 80000]
# robot.move(position[:6])