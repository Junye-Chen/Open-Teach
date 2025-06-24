
# 项目代码分析
1. 通信层：
  - 使用ZMQ实现实时数据传输
  - 设计清晰的消息协议
2. 控制层：
  - 实现坐标转换核心算法
  - 添加运动滤波和平滑处理
  - 设计分辨率控制机制
3. 安全层：
  - 添加运动边界检查
  - 实现急停机制
  - 添加碰撞检测
## 机器人控制代码
https://github.com/NYU-robot-learning/OpenTeach-Controllers
包含
-  Franka Emika
-  Kinova Jaco
-  Xela Tactile sensors
-  Allegro Hand Curved
https://www.kinovarobotics.com/
https://franka.de/franka-research-3
## 通信架构
```
VR设备 <-> ZMQ通信 <-> 机器人控制端
VR设备 -> 手部关键点数据 -> ZMQ通信 -> 机器人控制端
VR设备 -> 手臂帧数据 -> ZMQ通信 -> 机器人控制端
机器人状态 -> ZMQ通信 -> VR设备
```
## 控制逻辑流
```
teleop.py
  ↓
TeleOperator # openteach/components/initializers.py
  ↓
PiperArmOperator (openteach/components/operators/piper.py)  # 关键控制
  ↓
PiperArm（openteach/robot/piper.py）
  ↓
DexArmControl (openteach/ros_links/piper_control.py)
  ↓
Franka 机器人
```
配置文件：configs/robot/piper.yaml
## 相机配置
openteach/components/sensors/gemini.py
## VR设备通过APK发送：
- 手部关键点数据
- 手臂帧数据
- 控制命令（如暂停/继续）
## 添加piper机器人文件
要在 Open-Teach 中添加任何机器人操作臂或仿真环境，您只需要编写几个包装器（wrapper）。
1. 对于机器人操作臂
- [X] ROS 链接包装器： 您可以通过添加三个包装器将任何机器人连接到 Open-Teach。首先，在 ros_links 目录中添加一个 Python 文件，该文件将包含 DexArmControl 类。您可以通过查看此目录中的任何代码文件来了解任何操作臂的示例。这将建立控制器与 Open-Teach 之间的 ROS 链接。请查看模板文件[here](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/ros_links/ros_link.py)这里。
- [X] 机器人包装器： 您需要为任何机器人编写一个包装器，以获取机器人的基本信息并向机器人发送信息。有关如何创建此机器人包装器的示例，请查看[here](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/robot/robot.py)这里。
- [X] 操作器（Operator）： 这是帮助进行遥操作的包装器。要为此创建新的包装器，此文件中的大部分内容都可以从[here](https://github.com/aadhithya14/Open-Teach/tree/main/openteach/components/operators/template.py)的任何操作臂代码文件中复用。根据您的需求，可能需要相应地调整变换（transformations）。

# 快速开始
## 在VR眼镜中安装APK。
在 Oculus 头显中启动 VR 应用程序后，输入机器人服务器的 IP 地址。确保机器人服务器和 Oculus 头显处于同一wifi网络中。
### 用户界面
单臂 + 机械手
- 操作模式切换: 使用左手进行操作，右手进行关键点流传输。
- 操作模式:
  - 食指捏: 仅手部模式，边框颜色为绿色。
  - 中指捏: 手臂 + 手部模式，边框颜色为蓝色。
  - 无名指捏: 暂停，边框颜色为红色。
  - 小指捏: 分辨率选择，边框颜色为黑色。
使用方法
1. 安装 APK 文件后，您将看到一个带有红色边框的空白屏幕，屏幕上有一个“菜单”按钮。
2. 点击“菜单”按钮（确保您已启用 Oculus 中的手部追踪），您将看到“IP: 未定义”。
3. 点击“更改 IP”并使用下拉菜单输入 IP 地址（VR 和机器人应处于同一网络提供商下）。
4. 输入 IP 地址后，返回您点击“更改 IP”的屏幕，然后点击“流”。
5. 屏幕边框将变为绿色，此时您的应用程序已准备好进行关键点流传输。

## 连接机器人到PC:
```
cd workspace/piper_sdk/
sudo ethtool -i can0 | grep bus
bash can_activate.sh can0 1000000 
```
然后运行以下命令启动遥操作：
```
cd workspace/Open-Teach
python teleop.py robot=piper
```

### 关节正常运行信息
```
time stamp:1749799030.5636663
Hz:3.0
Control Mode: 1
Arm Status: 0
Mode Feed: 0
Teach Status: 0
Motion Status: 0
Trajectory Num: 0
Error Code: 0
Error Status:
 Joint 1 Angle Limit Status: False
 Joint 2 Angle Limit Status: False
 Joint 3 Angle Limit Status: False
 Joint 4 Angle Limit Status: False
 Joint 5 Angle Limit Status: False
 Joint 6 Angle Limit Status: False
 Joint 1 Communication Status: False
 Joint 2 Communication Status: False
 Joint 3 Communication Status: False
 Joint 4 Communication Status: False
 Joint 5 Communication Status: False
 Joint 6 Communication Status: False

time stamp:1749799030.5647788
Hz:3.0
ArmMsgJointFeedBack:
Joint 1:0, 0.000
Joint 2:18207, 18.207
Joint 3:-13674, -13.674
Joint 4:0, 0.000
Joint 5:-4526, -4.526
Joint 6:0, 0.000
```
## 录制数据
```
python data_collect.py robot=piper demo_num=数据编号
```

# 错误汇总
## 需要使用电脑的wifi IP
这里看到有线网和无线网不是同一个局域网，需要使头显和电脑（服务端）在同一个网段：
```
wlo1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.1.119  netmask 255.255.254.0  broadcast 192.168.1.255
enp4s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.0.23  netmask 255.255.254.0  broadcast 192.168.1.255
```
在头显中输入`192.168.1.119`进行连接。