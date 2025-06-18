import numpy as np
import time
import rospy
from copy import deepcopy as copy
from piper_sdk import *
# import transforms3d as tfs
from scipy.spatial.transform import Rotation, Slerp


class DexArmControl():
    def __init__(self, record_type=None, robot_type='both'):
    # Initialize Controller Specific Information
        # try:
        #         rospy.init_node("dex_arm", disable_signals = True, anonymous = True)
        # except:
        #         pass
        self.factor = 1000
        if robot_type == 'both':
            self._init_allegro_hand_control()
            self._init_robot_control(record_type)
        elif robot_type == '灵巧手':  # TODO
            self._init_allegro_hand_control()
        elif robot_type == 'piper':
            self._init_robot_control(record_type)

    # Controller initializers
    def _init_robot_control(self, record_type=False):
        self.piper = C_PiperInterface_V2("can0")
        self.piper.ConnectPort()
        # !这个函数可以恢复使能，但是会先失能导致机器人掉落
        # 先获取机器人当前姿态，要是没有错误就不用执行这个函数
        print(self.piper.GetArmStatus())
        print(self.piper.GetArmJointMsgs())
        ArmStatus = self.piper.GetArmStatus().arm_status
        print((ArmStatus.arm_status == 0x04 and ArmStatus.motion_status == 0x01))
        print(self.check_motor_enable(self.piper.GetArmLowSpdInfoMsgs()))
        if (ArmStatus.arm_status == 0x04 and ArmStatus.motion_status == 0x01) or self.check_motors(self.piper.GetArmLowSpdInfoMsgs())[0]:
            self.piper.MotionCtrl_1(emergency_stop=0x02, track_ctrl=0x00, grag_teach_ctrl=0x00)
            # self.piper.MotionCtrl_2(0, 0, 0, 0x00)

        self._enable_fun(piper=self.piper)
        self.home_arm()
        if record_type:
            self.piper.StartRecord()


    def _enable_fun(self, piper:C_PiperInterface_V2):        
        '''
        使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                        piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                        piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                        piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                        piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                        piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("使能状态:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,1000,0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        if(elapsed_time_flag):
            print("程序自动使能超时,退出程序")
            exit(0)

    def check_motors(self, info):
        """
        检查所有电机的状态
        返回: (bool, dict) - (是否有错误, 错误详情)
        """
        error_details = {}
        
        for motor_num in range(1, 7):  # 1 到 6
            motor_name = f"motor_{motor_num}"
            motor_status = getattr(getattr(info, motor_name), "foc_status")
            
            # 检查所有可能的错误状态
            error_conditions = {
                'voltage_too_low': motor_status.voltage_too_low,
                'motor_overheating': motor_status.motor_overheating,
                'driver_overcurrent': motor_status.driver_overcurrent,
                'driver_overheating': motor_status.driver_overheating,
                'collision_status': motor_status.collision_status,
                'driver_error_status': motor_status.driver_error_status,
                'stall_status': motor_status.stall_status
            }
            
            # 收集该电机的所有错误
            motor_errors = [error for error, condition in error_conditions.items() if condition]
            if motor_errors:
                error_details[motor_name] = motor_errors
        
        return bool(error_details), error_details

    def _init_allegro_hand_control(self):
        # for dexhand
        self.allegro_joint_state = None
        pass

    def get_hand_state(self):
        if self.allegro_joint_state is None:
            return None

        # raw_joint_state = copy(self.allegro_joint_state)

        # joint_state = dict(
        #     position = np.array(raw_joint_state.position, dtype = np.float32),
        #     velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
        #     effort = np.array(raw_joint_state.effort, dtype = np.float32),
        #     timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        # )
        # return joint_state


    # Commanded joint state is the joint state being sent as an input to the controller
    def get_commanded_robot_state(self):
        raw_joint_state = copy(self.robot_commanded_joint_state)

        joint_state = dict(
            position = np.array(raw_joint_state.position, dtype = np.float32),
            # velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
            # effort = np.array(raw_joint_state.effort, dtype = np.float32),
            # timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        )
        return joint_state
    
    def get_arm_cartesian_state(self):
        msg = self.piper.GetArmEndPoseMsgs()
        current_pos = [msg.end_pose.X_axis, msg.end_pose.Y_axis, msg.end_pose.Z_axis]
        euler_angles = [msg.end_pose.RX_axis,msg.end_pose.RY_axis,msg.end_pose.RZ_axis]
        # euler_angles = np.radians(euler_angles)
        # current_quat = tfs.euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], 'sxyz')
        current_quat = Rotation.from_euler('xyz', euler_angles, degrees=True).as_quat()

        cartesian_state = dict(
            position = np.array(current_pos, dtype=np.float32).flatten(),
            orientation = np.array(current_quat, dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return cartesian_state

    def get_arm_joint_state(self):
        msg = self.piper.GetArmJointMsgs()
        joint_positions = [msg.joint_state.joint_1, msg.joint_state.joint_2, msg.joint_state.joint_3,
                        msg.joint_state.joint_4, msg.joint_state.joint_5, msg.joint_state.joint_6]

        joint_state = dict(
            position = np.array(joint_positions, dtype=np.float32),
            timestamp = time.time()
        )
        return joint_state
    
    def get_arm_cartesian_coords(self):
        msg = self.piper.GetArmEndPoseMsgs()
        current_pos = np.array([msg.end_pose.X_axis, msg.end_pose.Y_axis, msg.end_pose.Z_axis], dtype=np.float32)
        euler_angles = np.array([msg.end_pose.RX_axis,msg.end_pose.RY_axis,msg.end_pose.RZ_axis], dtype=np.float32)
        # euler_angles = np.radians(euler_angles)
        # current_quat = tfs.euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], 'sxyz')
        current_quat = Rotation.from_euler('xyz', euler_angles, degrees=True).as_quat()

        cartesian_coord = np.concatenate(
            [current_pos, current_quat],
            axis=0
        )
        return cartesian_coord
    
    def get_arm_pose(self):
        pose = np.zeros([4,4])
        msg = self.piper.GetArmEndPoseMsgs()
        current_pos = np.array([msg.end_pose.X_axis, msg.end_pose.Y_axis, msg.end_pose.Z_axis], dtype=np.float32)
        current_axis_angle = np.array([msg.end_pose.RX_axis,msg.end_pose.RY_axis,msg.end_pose.RZ_axis], dtype=np.float32)
        current_axis_angle = current_axis_angle / self.factor
        rot_mat = Rotation.from_euler('zyz', current_axis_angle, degrees=True).as_matrix()
        pose[:3, :3] = rot_mat
        pose[:3, 3] = current_pos / self.factor  # 单位是mm
        pose[3, 3] = 1

        pose_state = dict(
            position = np.array(pose, dtype=np.float32),
            timestamp = time.time()
        )

        return pose_state
    
    def get_arm_osc_position(self):
        msg = self.piper.GetArmEndPoseMsgs()
        current_pos = np.array([msg.end_pose.X_axis, msg.end_pose.Y_axis, msg.end_pose.Z_axis], dtype=np.float32)
        current_axis_angle = np.array([msg.end_pose.RX_axis,msg.end_pose.RY_axis,msg.end_pose.RZ_axis], dtype=np.float32)

        osc_position = np.concatenate(
            [current_pos, current_axis_angle],
            axis=0
        )
        
        return osc_position
    
    def get_arm_position(self):
        joint_state = self.get_arm_joint_state()
        return joint_state['position']

    def move_arm_joint(self, joint_angles):        
        current_angles = joint_angles       

        joint_0 = round(current_angles[0])
        joint_1 = round(current_angles[1])
        joint_2 = round(current_angles[2])
        joint_3 = round(current_angles[3])
        joint_4 = round(current_angles[4])
        joint_5 = round(current_angles[5])
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        # 这个应该是轨迹运动，先这样
        # Moving
        # start_pose = self.get_arm_cartesian_coords()
        # poses = generate_cartesian_space_min_jerk(
        #     start = start_pose, 
        #     goal = cartesian_pos, 
        #     time_to_go = duration,
        #     hz = self.franka.control_freq
        # )
        current_status = cartesian_pos

        X = round(current_status[0])
        Y = round(current_status[1])
        Z = round(current_status[2])
        RX = round(current_status[3])
        RY = round(current_status[4])
        RZ = round(current_status[5])
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)

        
    # Get the robot joint/cartesian position
    def get_robot_position(self):
       #Get Robot Position
        pass

    # Get the robot joint velocity
    def get_robot_velocity(self):
        #Get Robot Velocity
        pass

    # Get the robot joint torque
    def get_robot_torque(self):
        # Get torque applied by the robot.
        pass

    # Get the commanded robot joint position
    def get_commanded_robot_joint_position(self):
        pass

    # Movement functions
    def move_robot(self, joint_angles):
        pass

    # Home Robot
    def home_arm(self):
        position = [210.0, 0.0, 220.0, 180, 45.0, 180, 50]
        X = round(position[0]*self.factor)
        Y = round(position[1]*self.factor)
        Z = round(position[2]*self.factor)
        RX = round(position[3]*self.factor)
        RY = round(position[4]*self.factor)
        RZ = round(position[5]*self.factor)
        joint_6 = round(position[6]*self.factor)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    # Reset the Robot
    def reset_robot(self):
        pass

    # Full robot commands
    def move_robot(self, joint_angles, arm_angles):
        pass

    def arm_control(self, arm_pose):  
        """ input: arm_pose [x, y, z, qx, qy, qz, qw] """
        pose_quat = arm_pose[3:]
        pose_angle = Rotation.from_quat(pose_quat).as_euler('zyz', degrees=True)
        target_status = np.concatenate([arm_pose[:3], pose_angle], axis=0)

        arm_status = self.get_arm_osc_position()
        print('  arm_status  ', arm_status/self.factor)
        print('target_status ', target_status)
        
        X = round(target_status[0]*self.factor)
        Y = round(target_status[1]*self.factor)
        Z = round(target_status[2]*self.factor)
        RX = round(target_status[3]*self.factor)
        RY = round(target_status[4]*self.factor)
        RZ = round(target_status[5]*self.factor)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
        

    def set_gripper_state(self, gripper_state, gripper_degree):
        scale = 1
        # if not gripper_state:
        #     return
        ctrl_degree = min(100*self.factor, max(50, int(gripper_degree * scale * self.factor)))
        # print('ctrl_degree', ctrl_degree)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.GripperCtrl(ctrl_degree, 300, 0x01, 0)

        # TODO 改成步进模式控制


    #Home the Robot
    def home_robot(self):
        pass
        # For now we're using cartesian values



if __name__ == "__main__":
    # euler = [16.03, 9.28, 24.99]

    # r = Rotation.from_euler('xyz', euler, degrees=True)  # 顺序和角度
    # quat = r.as_quat()
    # print(quat)
    # euler2 = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
    # print(euler2)

    # robot = DexArmControl()



    # angle0 = np.array([180, 65.0, 180])
    # angle1 = np.array([180, -65.0, 180])

    # quat0 = Rotation.from_euler('xyz', angle0, degrees=True).as_quat()
    # quat1 = Rotation.from_euler('xyz', angle1, degrees=True).as_quat()
    # print(quat0)
    # print(quat1)

    # mat0 = Rotation.from_euler('xyz', angle0, degrees=True).as_matrix()
    # mat1 = Rotation.from_euler('xyz', angle1, degrees=True).as_matrix()
    # # 令这两个矩阵中小于1e-8的元素变为0
    # mat0[np.abs(mat0) < 1e-8] = 0
    # mat1[np.abs(mat1) < 1e-8] = 0
    # print(mat0)
    # print(mat1)



    """
    根据你的指引我修改好了控制部分的代码，然而我发现控制效果不太理想，请你帮我分析一下原因。
    我的机器人末端初始姿态是：[115.0, 0.0, 250.0, 180, 65.0, 180]，分别代表（X,Y,Z,RX,RY,RZ）的值。
    我尝试对其进行控制的时候，发现可能存在控制信号不稳定的问题，以下是一些有用的信息：
    --move pose-- [ 0.26  0.43 -0.21  0.02  0.    0.  ]
    arm_status   [114.99   0.   250.17 180.    65.04 180.  ]
    target_status [ 117.68   -7.77  250.51 -179.09  -56.69 -179.14]

    --move pose-- [-0.63 -0.52 -0.5  -0.07  0.1  -0.07]
    arm_status   [114.99   0.   250.17 180.    65.04 180.  ]
    target_status [ 115.89   -7.51  250.42 -178.68  -56.46 -179.18]

    --move pose-- [ 0.17  0.49 -0.2   0.63  0.08 -1.51]
    arm_status   [114.99   0.   250.17 180.    65.04 180.  ]
    target_status [ 113.1    -8.06  250.59 -178.88  -56.45  176.08]

    --move pose-- [-0.17  0.01  0.    0.16 -0.02 -0.1 ]
    arm_status   [114.99   0.   250.17 180.    65.04 180.  ]
    target_status [112.64  -8.13 250.5  178.85 -57.04 171.73]

    可以看到尽管我的相对位移（move pose）很小，但是target_status却和arm_status（当前状态）有较大的偏差，这导致了目标点不可达。
    请你给出一些解决的办法和相应的代码，谢谢！
    """


    from scipy.spatial.transform import Rotation
    import numpy as np

    class PoseController:
        def __init__(self, euler_order='zyz', degrees=True):
            self.degrees = degrees
            self.euler_order = euler_order        

        def update_with_delta_matrix(self, current_H, delta_matrix_H):
            """
            根据一个微小的变换矩阵来更新姿态
            """
            if current_H.shape == (4, 4) or delta_matrix_H.shape == (4, 4):
                current_matrix = current_H[:3, :3]
                delta_matrix = delta_matrix_H[:3, :3]

            # 1. 初始化：将初始矩阵存为目标四元数
            current_R = Rotation.from_matrix(current_matrix)

            # 2. 增量计算：将增量矩阵转换为增量四元数
            r_delta = Rotation.from_matrix(delta_matrix)

            # 3. 姿态更新：在四元数空间中进行平滑更新
            self.q_target = current_R * r_delta # 使用Scipy的乘法重载

        def get_stable_quat_output(self):
            """
            获取稳定平滑的四元数输出
            """
            return self.q_target.as_quat()

        def get_stable_euler_output(self):
            """
            获取稳定平滑的欧拉角输出
            """
            # 4. 最终转换：仅在输出时才转回欧拉角
            stable_euler = self.q_target.as_euler(self.euler_order, degrees=self.degrees)
            return stable_euler


    def get_arm_pose(euler_pose, order = 'xyz'):        
        current_pos = euler_pose[:3]
        current_axis_angle = euler_pose[3:]
        rot_mat = Rotation.from_euler(order, current_axis_angle, degrees=True).as_matrix()
        pose = np.zeros([4,4])
        pose[:3, :3] = rot_mat
        pose[:3, 3] = current_pos  # 单位是mm
        pose[3, 3] = 1
        return pose


    # --- 使用示例 ---
    # 初始姿态
    initial_pose = np.array([180., 65.04, 180.])
    arm = np.array([124.96,   0.,   250.15, 180.,    65.04, 180.])
    controller = PoseController(euler_order='xyz', degrees=True)
    # 打印矩阵只显示两位小数，不使用科学计数法      
    np.set_printoptions(precision=2)    
    # print('arm\n', get_arm_pose(arm, 'xyz'))
    print('arm\n', get_arm_pose(arm, 'zyz'))

    # 假设这是由噪声产生的微小扰动矩阵
    # (您的例子中 R[0][2] 和 R[2][0] 符号反了)
    # 为了模拟，我们构造一个绕Y轴旋转-130度的矩阵，这和[180, -65, 180]等价
    # delta_R = Rotation.from_euler('zyz', [0, -130, 0], degrees=True).as_matrix()
    # 实际上，扰动可能更复杂，这里我们直接用您给出的"坏"矩阵作为目标
    # 更好的模拟是，用一个微小的扰动矩阵去乘以初始矩阵
    initial_R = Rotation.from_euler('xyz', initial_pose, degrees=True).as_matrix()
    # 假设一个微小的、几乎为单位矩阵的扰动
    # noise_R = Rotation.from_euler('zyx', [0.01, -0.01, 0.01], degrees=True).as_matrix()
    delta_R = np.array([[ 1.,  0.,  0., -0.],
                        [-0.,  1.,  0., -0.],
                        [ 0., -0.,  1.,  0.],
                        [ 0.,  0.,  0.,  1.]])

    controller.update_with_delta_matrix(initial_R, delta_R)

    # 获取稳定输出
    final_euler = controller.get_stable_euler_output()

    # 即使内部四元数变化了，输出的欧拉角也应该是平滑过渡的
    print(f"微小扰动后的稳定欧拉角输出: {final_euler}")
    # 这个输出会非常接近 [180, 65, 180]，而不是跳变到-65


    R_vr2robot = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) 
    R_2 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    R_vr2robot = R_2 @ R_vr2robot
    print(R_vr2robot)
