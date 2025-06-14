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
        if self.piper.GetArmStatus().arm_status == 0x04:
            self.piper.MotionCtrl_1(emergency_stop=0x02, track_ctrl=0x00, grag_teach_ctrl=0x00)

        self._enable_fun(piper=self.piper)
        self.home_arm()

        # 设置电机角度限制及最大速度
        # self.piper.MotorAngleLimitMaxSpdSet(1, -150, -150, 2000)
        # self.piper.MotorAngleLimitMaxSpdSet(2, 0, 180, 2000)
        # self.piper.MotorAngleLimitMaxSpdSet(3, -170, 0, 2000)
        # self.piper.MotorAngleLimitMaxSpdSet(4, -100, 100, 2000)
        # self.piper.MotorAngleLimitMaxSpdSet(5, -70, 70, 2000)
        # self.piper.MotorAngleLimitMaxSpdSet(6, -120, 120, 2000)

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
        euler_angles = np.array([msg.end_pose.RX_axis,msg.end_pose.RY_axis,msg.end_pose.RZ_axis], dtype=np.float32)
        # euler_angles = np.radians(euler_angles)
        # rot_mat = tfs.euler.euler2mat(euler_angles[0], euler_angles[1], euler_angles[2], 'sxyz')
        rot_mat = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
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
        position = [85.0, 0.0, 275.0, 0, 85.0, 0, 80]
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
        # print(pose_angle)
        # pose_mat = tfs.quaternions.quat2mat(pose_quat)
        # pose_angle = tfs.euler.mat2euler(pose_mat)
        pose_angle = Rotation.from_quat(pose_quat).as_euler('xyz', degrees=True)
        # print('pose_angle', pose_angle)        
        current_status = np.concatenate([arm_pose[:3], pose_angle], axis=0)
        # print('arm_pose', arm_pose[:3])

        arm_status = self.get_arm_osc_position()

        print('  arm_status  ', arm_status/self.factor)
        # print('current_status', current_status)

        # current_status = [95.0, 0.0, 260.0, 0, 85.0, 0, 80]
        print('current_status', current_status)
        
        X = round(current_status[0]*self.factor)
        Y = round(current_status[1]*self.factor)
        Z = round(current_status[2]*self.factor)
        RX = round(current_status[3]*self.factor)
        RY = round(current_status[4]*self.factor)
        RZ = round(current_status[5]*self.factor)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        # self.piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
        

    def set_gripper_state(self, gripper_state, gripper_degree):
        scale = 1.
        # if not gripper_state:
        #     return
        ctrl_degree = min(100*self.factor, max(500, int(gripper_degree * scale * self.factor)))
        # print('ctrl_degree', ctrl_degree)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.GripperCtrl(ctrl_degree, 300, 0x01, 0)

        # TODO 改成步进模式控制


    #Home the Robot
    def home_robot(self):
        pass
        # For now we're using cartesian values



if __name__ == "__main__":
    euler = [16.03, 9.28, 24.99]

    r = Rotation.from_euler('xyz', euler, degrees=True)  # 顺序和角度
    quat = r.as_quat()
    print(quat)
    euler2 = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
    print(euler2)
