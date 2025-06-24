import numpy as np
import matplotlib.pyplot as plt
import zmq
import cv2

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from copy import deepcopy as copy
from asyncio import threads
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *

from openteach.robot.piper import PiperArm
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from numpy.linalg import pinv

# from plot import draw_3d_curve, plot_realtime_coordinates


np.set_printoptions(precision=2, suppress=True)

# Filter for removing noise in the teleoperation
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state, prev_state=None):
        if prev_state is not None:
            self.pos_state = prev_state[:3]
            self.ori_state = prev_state[3:7]

        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_quat(
            np.stack([self.ori_state, next_state[3:7]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])
    

class PoseController:
    def __init__(self, euler_order='zyz', degrees=True):
        self.degrees = degrees
        self.euler_order = euler_order        

    def update_with_delta_matrix(self, current_H, delta_matrix_H):
        """
        根据一个微小的变换矩阵来更新姿态
        """
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


class PiperArmOperator(Operator):
    def __init__(
        self,
        host,
        transformed_keypoints_port,
        use_filter=False,
        arm_resolution_port = None, 
        gripper_port =None,
        cartesian_publisher_port = None,
        joint_publisher_port = None,
        teleoperation_reset_port = None,
        cartesian_command_publisher_port = None):

        self.notify_component_start('Piper arm operator')
        
        # Transformed Hand Keypoint Subscriber
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_coords'
        )
        # Subscribers for the transformed arm frame
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_frame'
        )

        # Initalizing the robot controller
        self._robot = PiperArm()
        self.resolution_scale = 1 # NOTE: Get this from a socket
        self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont

        # Subscribers for the resolution scale and teleop state
        self._arm_resolution_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = arm_resolution_port,
            topic = 'button'
        )

        self._arm_teleop_state_subscriber = ZMQKeypointSubscriber(
            host = host, 
            port = teleoperation_reset_port,
            topic = 'pause'
        )

        # Robot Initial Frame
        self.robot_init_H = self.robot.get_pose()['position']
        self.is_first_frame = True
        self._timer = FrequencyTimer(VR_FREQ)
        self._pose_stablizer = PoseController()
        self.robot_prev_H = None

        self.use_filter = use_filter
        # motion outlier detection
        self.use_filter0 = True
        if self.use_filter0: print(' #### Using outlier detection filter... #### ')

        if use_filter:
            print(' #### Using filter to smooth actions... #### ')
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.6)

        # Class variables
        self.gripper_flag = 1
        self.pause_flag = 1
        self.prev_pause_flag = 0        
        self.gripper_cnt = 0
        self.prev_gripper_flag = 0
        self.pause_cnt = 0
        self.gripper_correct_state = 1
        self.factor = 1000

        self.his_state = None
        self.MAX_DIS = 5.
        self.MAX_ANGLE = 8.


    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber
    
    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber        
    
    # Function Gets the transformed hand frame    
    def _get_hand_frame(self):
        data = None  # Initialize with a default value
        for i in range(10):
            data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if data is not None:
                break 
        if data is None:
            return None
        return np.asanyarray(data).reshape(4, 3)  # [t:R]
    """
    moving_hand_frame
    [[ 0.3   1.06  0.2 ]  # t
    [-0.81  0.49 -0.33]
    [ 0.59  0.73 -0.35]   # R
    [-0.24  0.56  0.79]]
    """
    
    # Get the resolution scale mode (High or Low)
    def _get_resolution_scale_mode(self):
        # 接收分辨率模式
        data = self._arm_resolution_subscriber.recv_keypoints()
        res_scale = np.asanyarray(data).reshape(1)[0] # Make sure this data is one dimensional
        return res_scale  

    # Get the teleop state (Pause or Continue)
    def _get_arm_teleop_state(self):
        reset_stat = self._arm_teleop_state_subscriber.recv_keypoints()
        reset_stat = np.asanyarray(reset_stat).reshape(1)[0] # Make sure this data is one dimensional
        return reset_stat

    # Converts a frame to a homogenous transformation matrix
    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0] * self.factor  # 单位是mm
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat
    
    # Converts Homogenous Transformation Matrix to Cartesian Coords
    def _homo2cart(self, homo_mat):
        
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_quat()

        cart = np.concatenate(
            [t, R], axis=0
        )
        return cart
    
    def _cart2homo(self, cart):
        t = cart[:3]
        R = Rotation.from_quat(cart[3:]).as_matrix()
        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = R
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1
        return homo_mat

    def _get_scaled_cart_pose(self, moving_robot_quat):
        """
        将四元数姿态转换为笛卡尔坐标，并应用分辨率缩放
        Args:
            moving_robot_quat: [x, y, z, qx, qy, qz, qw] 位置和四元数
        Returns:
            scaled_cart_pose: [x, y, z, qx, qy, qz, qw] 缩放后的位置和四元数
        """
        if moving_robot_quat is None:
            raise ValueError("Invalid input for _get_scaled_cart_pose. moving_robot_quat cannot be None!")
        # 获取当前位置和姿态
        current_homo_mat = copy(self.robot.get_pose()['position'])
        current_cart_pose = self._homo2cart(current_homo_mat)
        
        # 计算位置差异并应用缩放
        diff_in_translation = moving_robot_quat[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        
        # 构建缩放后的姿态
        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = moving_robot_quat[3:]  # 保持四元数不变
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation  # 应用缩放后的位置差异
        
        return scaled_cart_pose
    
    def _get_scaled_euler_pose(self, moving_robot_euler_pose):
        # 这个直接输入的就是欧拉角形式的动作姿态
        diff_in_translation = copy(moving_robot_euler_pose[:3])
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale        
        moving_robot_euler_pose[:3] = scaled_diff_in_translation # Get the scaled translation only

        return moving_robot_euler_pose
    
    # 将输入的4*4位姿矩阵转换成[位置+欧拉角]的形式
    def _get_aa_pose(self, homo_mat, order='xyz'):
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_euler(order, degrees=True)
        # R = np.rad2deg(R)

        aa_pose = np.concatenate(
            [t, R], axis=0
        )
        return aa_pose
    
    
    # Reset the teleoperation and get the first frame
    def _reset_teleop(self):
        # 还是使用绝对初始位置来计算增量
        # Just updates the beginning position of the arm
        print('****** RESETTING TELEOP ****** ')
        self.robot_init_H = self.robot.get_pose()['position']
        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])

        self.is_first_frame = False
        print('****** TELEOP RESETTED ***** ')
        return first_hand_frame
    
    # def _reset_teleop(self):
    #     """
    #     # reset the teleop state.
    #     # 不再记录 "init" 状态, 而是记录 "previous" 状态，作为增量计算的起点。
    #     """
    #     print('****** RESETTING TELEOP ****** ')
    #     # 获取并记录当前手部姿态作为上一帧
    #     hand_frame = self._get_hand_frame()
    #     while hand_frame is None:
    #         hand_frame = self._get_hand_frame()
    #     self.hand_prev_H = self._turn_frame_to_homo_mat(hand_frame)
    #     self.hand_init_t = copy(self.hand_prev_H[:3, 3])
        
    #     # 获取并记录机器人当前姿态作为上一帧
    #     # 这是实现增量控制的关键一步，确保手和机器人在复位时是对齐的。
    #     self.robot_prev_H = self.robot.get_pose()['position']
    #     print("robot init H\n", self.robot_prev_H)
    #     if self.robot_prev_H is None:
    #         print("Warning: Failed to get robot frame during reset.")
    #         return None

    #     # is_first_frame 标志仍然有用，用来处理第一次启动
    #     self.is_first_frame = False
    #     print('****** TELEOP RESETTED ***** ')
    #     return hand_frame

    
    def get_gripper_state_from_hand_keypoints(self):
        # 获取手部关键点坐标
        transformed_hand_coords = self._transformed_hand_keypoint_subscriber.recv_keypoints()
        
        # 计算食指指尖和拇指指尖之间的距离
        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['index'][-1]] - 
                                transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        
        # 设置距离范围
        min_dist = 0.005  # 最小距离 0.5cm
        max_dist = 0.1  # 最大距离 10cm
        
        # 将距离映射到夹抓器开合程度
        gripper_degree = np.clip(distance * self.factor, min_dist * self.factor, max_dist * self.factor)
        
        # 根据距离判断是否夹住
        gripper_state = distance < (min_dist + max_dist) / 2
        
        return gripper_state, True, True, gripper_degree

    # 限制剧烈变化的帧
    def filter_sharp_motion(self, next_state, type='H'):
        MAX_DIS = self.MAX_DIS
        MAX_ANGLE = self.MAX_ANGLE     
        first_flag = False   

        if self.his_state is None:
            self.his_state = copy(self.robot.get_pose()['position'])  # 4*4 matrix
            while self.his_state is None:
                self.his_state = copy(self.robot.get_pose()['position'])  # 4*4 matrix
            first_flag = True

        # print('his_state:\n', self.his_state)
        # print('next_state:\n', next_state)

        if type == 'H':
            if next_state[0, 3] < 0:
                return self.his_state
            try:
                H_relative = np.linalg.inv(self.his_state) @ next_state
                R_rel = H_relative[:3, :3]
                t_rel = H_relative[:3, 3]
                
                trace_val = np.trace(R_rel)
                arg_for_acos = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
                angle_rad = np.arccos(arg_for_acos)
                angle_deg = np.rad2deg(angle_rad)
                first_flag = False

                # 位置clip
                t_norm = np.linalg.norm(t_rel)
                if t_norm > MAX_DIS:
                    t_rel = t_rel / t_norm * MAX_DIS
                # 姿态clip
                if angle_deg > MAX_ANGLE:
                    axis, _ = cv2.Rodrigues(R_rel - np.eye(3)) if 'cv2' in globals() else (np.array([1,0,0]), None)
                    angle_clip_rad = np.deg2rad(MAX_ANGLE)
                    R_clip = Rotation.from_rotvec(axis * angle_clip_rad).as_matrix()
                    R_rel = R_clip
                # 重新组合
                H_relative_clip = np.eye(4)
                H_relative_clip[:3, :3] = R_rel
                H_relative_clip[:3, 3] = t_rel
                next_state_clip = self.his_state @ H_relative_clip
                self.his_state = next_state_clip
                return next_state_clip
                
            except np.linalg.LinAlgError:
                print("矩阵求逆失败，姿态可能无效。")
                return self.his_state
        
        elif type == 'euler':
            if next_state[0] < 0:
                return self.his_state
            if first_flag:
                self.his_state = self._get_aa_pose(self.his_state, 'zyz')
                first_flag = False

            his_quat = Rotation.from_euler('zyz', self.his_state[3:], degrees=True).as_quat()
            next_quat = Rotation.from_euler('zyz', next_state[3:], degrees=True).as_quat()
            dot = np.dot(his_quat, next_quat)
            dif_angle = np.arccos(np.abs(dot))
            if dot < 0: dif_angle = 2*np.pi - dif_angle

            # 位置clip
            pos_diff = next_state[:3] - self.his_state[:3]
            pos_norm = np.linalg.norm(pos_diff)
            if pos_norm > MAX_DIS:
                pos_diff = pos_diff / pos_norm * MAX_DIS
            # 姿态clip
            angle_deg = np.rad2deg(dif_angle)
            if angle_deg > MAX_ANGLE:
                comp_ratio = MAX_ANGLE / angle_deg                
                interp = Slerp([0, 1], Rotation.from_quat(np.stack([self.his_state[3:], next_state[3:7]], axis=0)),)
                next_quat = interp([1 - comp_ratio])[0].as_quat()
                next_euler = Rotation.from_quat(next_quat).as_euler('zyz', degrees=True)
            else:
                next_euler = next_state[3:]
            next_state_clip = np.concatenate([self.his_state[:3] + pos_diff, next_euler])
            self.his_state = next_state_clip
            return next_state_clip
        
        elif type == 'quat':
            if next_state[0] < 0:  # 不允许危险位置
                return self.his_state
            # 转成位置+四元数
            if first_flag:
                self.his_state = self._homo2cart(self.his_state)
                print('his_state', self.his_state)
                first_flag = False

            pos_diff = next_state[:3] - self.his_state[:3]
            pos_norm = np.linalg.norm(pos_diff)
            if pos_norm > MAX_DIS:
                pos_diff = pos_diff / pos_norm * MAX_DIS
            dot = np.dot(self.his_state[3:], next_state[3:])
            dif_angle = np.arccos(np.abs(dot))
            if dot < 0: dif_angle = 2*np.pi - dif_angle
            angle_deg = np.rad2deg(dif_angle)
            if angle_deg > MAX_ANGLE:
                comp_ratio = MAX_ANGLE / angle_deg                
                interp = Slerp([0, 1], Rotation.from_quat(np.stack([self.his_state[3:], next_state[3:7]], axis=0)),)
                next_quat = interp([1 - comp_ratio])[0].as_quat()
            else:
                next_quat = next_state[3:]
            next_state_clip = np.concatenate([self.his_state[:3] + pos_diff, next_quat])
            self.his_state = next_state_clip
            return next_state_clip
        else:
            raise TypeError("Error: Unknown type of input.")


    # Function to apply retargeted angles
    def _apply_retargeted_angles(self, log=False):
        """
            核心控制方法，实现：
                检查遥操作状态，必要时重置
                获取当前手部坐标系
                计算手部运动到机械臂运动的映射
                应用分辨率缩放
                可选地使用滤波器平滑运动
                发送控制指令到机械臂
        """
        # See if there is a reset in the teleop
        new_arm_teleop_state = self._get_arm_teleop_state()
        # 判断是否需要重置
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
            self.robot_prev_H = self.robot_init_H
            self.hand_prev_H = self.hand_init_H
            self.prev_H_HT_HI = self.hand_init_H  # 地一个结果不对
        else:
            moving_hand_frame = self._get_hand_frame()
        self.arm_teleop_state = new_arm_teleop_state

        if moving_hand_frame is None: 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()
        # 设置操作分辨率
        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1.8
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.8

        # Get the moving hand frame  # 将手部帧转换为齐次变换矩阵
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code
        # 初始手部→当前手部
        H_HI_HH = copy(self.hand_prev_H) # Homo matrix that takes P_HI  to P_HH - Point in Inital Hand Frame to Point in current hand Frame
        # 目标手部→当前手部
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        # 初始机械臂→当前机械臂
        H_RI_RH = copy(self.robot_prev_H) # Homo matrix that takes P_RI to P_RH

        # 计算当前手部相对初始位姿的位姿
        H_HT_HI = np.linalg.inv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI
        # print('H_HT_HI\n', H_HT_HI)
        # 在项目目录下打开一个move.txt文件，记录一千条H_HT_HI[:3, 3]的数据，可视化
        # with open('move.txt', 'a') as f:
        #     f.write(str(H_HT_HH[:3, 3])+'\n')

        # VR和机械臂的坐标系转换
        R_vr2robot = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        # R_2 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        H_HT_HI = R_vr2robot @ H_HT_HI @ np.linalg.inv(R_vr2robot)  # 转换到机器人坐标系

        # with open('move.txt', 'a') as f:
        #     f.write(str(H_HT_HI[:3, 3])+'\n')
        
        # print(' --move pose--', self._get_aa_pose(H_HT_HI))
        # diff_H_HT_HI = np.linalg.inv(self.prev_H_HT_HI) @ H_HT_HI  # 计算位姿差分
        # print(' --diff pose--', self._get_aa_pose(diff_H_HT_HI))


        # 映射到机械臂控制
        H_RT_RH = H_RI_RH @ H_HT_HI  # 相对于末端坐标系的移动，这里的单位是mm，坐标位置而不是控制信号
        t_RT_RH = H_RT_RH[:3, 3]

        # 将机械臂映射转换到四元数空间执行
        self._pose_stablizer.update_with_delta_matrix(H_RI_RH, H_HT_HI)
        quat_RT_RH = self._pose_stablizer.get_stable_quat_output()
 
        self.robot_moving_quat = copy(np.concatenate([t_RT_RH, quat_RT_RH], axis=0))
        # print('robot moving quat', self.robot_moving_quat)

        # 避免剧烈运动，需要对计算的位姿去掉剧变的点
        if self.use_filter0:
            pose_type = 'quat'  # 'H' or 'euler' or 'quat'
            self.robot_moving_quat = self.filter_sharp_motion(self.robot_moving_quat, pose_type)
            # print('robot moving quat', self.robot_moving_quat)
            
        # # Use the resolution scale to get the final cart pose
        # print(' --final pose--', self.robot_moving_quat)
        final_pose = self._get_scaled_cart_pose(self.robot_moving_quat)  # (7,) 位置+姿态四元数
        
        # Apply the filter
        if self.use_filter:  # 平滑滤波
            final_pose = self.comp_filter(final_pose, self._homo2cart(self.robot_prev_H))

        # 更新抓取器状态
        gripper_state, status_change, gripper_flag, gripper_degree = self.get_gripper_state_from_hand_keypoints()
        if status_change is True and gripper_flag:
            self.gripper_correct_state = gripper_state
            self.robot.set_gripper_state(self.gripper_correct_state, gripper_degree)  # 将浮点数转换为整数

        self.robot.arm_control(final_pose)  # input: quat

        # # 【关键步骤】更新上一帧的状态，为下一次循环做准备
        # self.hand_prev_H = H_HT_HH
        # self.robot_prev_H = self.robot.get_pose()['position']
        self.prev_H_HT_HI = copy(H_HT_HI)  # 记录上一次的位姿，用于计算差分

    def stream(self):
        self.notify_component_start('{} control'.format(self.robot.name))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        # Assume that the initial position is considered initial after 3 seconds of the start
        while True:
            try:
                if self.robot.get_joint_position() is not None:
                    self.timer.start_loop()

                    # Retargeting function
                    self._apply_retargeted_angles(log=False)

                    self.timer.end_loop()
            except KeyboardInterrupt:
                break

        self.transformed_arm_keypoint_subscriber.stop()
        print('Stopping the teleoperator!')
