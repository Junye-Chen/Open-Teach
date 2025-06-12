import numpy as np
import matplotlib.pyplot as plt
import zmq

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
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv

# from plot import draw_3d_curve, plot_realtime_coordinates


np.set_printoptions(precision=2, suppress=True)

# Filter for removing noise in the teleoperation
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_quat(
            np.stack([self.ori_state, next_state[3:7]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])


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
        # # Gripper Publisher
        # self.gripper_publisher = ZMQKeypointPublisher(
        #     host=host,
        #     port=gripper_port
        # )
        # # Cartesian Publisher
        # self.cartesian_publisher = ZMQKeypointPublisher(
        #     host=host,
        #     port=cartesian_publisher_port
        # )
        # # Joint Publisher
        # self.joint_publisher = ZMQKeypointPublisher(
        #     host=host,
        #     port=joint_publisher_port
        # )
        # # Cartesian Command Publisher
        # self.cartesian_command_publisher = ZMQKeypointPublisher(
        #     host=host,
        #     port=cartesian_command_publisher_port
        # )

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

        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.8)
            
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
        self.MAX_DIS = 10.
        self.MAX_ANGLE = 30.


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
    
    # def robot_pose_aa_to_affine(self,pose_aa: np.ndarray) -> np.ndarray:
    #     """Converts a robot pose in axis-angle format to an affine matrix.
    #     Args:
    #         pose_aa (list): [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
    #         x, y, z are in mm and ax, ay, az are in radians.
    #     Returns:
    #         np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
    #     """

    #     rotation = R.from_rotvec(pose_aa[3:]).as_matrix()
    #     translation = np.array(pose_aa[:3]) / SCALE_FACTOR

    #     return np.block([[rotation, translation[:, np.newaxis]],[0, 0, 0, 1]])
    
    # #Function to differentiate between real and simulated robot
    # def return_real(self):
    #     return True

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
    
    # Gets the Scaled Resolution pose
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        current_homo_mat = copy(self.robot.get_pose()['position'])
        current_cart_pose = self._homo2cart(current_homo_mat)
        # print('CURRENT_CART_POSE: {}'.format(current_cart_pose))

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        # print('SCALED_DIFF_IN_TRANSLATION: {}'.format(scaled_diff_in_translation))
        
        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

        return scaled_cart_pose

    # Reset the teleoperation and get the first frame
    def _reset_teleop(self):
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

    # Function to get gripper state from hand keypoints
    def get_gripper_state_from_hand_keypoints(self):
        # 获取手部关键点坐标
        transformed_hand_coords = self._transformed_hand_keypoint_subscriber.recv_keypoints()
        
        # 计算食指指尖和拇指指尖之间的距离
        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['index'][-1]] - 
                                transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        
        thresh = 0.03  # 距离阈值：3cm
        gripper_fl = False
        gripper_degree = None
        
        # 判断是否触发抓取器状态切换
        if distance < thresh:  # 如果距离小于阈值
            self.gripper_cnt += 1
            if self.gripper_cnt == 1:  # 只在第一次检测到时切换状态
                self.prev_gripper_flag = self.gripper_flag  # 保存前一个状态
                self.gripper_flag = not self.gripper_flag   # 切换状态
                gripper_fl = True
                gripper_degree = distance * self.factor  # 转换单位到mm
        else: 
            self.gripper_cnt = 0  # 重置计数器
        
        # 获取当前抓取器状态
        gripper_state = np.asanyarray(self.gripper_flag).reshape(1)[0]
        
        # 判断状态是否发生变化
        status = False
        if gripper_state != self.prev_gripper_flag:
            status = True
        
        return gripper_state, status, gripper_fl, gripper_degree
    
    # 去掉剧烈变化的帧
    def filter_sharp_motion(self, next_state):  
        MAX_DIS = self.MAX_DIS
        MAX_ANGLE = self.MAX_ANGLE

        if self.his_state is None:
            self.his_state = copy(self.robot.get_pose()['position'])

        try:
            H_relative = np.linalg.inv(self.his_state) @ next_state
            R_rel = H_relative[:3, :3]
            t_rel = H_relative[:3, 3]
            
            trace_val = np.trace(R_rel)
            arg_for_acos = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
            angle_rad = np.arccos(arg_for_acos)
            angle_deg = np.rad2deg(angle_rad)

            if np.linalg.norm(t_rel) > MAX_DIS or angle_deg > MAX_ANGLE:
                return self.his_state
            else:
                self.his_state = next_state
                return next_state
            
        except np.linalg.LinAlgError:
            # 如果矩阵求逆失败，也认为是不安全的
            print("矩阵求逆失败，姿态可能无效。")
            return self.his_state
        

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
        # print(self.is_first_frame, self.arm_teleop_state, new_arm_teleop_state)
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
        else:
            moving_hand_frame = self._get_hand_frame()
        self.arm_teleop_state = new_arm_teleop_state
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()
        # print('arm_teleoperation_scale_mode', arm_teleoperation_scale_mode)

        # 设置操作分辨率
        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        # self.resolution_scale = 0.5  # !!!!!!!!!!精细操作!!!!!!!!!!!!

        if moving_hand_frame is None: 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        # print('moving_hand_frame\n', moving_hand_frame)
        # Get the moving hand frame  # 将手部帧转换为齐次变换矩阵
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)
        # print('hand_moving_H\n', self.hand_moving_H)

        # Transformation code
        # 初始手部→当前手部
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI  to P_HH - Point in Inital Hand Frame to Point in current hand Frame
        # 目标手部→当前手部
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        # 初始机械臂→当前机械臂
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH
        # print('hand_init_H\n', self.hand_init_H)
        # print('hand_moving_H\n', self.hand_moving_H)
        # print('H_RI_RH\n', self.robot_init_H)


        # Rotation from allegro to franka
        # 夹抓到Piper臂的固定变换
        H_A_R = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0], # The height of the allegro mount is 6cm
                          [0, 0, 0, 1]])  

        # 计算当前手部相对初始位姿的位姿
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI
        # print('H_HT_HI\n', H_HT_HI)
        # print('H_HT_HI\n', H_HT_HI[:3, 3])
        # 在项目目录下打开一个move.txt文件，记录一千条H_HT_HI[:3, 3]的数据，可视化
        # with open('move.txt', 'a') as f:
        #     f.write(str(H_HT_HH[:3, 3])+'\n')

        # VR和机械臂的坐标系转换
        R_vr2robot = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) 
        H_HT_HI = R_vr2robot @ H_HT_HI @ np.linalg.inv(R_vr2robot)

        # print('H_HT_HI after trans.\n', H_HT_HI)
        # print('H_HT_HI after trans. \n', H_HT_HI[:3, 3])
        # with open('move.txt', 'a') as f:
        #     f.write(str(H_HT_HI[:3, 3])+'\n')

        # 映射到机械臂
        # H_RT_RH = H_RI_RH @ H_A_R @ H_HT_HI @ np.linalg.pinv(H_A_R) # Homo matrix that takes P_RT to P_RH
        H_RT_RH = H_RI_RH @ H_HT_HI  # 相对于末端坐标系的移动，这里的单位是mm，坐标位置而不是控制信号
        # H_RT_RH = H_HT_HI @ H_RI_RH  # 相对于基座坐标系的移动，这里的单位是mm，坐标位置而不是控制信号
        # print('H_RT_RH\n', H_RT_RH)

        # 可以根据需要设置旋转和平移矩阵
        # # This matrix will change accordingly on how you want the rotations to be.
        # H_R_V= np.array([[0 , 0, -1, 0], 
        #                 [0 , -1, 0, 0],
        #                 [-1, 0, 0, 0],
        #                 [0, 0 ,0 , 1]])

        # # This matrix will change accordingly on how you want the translation to be.
        # H_T_V = np.array([[0, 0 ,1, 0],
        #                  [0 ,1, 0, 0],
        #                  [-1, 0, 0, 0],
        #                 [0, 0, 0, 1]])

        # H_HT_HI_r=(pinv(H_R_V) @ H_HT_HI @ H_R_V)[:3,:3] 
        # H_HT_HI_t=(pinv(H_T_V) @ H_HT_HI @ H_T_V)[:3,3]
         
        # relative_affine = np.block(
        # [[ H_HT_HI_r,  H_HT_HI_t.reshape(3, 1)], [0, 0, 0, 1]])
        # target_translation = H_RI_RH[:3,3] + relative_affine[:3,3]

        # target_rotation = H_RI_RH[:3, :3] @ relative_affine[:3,:3]
        # H_RT_RH = np.block(
        #             [[target_rotation, target_translation.reshape(-1, 1)], [0, 0, 0, 1]])

        self.robot_moving_H = copy(H_RT_RH)

        # 避免剧烈运动，需要对计算的位姿去掉剧变的点
        self.robot_moving_H = self.filter_sharp_motion(self.robot_moving_H)

        # Use the resolution scale to get the final cart pose
        final_pose = self._get_scaled_cart_pose(self.robot_moving_H)  # (7,) 位置+姿态四元数
        # final_pose[0:3]=final_pose[0:3]*self.factor  # 是否需要？
        # print('final_pose', final_pose)

        
        # Apply the filter
        if self.use_filter:
            final_pose = self.comp_filter(final_pose)

        # 更新抓取器状态
        gripper_state, status_change, gripper_flag, gripper_degree = self.get_gripper_state_from_hand_keypoints()
        if gripper_flag and status_change:
            # self.gripper_correct_state=gripper_state
            self.robot.set_gripper_state(gripper_state, gripper_degree)
        
        # # We save the states here during teleoperation as saving directly at 90Hz seems to be too fast for XArm.
        # # 发布抓取器和机械臂状态信息
        # self.gripper_publisher.pub_keypoints(self.gripper_correct_state,"gripper_right")
        # position=self.robot.get_cartesian_position()
        # joint_position= self.robot.get_joint_position()
        # self.cartesian_publisher.pub_keypoints(position,"cartesian")
        # self.joint_publisher.pub_keypoints(joint_position,"joint")
        # self.cartesian_command_publisher.pub_keypoints(final_pose, "cartesian")
        # # 连续控制机械臂
        # if self.arm_teleop_state == ARM_TELEOP_CONT and gripper_flag == False:
        #     self.robot.arm_control(final_pose)

        self.robot.arm_control(final_pose)

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
