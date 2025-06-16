import numpy as np
from copy import deepcopy as copy


ARM_TELEOP_CONT = 1
ARM_TELEOP_STOP = 0


def turn_frame_to_homo_mat(self, frame):
        t = frame[0] * self.factor  # 单位是mm
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

def _get_hand_frame(self):
    """
    return sampled hand frame
    [[ 0.3   1.06  0.2 ]  # t
    [-0.81  0.49 -0.33]
    [ 0.59  0.73 -0.35]   # R
    [-0.24  0.56  0.79]]
    """
    # 获取当前手部位姿
    pass

def _reset_teleop(self):
    """
    reset the teleop state
    使用self._get_hand_frame获取第一帧手部姿态作为self.hand_init_H
    """
    pass


def apply_retargeted_angles(self, log=False):
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
        else:
            moving_hand_frame = self._get_hand_frame()
        self.arm_teleop_state = new_arm_teleop_state
        
        if moving_hand_frame is None: 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        
        # Get the moving hand frame  # 将手部帧转换为齐次变换矩阵
        self.hand_moving_H = self.turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code
        # 初始手部→当前手部
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI  to P_HH - Point in Inital Hand Frame to Point in current hand Frame
        # 目标手部→当前手部
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        # 初始机械臂→当前机械臂
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH

        # 计算当前手部相对初始位姿的位姿
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI

        # VR和机械臂的坐标系转换
        R_vr2robot = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # !可能不对，需要验证
        H_HT_HI = R_vr2robot @ H_HT_HI @ np.linalg.inv(R_vr2robot)

        # 映射到机械臂控制
        H_RT_RH = H_RI_RH @ H_HT_HI  # 相对于末端坐标系的移动，这里的单位是mm，坐标位置而不是控制信号
        self.robot_moving_H = copy(H_RT_RH)

        # Use the resolution scale to get the final cart pose
        final_pose = self._get_scaled_cart_pose(self.robot_moving_H)  # (7,) 位置+姿态四元数

        # 更新抓取器状态
        gripper_state, status_change, gripper_flag, gripper_degree = self.get_gripper_state_from_hand_keypoints()
        if status_change is True and gripper_flag:
            self.gripper_correct_state = gripper_state
            self.robot.set_gripper_state(self.gripper_correct_state, gripper_degree)  # 将浮点数转换为整数

        self.robot.arm_control(final_pose)