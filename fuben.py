import numpy as np
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation


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
    与原版一致，获取当前手部姿态。
    """
    pass

### --- 变化开始 --- ###
def _get_robot_frame(self):
    """
    [新增] 获取机器人末端的当前姿态，并转换为4x4齐次矩阵。
    你需要根据你的机器人SDK来实现这个函数。
    """
    # 示例实现，你需要替换为真实代码
    # current_pose = self.robot.get_current_pose() # 假设返回一个 (7,) 的 [pos, quat]
    # t = current_pose[:3]
    # R = ... # 从四元数 current_pose[3:] 转换到旋转矩阵
    # H = np.eye(4)
    # H[:3, :3] = R
    # H[:3, 3] = t
    # return H
    pass

def _reset_teleop(self):
    """
    reset the teleop state.
    不再记录 "init" 状态, 而是记录 "previous" 状态，作为增量计算的起点。
    """
    print("Resetting teleop state...")
    # 获取并记录当前手部姿态作为上一帧
    hand_frame = self._get_hand_frame()
    if hand_frame is None:
        print("Warning: Failed to get hand frame during reset.")
        return None
    self.hand_prev_H = self.turn_frame_to_homo_mat(hand_frame)
    
    # 获取并记录机器人当前姿态作为上一帧
    # 这是实现增量控制的关键一步，确保手和机器人在复位时是对齐的。
    self.robot_prev_H = self._get_robot_frame()
    if self.robot_prev_H is None:
        print("Warning: Failed to get robot frame during reset.")
        return None

    # is_first_frame 标志仍然有用，用来处理第一次启动
    self.is_first_frame = False
    return hand_frame
### --- 变化结束 --- ###


def apply_retargeted_angles(self, log=False):
    """
        控制逻辑（已更新为增量映射）:
        1. 判断是否需要重置。当首次启动或从暂停恢复时，调用_reset_teleop同步手和机器人的当前状态。
        2. 获取当前手部姿态。
        3. 计算当前手部相对于【上一帧手部】的相对位移（增量）。
        4. 将这个增量转换到机器人坐标系下。
        5. 将这个增量应用到【上一帧机器人】的位姿上，得到新的目标位姿。
        6. 发送控制信号到机械臂，并更新上一帧状态。
    """
    new_arm_teleop_state = self._get_arm_teleop_state()

    # 判断是否需要重置 (逻辑不变)
    if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
        # _reset_teleop 现在会同步手和机器人的状态
        moving_hand_frame = self._reset_teleop()
        # 重置后，我们已经有了第一帧作为参考，可以直接返回，等待下一帧进行增量计算
        if moving_hand_frame is None:
            return
    else:
        moving_hand_frame = self._get_hand_frame()

    self.arm_teleop_state = new_arm_teleop_state
    
    if moving_hand_frame is None: 
        return

    ### --- 变化开始 --- ###
    # 将当前手部帧转换为齐次变换矩阵
    hand_current_H = self.turn_frame_to_homo_mat(moving_hand_frame)

    # 计算手部从上一帧到当前帧的增量变换
    # delta_hand = T_prev_current = T_prev_world * T_world_current = inv(T_world_prev) * T_world_current
    delta_hand_H = np.linalg.pinv(self.hand_prev_H) @ hand_current_H

    # VR和机械臂的坐标系转换 (逻辑不变, R_vr2robot 需要被仔细标定)
    R_vr2robot = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    delta_robot_H = R_vr2robot @ delta_hand_H @ np.linalg.inv(R_vr2robot)

    # 将增量应用到机器人上一帧的姿态上，得到新的目标姿态
    # T_robot_target = T_robot_prev * delta_robot
    robot_target_H = self.robot_prev_H @ delta_robot_H
    
    # 使用解析度缩放函数处理新的目标姿态
    final_pose = self._get_scaled_cart_pose(robot_target_H)  # (7,) 位置+姿态四元数

    # 发送控制指令
    self.robot.arm_control(final_pose)

    # 【关键步骤】更新上一帧的状态，为下一次循环做准备
    self.hand_prev_H = copy(hand_current_H)
    self.robot_prev_H = copy(robot_target_H) # 使用目标姿态作为下一帧的机器人前一姿态
    ### --- 变化结束 --- ###

    # 关于抓取器的逻辑可以保持不变
    # gripper_state, status_change, gripper_flag, gripper_degree = self.get_gripper_state_from_hand_keypoints()
    # if status_change is True and gripper_flag:
    #     self.gripper_correct_state = gripper_state
    #     self.robot.set_gripper_state(self.gripper_correct_state, gripper_degree)


    # todo
    # 实现一个四元数到旋转矩阵的转换函数

def homo2cart(homo_mat):
    
    t = homo_mat[:3, 3]
    R = Rotation.from_matrix(
        homo_mat[:3, :3]).as_quat()

    cart = np.concatenate(
        [t, R], axis=0
    )
    return cart

def cart2homo(cart):
    t = cart[:3]
    R = Rotation.from_quat(cart[3:]).as_matrix()
    homo_mat = np.eye(4)
    homo_mat[:3, :3] = R
    homo_mat[:3, 3] = t
    return homo_mat

if __name__ == '__main__':
    # 随机一个四元数
    # [ 0.3   1.06  0.2 ],
    a = np.array([[-0.81,  0.49, -0.33, 0.3],
                  [ 0.59,  0.73, -0.35, 1.06],   
                  [-0.24,  0.56,  0.79, 0.2],
                  [0, 0, 0, 1]])
    
    a[:3, :3] = Rotation.random().as_matrix()
    print(a)
    
    r1 = a[:3, :3]
    r2 = np.transpose(a[:3, :3])
    r = r1@r2
    print(r)
    
    # 转换为旋转矩阵    
    c = homo2cart(a)
    print(c)

    # 转换为齐次矩阵
    b = cart2homo(c)
    print(b)

    """
    输出：
    [[ 1.0051 -0.0047  0.2081]
    [-0.0047  1.0035 -0.0093]
    [ 0.2081 -0.0093  0.9953]]
    [ 0.3         1.06        0.2        -0.28952667  0.10666772  0.94985065
    0.05079415]
    [[-0.82718852 -0.15826002 -0.539178    0.3       ]
    [ 0.03472742 -0.9720839   0.23204933  1.06      ]
    [-0.56085038  0.17322428  0.80959261  0.2       ]
    [ 0.          0.          0.          1.        ]]
    """