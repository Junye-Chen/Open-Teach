# 读取指定文件夹下.h5文件,并转换为字典
import h5py
import numpy as np

def synchronize_and_merge_data(
    cam_video_path: str,
    joint_states_path: str,
    output_path: str
):
    """
    从相机视频和机器人关节状态文件中读取数据，根据时间戳进行对齐和合并，
    并保存为目标HDF5文件格式。

    Args:
        cam_video_path (str): cam_0_rgb_video .h5 文件的路径。
        joint_states_path (str): piper_joint_states .h5 文件的路径。
        output_path (str): 合并后的目标 .h5 文件的保存路径。
    """

    print(f"正在从 {cam_video_path} 读取相机数据...")
    with h5py.File(cam_video_path, 'r') as cam_file:
        cam_rgb_images = cam_file['rgb_images'][()]
        cam_timestamps = cam_file['timestamps'][()]

    print(f"正在从 {joint_states_path} 读取机器人关节状态数据...")
    with h5py.File(joint_states_path, 'r') as joint_file:
        joint_positions = joint_file['positions'][()]
        joint_timestamps = joint_file['timestamps'][()]
        # 假设 actions 和 positions 的维度和时间戳是一致的
        # 这里我们直接复制 positions 作为 actions，如果你的 actions 有不同数据来源，需要调整
        joint_actions = joint_file['positions'][()] # 你的需求中 state 和 action 格式相同

    # 1. 找到第一个对齐的时间戳
    # 找到相机和机器人时间戳中都存在的最小时间戳作为起始点
    # 为了避免浮点数比较问题，我们可以找到最接近的起始点
    first_cam_timestamp = cam_timestamps[0]

    # 找到机器人时间戳中第一个大于或等于相机第一个时间戳的索引
    # start_joint_idx = np.searchsorted(joint_timestamps, first_cam_timestamp, side='left')
    start_joint_idx = np.argmin(np.abs(joint_timestamps - first_cam_timestamp))
    
    # 确保找到的索引没有超出机器人数据范围
    if start_joint_idx >= len(joint_timestamps):
        raise ValueError("相机数据的起始时间戳晚于机器人数据的结束时间戳，无法对齐。")

    # 裁剪机器人数据，从与相机起始时间戳最近的那个时间戳开始
    # 我们需要找到 joint_timestamps 中与 cropped_cam_timestamps[0] 最接近的那个索引
    # 假设我们已经通过 start_joint_idx 找到了一个不错的起点
    cropped_joint_positions = joint_positions[start_joint_idx:]
    cropped_joint_actions = joint_actions[start_joint_idx:]
    cropped_joint_timestamps = joint_timestamps[start_joint_idx:]

    # 2. 根据相机帧率和机器人帧率进行数据匹配
    # 相机频率30fps，机器人频率60Hz，即每帧图像对应两帧机器人数据
    # 定义采样频率
    cam_fps = 30.0
    robot_hz = 60.0

    merged_images = []
    merged_states = []
    merged_actions = []

    # 遍历裁剪后的相机数据
    for i in range(len(cam_timestamps)):
        cam_time = cam_timestamps[i]
        
        # 找到机器人时间戳中第一个大于或等于当前相机时间戳的索引
        # np.searchsorted 会找到插入点，所以这个索引对应的就是第一个大于或等于 cam_time 的机器人时间戳
        # joint_start_idx_for_cam_frame = np.searchsorted(cropped_joint_timestamps, cam_time, side='left')
        joint_start_idx_for_cam_frame = np.argmin(np.abs(cropped_joint_timestamps - cam_time))
        
        # 检查是否还有足够的机器人数据可以取两帧
        if joint_start_idx_for_cam_frame + 1 < len(cropped_joint_timestamps):
            # 取两帧机器人数据
            merged_images.append(cam_rgb_images[i])
            merged_states.append(cropped_joint_positions[joint_start_idx_for_cam_frame])
            merged_states.append(cropped_joint_positions[joint_start_idx_for_cam_frame + 1])
            merged_actions.append(cropped_joint_actions[joint_start_idx_for_cam_frame])
            merged_actions.append(cropped_joint_actions[joint_start_idx_for_cam_frame + 1])
        else:
            # 如果机器人数据不够了，停止合并
            print(f"机器人数据不足，在相机帧 {i} 停止合并。")
            break

    # 转换为 NumPy 数组
    final_images = np.array(merged_images, dtype=np.uint8)
    final_states = np.array(merged_states, dtype=np.float32)
    final_actions = np.array(merged_actions, dtype=np.float32)
    
    print(f"合并完成。相机图像数量: {len(final_images)}, 机器人状态/动作数量: {len(final_states)}")

    # 3. 保存为目标文件格式
    print(f"正在将合并后的数据保存到 {output_path}...")
    with h5py.File(output_path, 'w') as f:
        observations_group = f.create_group('observations')
        images_group = observations_group.create_group('images')
        images_group.create_dataset('left', data=final_images, compression="gzip")
        observations_group.create_dataset('state', data=final_states, compression="gzip")
        f.create_dataset('actions', data=final_actions, compression="gzip")

    print("数据保存成功！")



if __name__ == '__main__':
    cam_file = './extracted_data/demonstration_5/cam_0_rgb_video.h5'
    robot_file = 'extracted_data/demonstration_5/piper_joint_states.h5'
    output_path = 'extracted_data/demonstration_5/merged_data_zip.h5'

    synchronize_and_merge_data(cam_file, robot_file, output_path)