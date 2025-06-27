# 打开指定目录下的.h5文件
import h5py
import numpy as np  
import os


def crop_data():
    # 定义数据存放路径
    data_path = '/home/eigindustry/workspace/Open-Teach/extracted_data/banana'
    idx = 15
    fold_path = 'demonstration_' + str(idx)
    file_path = 'merged_data_zip.h5'

    # 读取数据
    print(os.path.join(data_path, fold_path, file_path))
    f = h5py.File(os.path.join(data_path, fold_path, file_path), 'r')
    qpos_data = np.array(f['observations']['state'])
    action_data = np.array(f['actions'])
    image_data = np.array(f['observations']['images']['left'])

    # 打印数据信息
    print(f"qpos_data shape: {qpos_data.shape}")
    print(f"action_data shape: {action_data.shape}")
    print(f"image_data shape: {image_data.shape}")

    # 
    start_idx = 37
    qpos_data = qpos_data[start_idx:]
    action_data = action_data[start_idx:]
    image_data = image_data[start_idx:]

    # 打印数据信息
    print(f"qpos_data shape: {qpos_data.shape}")
    print(f"action_data shape: {action_data.shape}")
    print(f"image_data shape: {image_data.shape}")

    # 保存数据
    id = 11
    folder = 'demo_' + str(id)
    save_path = '/home/eigindustry/workspace/Open-Teach/experiments/recordings/banana_crop'
    output_path = os.path.join(save_path, folder, 'processed_episode_data.h5')
    print(f"正在将裁剪后的数据保存到 {output_path}...")
    with h5py.File(output_path, 'w') as f:
        observations_group = f.create_group('observations')
        images_group = observations_group.create_group('images')
        images_group.create_dataset('left', data=image_data, compression="gzip")
        observations_group.create_dataset('state', data=qpos_data, compression="gzip")
        f.create_dataset('actions', data=action_data, compression="gzip")

    # 关闭文件
    f.close()


if __name__ == '__main__':
    import os

    folder = '/home/eigindustry/workspace/Open-Teach/extracted_data/banana'
    ob_dir = '/home/eigindustry/workspace/Open-Teach/experiments/recordings/banana_crop'
    # for id, f in enumerate(sorted(os.listdir(folder))):
    #     print(id, f)
    crop_data()
