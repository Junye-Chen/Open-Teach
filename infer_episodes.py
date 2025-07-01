import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from pathlib import Path
import cv2
from act.utils import set_seed
from act.policy import ACTPolicy, CNNMLPPolicy
import sys
import time
from openteach.utils.timer import FrequencyTimer

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# 保留真实机器人和相机接口
from openteach.robot.piper import PiperArm
from act.read_camera import GeminiCamera
# from visualize_episodes import save_videos

# 你可以根据实际情况导入真实环境和仿真环境

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(img):
    curr_image = rearrange(img, 'h w c -> c h w')
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_policy(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    # task_name = config['task_name']
    temporal_agg = config['temporal_agg']

    # 加载模型和统计信息
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(f'Loading: {ckpt_path}')
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 加载环境
    real_robot = True
    if real_robot:
        robot = PiperArm()
        camera = GeminiCamera()

    query_frequency = policy_config.get('num_queries', 1)
    # query_frequency = 120
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)
    num_rollouts = config.get('num_rollouts', 10)
    
    print(f'\nEvaluating ACT policy  --------------')
    infer_timer = FrequencyTimer(20) 
    for rollout_id in range(num_rollouts):
        
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(camera.get_image())
            plt.ion()
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []
        qpos_list = []
        target_qpos_list = []
 
        with torch.inference_mode():
            for t in range(max_timesteps):
                infer_timer.start_loop()
                start_time = time.time()
                if onscreen_render:
                    image = camera.get_image()
                    # plt_img.set_data(image)
                    # plt.pause(0.01)
                    cv2.imshow('Robot camera', image)
                    cv2.waitKey(1)
                    
                image_list.append(camera.get_image())
                qpos_numpy = np.array(robot.get_joint_position())
                print('qpos_numpy', qpos_numpy)
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(camera.get_image()).unsqueeze(0)

                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # print('all_actions', all_actions.shape)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]

                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # print('raw_action', raw_action)
                action = post_process(raw_action)
                target_qpos = action
                print('t_qpos', target_qpos)
                
                # 执行动作
                # robot.move(target_qpos[:6])  # 关节角度
                gripper_offset = 1000*2
                robot.set_gripper(target_qpos[6] - gripper_offset)  # 夹爪状态

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                infer_timer.end_loop()
                # print('freq:', 1/(time.time() - start_time))

            plt.close()
            
        # 可选：保存视频
        save_episode = True
        if save_episode:
            # 用cv2保存images_list
            images_list = np.array(image_list)
            print('images_list', images_list.shape)
            # images_list = np.transpose(images_list, (0, 3, 1, 2))
            images_list = (images_list).astype(np.uint8)
            if not os.path.exists(os.path.join('inference', 'videos', config['taskid'])):
                os.makedirs(os.path.join('inference', 'videos', config['taskid']), exist_ok=True)
            video_path = os.path.join('inference', 'videos', config['taskid'], f'episode_{rollout_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            size = (images_list.shape[2], images_list.shape[1])
            writer = cv2.VideoWriter(video_path, fourcc, fps, size)
            for image in images_list:
                writer.write(image)
            writer.release()
            print(f'Saved video: {video_path}')

    cv2.destroyAllWindows()
    # 保存结果
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join('inference', 'actions', result_file_name), 'w') as f:
        for qpos in target_qpos_list:
            np.savetxt(f, [qpos], fmt='%.4f', delimiter=', ', newline='\n')
            # f.write('\n')

    # print('qpos_history', qpos_history)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='experiments/logs/banana/banana-0', help='模型权重目录')
    parser.add_argument('--ckpt_name', type=str, default='policy_last.ckpt', help='模型权重文件名')
    parser.add_argument('--policy_class', type=str, default='ACT', help='策略类别')
    # parser.add_argument('--task_name', type=str, required=True, help='任务名')
    parser.add_argument('--episode_len', type=int, default=100, help='每个episode步数')
    parser.add_argument('--state_dim', type=int, default=7, help='状态维度')
    parser.add_argument('--action_dim', type=int, default=7, help='动作维度')
    parser.add_argument('--num_rollouts', type=int, default=1, help='推理回合数')
    parser.add_argument('--real_robot', action='store_true', help='是否为真实机器人')
    parser.add_argument('--onscreen_render', action='store_true', help='是否渲染')
    parser.add_argument('--temporal_agg', action='store_true', help='是否时间聚合')
    parser.add_argument('--camera_names', type=str, default='l', help='相机名称')
    # policy config相关参数可根据需要补充
    parser.add_argument('--num_queries', type=int, default=60)
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--qpos_noise_std', action='store', default=0, type=float, help='lr', required=False)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--backbone', type=str, default='dino_v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--taskid', type=str, default='banana')
    parser.add_argument('--exptid', type=str, default='banana-0')
    parser.add_argument('--num_epochs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 构建 policy_config
    if args.policy_class == 'ACT':
        policy_config = {
            'num_queries': args.num_queries,
            'kl_weight': args.kl_weight,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'backbone': args.backbone,
            'enc_layers': 4,
            'dec_layers': 7,
            'seed': args.seed,
            'nheads': 8,
            'state_dim': args.state_dim,
            'action_dim': args.action_dim,
            'num_epochs':args.num_epochs,
            'taskid': args.taskid,
            'exptid': args.exptid,
            'qpos_noise_std': args.qpos_noise_std,
            'camera_names': args.camera_names,
        }
    elif args.policy_class == 'CNNMLP':
        policy_config = {
            'backbone': args.backbone,
            'num_queries': 1,
        }
    else:
        raise NotImplementedError
    
    config = {
        'ckpt_dir': args.ckpt_dir,
        'state_dim': args.state_dim,
        'action_dim': args.action_dim,
        'real_robot': args.real_robot,
        'policy_class': args.policy_class,
        'onscreen_render': args.onscreen_render,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'num_rollouts': args.num_rollouts,
        'episode_len': args.episode_len,
        'taskid': args.taskid,
        'exptid': args.exptid,
    }

    eval_policy(config, args.ckpt_name)

if __name__ == '__main__':
    main() 

    # import time

    # robot = PiperArm()
    # position = [0, 70000, -44000, 0, 44000, 0, 80000]
    # robot.move(position[:6])
    # time.sleep(0.1)
    # position = [-1949, 71571, -46483, 1165, 44134, -1992, 85000]
    # robot.move(position[:6])



"""
python infer_episodes.py --policy_class ACT --seed 0 --taskid banana --exptid banana-0 --num_epochs 1 --onscreen_render --ckpt_name 

python infer_episodes.py --policy_class ACT --seed 0 --taskid banana_crop --exptid banana-1 --num_epochs 1 --onscreen_render

"""