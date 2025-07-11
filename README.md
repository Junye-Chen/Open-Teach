# OPEN TEACH: A Versatile Teleoperation System for Robotic Manipulation

##### Authors: Aadhithya Iyer ,Zhuoran Peng, Yinlong Dai, Irmak Guzey, Siddhant Haldar, Soumith Chintala, Lerrel Pinto 

[Paper](https://arxiv.org/abs/2403.07870) [Website](https://open-teach.github.io/)

## ACT模型训练

### 1. 安装依赖

```
conda create -n tv python=3.10
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
```
### 2. 训练命令

```
# 训练ACT模型
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid banana --exptid banana-0
# 测试ACT模型
python infer_episodes.py --policy_class ACT --seed 0 --taskid banana_crop --exptid banana-1 --num_epochs 1 --onscreen_render
```

### 3. 修改数据集
```
# act/utils.py
self.dataset_paths.append(os.path.join(dataset_dir, f'demo_{episode_id}', 'processed_episode_data.h5'))
```
---

## Quickstart
打开新的终端:
```
# 连接机器人：
cd workspace/piper_sdk/
sudo ethtool -i can0 | grep bus
bash can_activate.sh can0 1000000 
# or
bash can_activate.sh can0 1000000 "1-10:1.0"
```
运行PC后端服务:
```
# 运行远程操作
python teleop.py robot=piper
```
录制数据集:
```
# 打开 `data_collect.py`
python data_collect.py robot=piper demo_num=<数据编号>

# 例如
python data_collect.py robot=piper demo_num=0
```
---


This is the official implementation of the Open Teach including unity scripts for the VR application, teleoperation pipeline and demonstration collection pipeline.

Open Teach consists of two parts. 

- [x] Teleoperation using Meta Quest 3 and data collection over a range of robot morphologies and simulation environments.

- [x] Policy training for various dexterous manipulation tasks across different robots and simulations.

### VR Code and User Interface

Read VR specific information, User Interface and APK files [here](/docs/vr.md)

### Server Code Installation 

Install the conda environment from the yaml file in the codebase

**Allegro Sim**

`conda env create -f env_isaac.yml`

**Others**

`conda env create -f environment.yml`

This will install all the dependencies required for the server code.  

After installing all the prerequisites, you can install this pipeline as a package with pip:

`pip install -e . `

You can test if it had installed correctly by running ` import openteach` from the python shell.

### Robot Controller Installation Specific Information

1. For Simulation specific information, follow the instructions [here](/docs/simulation.md).

2. For Robot controller installation, follow the instructions [here](https://github.com/NYU-robot-learning/OpenTeach-Controllers)

### For starting the camera sensors

For starting the camera sensors and streaming them inside the screen in the oculus refer [here](/docs/sensors.md)

### Running the Teleoperation and Data Collection

For information on running the teleoperation and data collection refer [here](/docs/teleop_data_collect.md).


### Policy Learning 

For open-source code of the policies we trained on the robots refer [here](/docs/policy_learning.md) 

### Policy Learning API

For using the API we use for policy learning, use [this](https://github.com/NYU-robot-learning/Open-Teach-API)

### Call for contributions

For adding your own robot and simulation refer [here](/docs/add_your_own_robot.md)

### Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@misc{iyer2024open,
      title={OPEN TEACH: A Versatile Teleoperation System for Robotic Manipulation}, 
      author={Aadhithya Iyer and Zhuoran Peng and Yinlong Dai and Irmak Guzey and Siddhant Haldar and Soumith Chintala and Lerrel Pinto},
      year={2024},
      eprint={2403.07870},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}



