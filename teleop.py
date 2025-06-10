import hydra
from openteach.components import TeleOperator

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'teleop')
def main(configs):
    teleop = TeleOperator(configs)
    processes = teleop.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()

"""
连接机器人：
cd workspace/piper_sdk/
bash can_activate.sh can0 1000000



"""

"""
# 启动服务器
python deploy_server.py

# 运行远程操作
python teleop.py robot=piper
"""