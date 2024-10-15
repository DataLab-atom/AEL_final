import multiprocessing
import os
import subprocess

# 定义要运行的 Python 命令
commands = []
d = "/home/xtanghao/THPycharm/AEL_main/examples_th_new/unsupervised_TU_COLLAB/ael_results/history"
for name in os.listdir(d):
    p = os.path.join(d, name)
    commands.append(f"python evaluate.py --DS NCI1 --aug_point_path {p}")

def run_command(command):
    """运行单个命令"""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Command '{command}' finished with output:\n{result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr.decode()}")

if __name__ == "__main__":
    import time
    # 创建一个进程池，限制最大进程数为3
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=4) as pool:
        # 使用进程池并行运行命令
        pool.map(run_command, commands)
        time.sleep(60)  # 防止写入文件冲突