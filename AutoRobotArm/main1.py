# 导入env和rl模块，分别定义了环境类和算法类
from env1 import ArmEnv
from rlp import DDPG
import numpy as np
import seaborn as sns  # 导入seaborn库
import matplotlib.pyplot as plt  # 导入matplotlib库
import time

rewards = []  # 定义一个空列表，用来存储每个回合的累积奖励

MAX_EPISODES = 2000  # 定义一个常量，表示最大训练回合数
MAX_EP_STEPS = 300  # 定义一个常量，表示每个回合的最大步数
ON_TRAIN = True  # 定义一个布尔值，表示是否进行训练
import seaborn as sns  # 导入seaborn库
import matplotlib.pyplot as plt  # 导入matplotlib库

rewards = []  # 定义一个空列表，用来存储每个回合的累积奖励
# set env
env = ArmEnv()  # 创建一个ArmEnv对象，作为环境
s_dim = env.state_dim  # 取出环境的状态维度
a_dim = env.action_dim  # 取出环境的动作维度
a_bound = env.action_bound  # 取出环境的动作范围

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)  # 创建一个DDPG对象，作为算法

steps = []  # 定义一个空列表，用来存储步数

n_training_steps = MAX_EPISODES * MAX_EP_STEPS
n_steps_to_reduce_sigma = n_training_steps // 10
initial_sigma = 0.2
final_sigma = 0.01


# ...

def train():  # 定义一个训练的函数
    global_step = 0  # 添加一个全局步数计数器

    # start training
    for i in range(MAX_EPISODES):  # 遍历每个回合
        s = env.reset()  # 重置环境，得到初始状态
        ep_r = 0.  # 定义一个变量，用来记录累积奖励

        for j in range(MAX_EP_STEPS):  # 遍历每个步骤
            # env.render()

            a = rl.choose_action(s)  # 根据状态选择动作

            s_, r, done = env.step(a)  # 根据动作执行环境，得到下一状态，奖励和结束标志

            rl.store_transition(s, a, r, s_)  # 将转移元组存储到记忆库中

            ep_r += r  # 累积奖励加上当前奖励
            if rl.memory_full:  # 如果记忆库已满
                # start to learn once has fulfilled the memory
                rl.learn()  # 开始学习更新网络参数

            s = s_  # 将下一状态赋值给当前状态

            # 根据需要逐步减小噪声
            if global_step % n_steps_to_reduce_sigma == 0:
                current_sigma = initial_sigma - (global_step / n_training_steps) * (initial_sigma - final_sigma)
                rl.noise.set_sigma(current_sigma)

            global_step += 1  # 更新全局步数计数器

            if done or j == MAX_EP_STEPS - 1:  # 如果结束或达到最大步数
                if done: 
                    ep_r = 50
               
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (
                    i, '---' if not done else 'done', ep_r, j))  # 打印回合数，结束标志，累积奖励和步数
                break  # 跳出循环
        rewards.append(ep_r)  # 将每个回合的累积奖励添加到列表中
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间：{elapsed_time:.2f}秒")
    rl.save()
    episodes = range(1, MAX_EPISODES + 1)  # 定义一个列表，表示回合数
    sns.relplot(x=episodes, y=rewards, kind="line", errorbar="sd")
    plt.xlabel("Episode")  # 设置x轴标签为Episode
    plt.ylabel("Accumulated Reward")  # 设置y轴标签为Accumulated Reward
    plt.title("Learning Curve")  # 设置标题为Learning Curve
    plt.ylim(-400, 400)
    xticks = np.arange(0, MAX_EPISODES + 1, 100)  # 设置从0开始，间隔为100的刻度位置
    plt.xticks(xticks)  #
    plt.show()  # 显示图像

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def eval():  # 定义一个评估的函数
    rl.restore()  # 恢复网络参数
    env.render()  # 显示环境
    env.viewer.set_vsync(True)  # 设置垂直同步为真
    s = env.reset()  # 重置环境，得到初始状态
    while True:  # 循环执行以下操作
        env.render()  # 显示环境
        a = rl.choose_action(s)  # 根据状态选择动作
        s, r, done = env.step(a)  # 根据动作执行环境，得到下一状态，奖励和结束标志


if ON_TRAIN:  # 如果进行训练
    start_time = time.time()
    train()  # 调用训练函数

else:  # 否则
    eval()  # 调用评估函数
