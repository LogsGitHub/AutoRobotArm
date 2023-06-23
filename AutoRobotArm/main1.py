# 导入env和rl模块，分别定义了环境类和算法类
from env import ArmEnv
from rlp import DDPG

MAX_EPISODES = 1200  # 定义一个常量，表示最大训练回合数
MAX_EP_STEPS = 200  # 定义一个常量，表示每个回合的最大步数
ON_TRAIN = False  # 定义一个布尔值，表示是否进行训练

# set env
env = ArmEnv()  # 创建一个ArmEnv对象，作为环境
s_dim = env.state_dim  # 取出环境的状态维度
a_dim = env.action_dim  # 取出环境的动作维度
a_bound = env.action_bound  # 取出环境的动作范围

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)  # 创建一个DDPG对象，作为算法

steps = []  # 定义一个空列表，用来存储步数


def train():  # 定义一个训练的函数
    # start training
    for i in range(MAX_EPISODES):  # 遍历每个回合
        s = env.reset()  # 重置环境，得到初始状态
        ep_r = 0.  # 定义一个变量，用来记录累积奖励
        for j in range(MAX_EP_STEPS):  # 遍历每个步骤
            #env.render()

            a = rl.choose_action(s)  # 根据状态选择动作

            s_, r, done = env.step(a)  # 根据动作执行环境，得到下一状态，奖励和结束标志

            rl.store_transition(s, a, r, s_)  # 将转移元组存储到记忆库中

            ep_r += r  # 累积奖励加上当前奖励
            if rl.memory_full:  # 如果记忆库已满
                # start to learn once has fulfilled the memory
                rl.learn()  # 开始学习更新网络参数

            s = s_  # 将下一状态赋值给当前状态
            if done or j == MAX_EP_STEPS - 1:  # 如果结束或达到最大步数
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (
                i, '---' if not done else 'done', ep_r, j))  # 打印回合数，结束标志，累积奖励和步数
                break  # 跳出循环
    rl.save()  # 保存网络参数


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
    train()  # 调用训练函数

else:  # 否则
    eval()  # 调用评估函数
