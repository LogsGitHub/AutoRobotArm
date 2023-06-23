import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001  # 定义一个常量，表示actor的学习率
LR_C = 0.001  # 定义一个常量，表示critic的学习率
GAMMA = 0.9  # 定义一个常量，表示奖励的折扣因子
TAU = 0.01  # 定义一个常量，表示软更新的系数
MEMORY_CAPACITY = 30000  # 定义一个常量，表示记忆库的容量
BATCH_SIZE = 32  # 定义一个常量，表示每次训练的批次大小


class Actor(nn.Module):  # 定义一个Actor类，继承自nn.Module类，用来构建actor网络
    def __init__(self, s_dim, a_dim, a_bound):  # 初始化方法，传入状态维度，动作维度和动作范围
        super(Actor, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(s_dim, 300)  # 定义一个全连接层，输入为状态维度，输出为300
        self.fc2 = nn.Linear(300, a_dim)  # 定义一个全连接层，输入为300，输出为动作维度
        self.a_bound = a_bound  # 将动作范围赋值给实例属性

    def forward(self, x):  # 前向传播方法，传入状态x
        x = torch.relu(self.fc1(x))  # 将x通过第一个全连接层，并使用relu激活函数
        x = torch.tanh(self.fc2(x))  # 将x通过第二个全连接层，并使用tanh激活函数
        return x * self.a_bound  # 返回x乘以动作范围，作为输出动作


class Critic(nn.Module):  # 定义一个Critic类，继承自nn.Module类，用来构建critic网络
    def __init__(self, s_dim, a_dim):  # 初始化方法，传入状态维度和动作维度
        super(Critic, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(s_dim + a_dim, 300)  # 定义一个全连接层，输入为状态维度加动作维度，输出为300
        self.fc2 = nn.Linear(300, 1)  # 定义一个全连接层，输入为300，输出为1

    def forward(self, s, a):  # 前向传播方法，传入状态s和动作a
        x = torch.cat([s, a], 1)  # 将s和a拼接在一起，作为输入x
        x = torch.relu(self.fc1(x))  # 将x通过第一个全连接层，并使用relu激活函数
        return self.fc2(x)  # 将x通过第二个全连接层，并返回输出值


class DDPG(object):  # 定义一个DDPG类，用来实现DDPG算法
    def __init__(self, a_dim, s_dim, a_bound, ):  # 初始化方法，传入动作维度，状态维度和动作范围
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1),
                               dtype=np.float32)  # 定义一个数组，用来存储记忆库中的转移元组（s,a,r,s_）
        self.pointer = 0  # 定义一个指针，用来记录记忆库中存储的数量
        self.memory_full = False  # 定义一个布尔值，用来表示记忆库是否已满

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]  # 将动作维度，状态维度和动作范围赋值给实例属性
        self.actor_eval = Actor(s_dim, a_dim, self.a_bound)  # 创建一个Actor对象，作为评估网络
        self.actor_target = Actor(s_dim, a_dim, self.a_bound)  # 创建一个Actor对象，作为目标网络
        self.critic_eval = Critic(s_dim, a_dim)  # 创建一个Critic对象，作为评估网络
        self.critic_target = Critic(s_dim, a_dim)  # 创建一个Critic对象，作为目标网络

        self.optimizer_actor = optim.Adam(self.actor_eval.parameters(), lr=LR_A)  # 定义一个优化器，用来更新actor评估网络的参数
        self.optimizer_critic = optim.Adam(self.critic_eval.parameters(), lr=LR_C)  # 定义一个优化器，用来更新critic评估网络的参数
        self.loss_func = nn.MSELoss()  # 定义一个损失函数，用来计算critic评估网络的均方误差

    def choose_action(self, s):  # 定义一个选择动作的方法，传入状态s
        s = torch.FloatTensor(s).unsqueeze(0)  # 将s转换为torch张量，并增加一个维度
        return self.actor_eval(s).detach().numpy()[0]  # 将s输入actor评估网络，得到动作，并转换为numpy数组，并去掉多余的维度

    def learn(self):  # 定义一个学习的方法，用来更新网络参数
        # soft target replacement
        for target_param, eval_param in zip(self.actor_target.parameters(),
                                            self.actor_eval.parameters()):  # 遍历actor目标网络和评估网络的参数
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)  # 用软更新的方式更新目标网络的参数

        for target_param, eval_param in zip(self.critic_target.parameters(),
                                            self.critic_eval.parameters()):  # 遍历critic目标网络和评估网络的参数
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)  # 用软更新的方式更新目标网络的参数

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # 从记忆库中随机采样一批索引
        bt = self.memory[indices, :]  # 根据索引取出一批转移元组
        bs = torch.FloatTensor(bt[:, :self.s_dim])  # 取出一批状态，并转换为torch张量
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])  # 取出一批动作，并转换为torch张量
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])  # 取出一批奖励，并转换为torch张量
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])  # 取出一批下一状态，并转换为torch张量

        # Critic update
        a_ = self.actor_target(bs_)  # 将下一状态输入actor目标网络，得到下一动作
        q_ = self.critic_target(bs_, a_)  # 将下一状态和下一动作输入critic目标网络，得到下一价值
        q_target = br + GAMMA * q_  # 根据奖励和折扣因子计算目标价值
        q_eval = self.critic_eval(bs, ba)  # 将状态和动作输入critic评估网络，得到评估价值
        critic_loss = self.loss_func(q_eval, q_target)  # 根据评估价值和目标价值计算critic评估网络的损失函数
        self.optimizer_critic.zero_grad()  # 清空优化器中的梯度
        critic_loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), 10)  # 增加一个梯度裁剪的操作，限制critic评估网络的梯度范数在10以内
        self.optimizer_critic.step()  # 更新critic评估网络的参数

        # Actor update
        a = self.actor_eval(bs)  # 将状态输入actor评估网络，得到动作
        q = self.critic_eval(bs, a)  # 将状态和动作输入critic评估网络，得到价值
        actor_loss = -torch.mean(q)  # 根据价值计算actor评估网络的损失函数，取负号是为了梯度上升
        self.optimizer_actor.zero_grad()  # 清空优化器中的梯度
        actor_loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), 10)  # 增加一个梯度裁剪的操作，限制actor评估网络的梯度范数在10以内
        self.optimizer_actor.step()  # 更新actor评估网络的参数

    def store_transition(self, s, a, r, s_):  # 定义一个存储转移元组的方法，传入状态，动作，奖励和下一状态
        transition = np.hstack((s, a, [r], s_))  # 将转移元组拼接在一起，作为一个数组
        #r -= np.sum(np.square(a)) * 0.1  # 增加一个动作惩罚项，使得动作越小越好
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition  # 将转移元组存储到记忆库中，根据指针位置替换旧的记忆
        self.pointer += 1  # 指针加一
        if self.pointer > MEMORY_CAPACITY:  # indicator for learning
            self.memory_full = True  # 如果指针超过记忆库容量，表示记忆库已满

    def save(self, path='./params'):  # 定义一个保存网络参数的方法，传入保存路径，默认为'./params'
        torch.save(self.actor_eval.state_dict(), path + '_actor.pth')  # 保存actor评估网络的参数到指定路径
        torch.save(self.critic_eval.state_dict(), path + '_critic.pth')  # 保存critic评估网络的参数到指定路径

    def restore(self, path='./params'):  # 定义一个恢复网络参数的方法，传入恢复路径，默认为'./params'
        self.actor_eval.load_state_dict(torch.load(path + '_actor.pth'))  # 加载actor评估网络的参数从指定路径
        self.critic_eval.load_state_dict(torch.load(path + '_critic.pth'))  # 加载critic评估网络的参数从指定路径
