import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def set_sigma(self, sigma):
        self.sigma = sigma


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 300)
        self.fc2 = nn.Linear(300, a_dim)
        self.a_bound = a_bound

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x * self.a_bound


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 300)
        self.fc2 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        self.actor_eval = Actor(s_dim, a_dim, self.a_bound).to(device)
        self.actor_target = Actor(s_dim, a_dim, self.a_bound).to(device)
        self.critic_eval = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor_eval.parameters(), lr=LR_A)  # 定义一个优化器，用来更新actor评估网络的参数
        self.optimizer_critic = optim.Adam(self.critic_eval.parameters(), lr=LR_C)  # 定义一个优化器，用来更新critic评估网络的参数
        #self.optimizer_actor = optim.RMSprop(self.actor_eval.parameters(), lr=LR_A)  # 使用新的优化器
        #self.optimizer_critic = optim.RMSprop(self.critic_eval.parameters(), lr=LR_C)
        self.noise = OrnsteinUhlenbeckNoise(a_dim)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        # 将输入数据转换为张量并移动到指定的设备上
        s = torch.FloatTensor(s).unsqueeze(0).to(device)

        # 获取不带噪声的动作
        action = self.actor_eval(s).detach().cpu().numpy()[0]

        # 为动作添加噪声
        action += self.noise.sample()

        # 将带噪声的动作返回
        return action

    def learn(self):
        # soft target replacement
        for target_param, eval_param in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)

        for target_param, eval_param in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        # 将批数据转换为张量并移动到指定的设备上
        bt = torch.FloatTensor(self.memory[indices]).to(device)

        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Critic update
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + GAMMA * q_
        q_eval = self.critic_eval(bs, ba)

        critic_loss = self.loss_func(q_eval, q_target)

        # 清空梯度缓存
        self.optimizer_critic.zero_grad()

        # 反向传播计算梯度
        critic_loss.backward()

        # 增加一个梯度裁剪的操作，限制critic评估网络的梯度范数在10以内
        # 这里的参数是一个可迭代的对象，可以直接传入网络模型的参数
        torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), 10)

        # 更新网络参数
        self.optimizer_critic.step()

        # Actor update
        a = self.actor_eval(bs)
        q = self.critic_eval(bs, a)
        actor_loss = -torch.mean(q)

        # 清空梯度缓存
        self.optimizer_actor.zero_grad()

        # 反向传播计算梯度
        actor_loss.backward()

        # 增加一个梯度裁剪的操作，限制actor评估网络的梯度范数在10以内
        # 这里的参数是一个可迭代的对象，可以直接传入网络模型的参数
        torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), 10)

        # 更新网络参数
        self.optimizer_actor.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # r -= np.sum(np.square(a)) * 0.1  # 增加一个动作惩罚项，使得动作越小越好
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:  # indicator for learning
            self.memory_full = True

    def save(self, path='./params'):

        # 保存模型参数时，需要指定保存到cpu上，否则在加载时会出错
        torch.save(self.actor_eval.state_dict(), path + '_actor.pth', _use_new_zipfile_serialization=False)
        torch.save(self.critic_eval.state_dict(), path + '_critic.pth', _use_new_zipfile_serialization=False)

    def restore(self, path='./params'):

        # 加载模型参数时，需要指定加载到cpu上，否则在加载时会出错
        self.actor_eval.load_state_dict(torch.load(path + '_actor.pth', map_location='cpu'))
        self.critic_eval.load_state_dict(torch.load(path + '_critic.pth', map_location='cpu'))
