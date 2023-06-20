import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


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
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.actor_eval = Actor(s_dim, a_dim, self.a_bound)
        self.actor_target = Actor(s_dim, a_dim, self.a_bound)
        self.critic_eval = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        self.optimizer_actor = optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.optimizer_critic = optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        return self.actor_eval(s).detach().numpy()[0]

    def learn(self):
        # soft target replacement
        for target_param, eval_param in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)

        for target_param, eval_param in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            target_param.data.copy_(TAU * eval_param.data + (1 - TAU) * target_param.data)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        # Critic update
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + GAMMA * q_
        q_eval = self.critic_eval(bs, ba)
        critic_loss = self.loss_func(q_eval, q_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        a = self.actor_eval(bs)
        q = self.critic_eval(bs, a)
        actor_loss = -torch.mean(q)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def save(self, path='./params'):
        torch.save(self.actor_eval.state_dict(), path + '_actor.pth')
        torch.save(self.critic_eval.state_dict(), path + '_critic.pth')

    def restore(self, path='./params'):
        self.actor_eval.load_state_dict(torch.load(path + '_actor.pth'))
        self.critic_eval.load_state_dict(torch.load(path + '_critic.pth'))