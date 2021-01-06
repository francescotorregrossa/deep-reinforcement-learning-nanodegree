import torch
import torch.optim as optim
from torch import nn
import numpy as np

from .Actor import Actor
from .Critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent:

    # -- initialization -- #

    def __init__(self, state_size, action_size, calc_advantages, n=4,
                 alpha=0.0001, gamma=0.95, tau=0.5):

        self.state_size, self.action_size = state_size, action_size
        self.alpha, self.gamma, self.tau = alpha, gamma, tau
        self.calc_advantages = calc_advantages
        self.n = n
        self.reset()

    def reset_temporary_buffer(self):
        self.tmp_r, self.tmp_ns, self.tmp_d = [], [], []
        self.tmp_log_prob, self.tmp_critic_out = [], []

    def reset(self):
        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.alpha)

        self.critic = Critic(self.state_size).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.alpha)

        self.reset_temporary_buffer()
        self.i = 0

    # -- initialization -- #

    def act(self, state):
        state = torch.FloatTensor(state).to(device)

        mu, sigma = self.actor(state)
        value = self.critic(state)

        dist = torch.distributions.Normal(mu, sigma)
        # note clamp happens later
        action = dist.sample().detach()
        log_prob = dist.log_prob(action)

        self.tmp_log_prob.append(log_prob.unsqueeze(0))
        self.tmp_critic_out.append(value.unsqueeze(0))

        return np.clip(np.array(action), -1, 1)

    def store(self, s, a, r, ns, d):
        # ignore s and a because they're already handled in act(state)
        self.tmp_r.append(torch.FloatTensor(r).unsqueeze(1).to(device))
        self.tmp_ns.append(torch.FloatTensor(ns).to(device))
        self.tmp_d.append(torch.FloatTensor(d).unsqueeze(1).to(device))

        if self.i == self.n - 1:
            self.learn()

        self.i = (self.i + 1) % self.n

    def learn(self):

        log_prob = torch.cat(self.tmp_log_prob, dim=0)
        critic_out = torch.cat(self.tmp_critic_out, dim=0)

        future = self.critic(self.tmp_ns[-1])
        tmp_returns = []
        ret = future
        for reward, done in zip(reversed(self.tmp_r), reversed(self.tmp_d)):
            ret = reward + self.gamma * (1 - done) * ret
            tmp_returns.insert(0, ret.unsqueeze(0))
        returns = torch.cat(tmp_returns, dim=0).to(device)

        # calc advantages
        advantages = self.calc_advantages(
            self.tmp_r, self.tmp_d, future, returns, critic_out, self.gamma, self.tau)
        # advantages = returns - critic_out

        policy_loss = -torch.mean(log_prob * advantages.detach())
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optimizer.step()

        value_loss = torch.mean((returns.detach() - critic_out) ** 2)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

        self.reset_temporary_buffer()
