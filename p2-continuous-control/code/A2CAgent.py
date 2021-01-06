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
        # store some values in temporary buffers to avoid
        # re-evaluating things and speed up the execution
        self.tmp_r, self.tmp_ns, self.tmp_d = [], None, []
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

        # evaluate the current state and choose
        #  - its value
        value = self.critic(state)

        #  - an action from its probability distribution
        mu, sigma = self.actor(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().detach()

        # then also evaluate the log probability of choosing that action.
        # this is what we'll use for backpropagation, together with the estimated advantage
        log_prob = dist.log_prob(action)

        # store these values in a temporary buffer so that we don't need to calculate them again.
        # this is like a replay buffer in DQN, but it gets wiped after an update step.
        self.tmp_log_prob.append(log_prob.unsqueeze(0))
        self.tmp_critic_out.append(value.unsqueeze(0))

        # note: clipping the action to [-1, 1] is required by the environment, but in my tests,
        # clipping it before calculating its log probability resulted in poor performance
        return np.clip(np.array(action), -1, 1)

    def store(self, s, a, r, ns, d):
        # ignore s and a because they're already handled in act(state)
        self.tmp_r.append(torch.FloatTensor(r).unsqueeze(1).to(device))
        self.tmp_ns = torch.FloatTensor(ns).to(device)
        self.tmp_d.append(torch.FloatTensor(d).unsqueeze(1).to(device))

        # every n steps we should perform an update step
        if self.i == self.n - 1:
            self.learn()
        self.i = (self.i + 1) % self.n

    def learn(self):
        # note that this is called automatically by the agent

        # create a single tensor for the values stored in act(), namely tmp_log_prob and tmp_critic_out
        log_prob = torch.cat(self.tmp_log_prob, dim=0)
        critic_out = torch.cat(self.tmp_critic_out, dim=0)

        # calculate discounted rewards backwards, starting from G or future (the bootstrapping step)
        future = self.critic(self.tmp_ns)
        tmp_returns = []
        ret = future
        for reward, done in zip(reversed(self.tmp_r), reversed(self.tmp_d)):
            ret = reward + self.gamma * (1 - done) * ret
            tmp_returns.insert(0, ret.unsqueeze(0))
        returns = torch.cat(tmp_returns, dim=0).to(device)

        # use either n_step or gae to estimate the advantages.
        # note that not all of these parameters are guaranteed to be used by the given function.
        advantages = self.calc_advantages(
            self.tmp_r, self.tmp_d, future, returns, critic_out, self.gamma, self.tau)

        # train the actor by guiding the increase or decrease of log_prob with the advantages.
        # note the negative sign: pytorch executes gradient descent, so we want to invert that to gradient ascent.
        policy_loss = -torch.mean(log_prob * advantages.detach())
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # the use of gradient clipping helped in my tests
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optimizer.step()

        # train the critic on the mean squared error between its output and the discounted returns
        value_loss = torch.mean((returns.detach() - critic_out) ** 2)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # the use of gradient clipping helped in my tests
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

        # finally empty the temporary buffer
        self.reset_temporary_buffer()
