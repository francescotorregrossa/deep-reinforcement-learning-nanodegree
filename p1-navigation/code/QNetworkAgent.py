import numpy as np

import torch
import torch.optim as optim


class QNetworkAgent():

    # -- initialization -- #
    def __init__(self, QNetwork, state_size, action_size,
                 replay_buffer, Delta,
                 eps=1, eps_decay=0.9995, min_eps=0.0001, gamma=0.99,
                 alpha=0.001, tau=0.01,
                 update_every=15, batch_size=64, learning=True):
        self.state_size, self.action_size = state_size, action_size
        self.original_eps = eps
        self.QNetwork = QNetwork
        self.replay_buffer = replay_buffer
        self.Delta = Delta
        self.learning = learning
        self.eps, self.eps_decay, self.min_eps = eps, eps_decay, min_eps
        self.gamma, self.alpha, self.tau = gamma, alpha, tau
        self.update_every, self.batch_size = update_every, batch_size
        self.reset()

    def reset(self):
        self.replay_buffer.reset()
        self.eps = self.original_eps
        self.q_local = self.QNetwork(
            self.state_size, self.action_size).to(device)
        self.q_target = self.QNetwork(
            self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.alpha)
        self.update_i = 0
    # -- initialization -- #

    def act(self, s):

        # eps-greedy policy: decide if the action should be random or greedy
        # (always greedy if agent.learning = False)
        if not self.learning or np.random.uniform() > self.eps:
            # greedy action, no need for autograd in this step. simply estimate
            # Q values for this state and return the action with the highest value
            with torch.no_grad():
                s = torch.FloatTensor(s).unsqueeze(0).to(device)
                return int(self.q_local(s).max(1)[1])
        else:
            # choose uniformly between all actions
            return np.random.randint(self.action_size)

    def store(self, s, a, r, ns, d):
        # store a new experience
        self.replay_buffer.add((s, a, r, ns, d))

        if self.update_i == 0 and self.replay_buffer.size() >= self.batch_size:
            # after you've stored enough new experiences update q_local and q_target
            # (note however that the batch used to learn might not contain the new experiences)
            self.learn()

            # also decrease epsilon
            self.eps = max(self.eps * self.eps_decay, self.min_eps)

        # keep track of how many new experiences we get
        self.update_i = (self.update_i + 1) % self.update_every

    def learn(self):
        # note that this is called automatically by the agent

        # sample tuples of experiences from memory (each of these variables is a torch tensor)
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)

        # use the given function to calculate the difference between the TD target and our estimate
        td_delta = self.Delta(s, a, r, ns, d, self.q_local,
                              self.q_target, self.gamma)

        # use autograd to backpropagate, the error is MSE on td_delta
        self.optimizer.zero_grad()
        loss = torch.mean(td_delta ** 2)
        loss.backward()
        self.optimizer.step()

        # after updating q_local we also update q_target. the original paper makes a copy of the parameters,
        # however we can perform a 'soft' update by interpolating between the current parameters and the new ones
        with torch.no_grad():
            for local, target in zip(self.q_local.parameters(), self.q_target.parameters()):
                target.copy_(target + self.tau * (local - target))
