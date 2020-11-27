import numpy as np

import torch
import torch.optim as optim


class QNetworkAgent():

    def __init__(self, QNetwork, state_size, action_size,
                 replay_buffer, Delta,
                 eps=1, eps_decay=0.9995, min_eps=0.0001, gamma=0.99,
                 alpha=0.001, tau=0.01,
                 update_every=4, batch_size=64, learning=True):

        self.state_size = state_size
        self.action_size = action_size

        self.q_local = QNetwork(state_size, action_size)
        self.q_target = QNetwork(state_size, action_size)
        self.q_target.eval()

        self.replay_buffer = replay_buffer

        self.Delta = Delta

        self.optimizer = optim.Adam(self.q_local.parameters(), lr=alpha)

        self.learning = learning

        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        self.update_every = update_every
        self.update_i = 0
        self.batch_size = batch_size

    def act(self, s):
        if not self.learning or np.random.uniform() > self.eps:
            with torch.no_grad():
                s = torch.FloatTensor(s).unsqueeze(0)
                return int(self.q_local(s).max(1)[1])
        else:
            return np.random.randint(self.action_size)

    def store(self, s, a, r, ns, d):
        p = self.replay_buffer.max_priority()
        self.replay_buffer.add((s, a, r, ns, d, p))
        if self.update_i == 0 and self.replay_buffer.size() >= self.batch_size:
            self.learn()
        self.update_i = (self.update_i + 1) % self.update_every
        self.eps = max(self.eps * self.eps_decay, self.min_eps)

    def learn(self):
        s, a, r, ns, d, w = self.replay_buffer.sample(self.batch_size)
        td_delta = self.Delta(s, a, r, ns, d, self.q_local,
                              self.q_target, self.gamma)
        self.replay_buffer.update_priority(
            zip(s, a, r, ns, d, torch.abs(td_delta)))

        self.optimizer.zero_grad()
        loss = torch.sum(w * (td_delta ** 2))  # weighted mse
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for local, target in zip(self.q_local.parameters(), self.q_target.parameters()):
                target.copy_(target + self.tau * (local - target))
