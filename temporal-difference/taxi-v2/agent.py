import numpy as np
from collections import defaultdict


class Agent:

    def get_greedy_action(self, state):
        greedy_actions = np.argwhere(self.Q[state] == np.amax(self.Q[state]))
        greedy_actions = greedy_actions.flatten()
        return np.random.choice(greedy_actions)

    def get_random_action(self, state):
        return np.random.choice(np.arange(self.nA))

    def eps_greedy_policy(self, state):
        return self.get_random_action(state) if np.random.uniform() <= self.eps \
            else self.get_greedy_action(state)

    def get_estimated_return_for_eps_greedy_policy(self, state):
        prob = np.ones(self.nA) * (self.eps / self.nA)
        prob[np.argmax(self.Q[state])] = 1 - self.eps + self.eps / self.nA
        return np.dot(prob, self.Q[state])

    def __init__(self, nA=6, eps=0.001, alpha=0.1, gamma=0.75):
        # 0.001, 0.1, 0.75 -> 9.228 - 9.435
        # 0.002, 0.2, 0.75 -> 9.239 - 9.448
        self.nA = nA
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        return self.eps_greedy_policy(state)

    def step(self, state, action, reward, next_state, done):
        G = self.get_estimated_return_for_eps_greedy_policy(next_state)
        self.Q[state][action] += self.alpha * \
            (reward + self.gamma * G - self.Q[state][action])
