import numpy as np
import copy


class OUNoise:
    """Sample noise from a Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
