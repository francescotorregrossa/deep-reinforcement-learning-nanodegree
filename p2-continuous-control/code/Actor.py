import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[64, 128, 64]):
        super(Actor, self).__init__()
        self.action_size = action_size

        # prepare the first hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        # prepare the rest of the hidden layers
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        # the actor will output the parameters of a normal distribution,
        self.mu_layer = nn.Linear(hidden_layers[-1], action_size)
        self.sigma = nn.Parameter(torch.ones(1, action_size))

    def forward(self, state):
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        mu = self.mu_layer(state)
        return mu, self.sigma
