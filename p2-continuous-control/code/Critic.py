import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):

    def __init__(self, state_size, hidden_layers=[64, 128, 64]):
        super(Critic, self).__init__()

        # prepare the first hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        # prepare the rest of the hidden layers
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        # the critic outputs only a scalar V(s)
        self.value_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, state):
        # connect layers to each other and put relu activations between them
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        value = self.value_layer(state)
        return value
