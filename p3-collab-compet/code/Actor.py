import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
        super(Actor, self).__init__()

        # prepare the first hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        # prepare the rest of the hidden layers
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        # the actor will output the action a that maximizes Q(s, a)
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        out = self.output_layer(state)
        return F.tanh(out)
