import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[64, 128, 64]):
        super(DQN, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        A = hidden_layers[:-1]  # -> [64, 128]
        B = hidden_layers[1:]  # -> [128, 64]
        # so that zip(A, B) will be [(64, 128), (128, 64)] (using the predefined values)
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        state = self.output_layer(state)
        return state
