import torch.nn.functional as F
from torch import nn
import torch


class DuelingDQN(nn.Module):

    def __init__(self, state_size, action_size, hidden_layers=[64, 128, 64]):
        super(DuelingDQN, self).__init__()

        self.value_hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])
        self.advantage_hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        A = hidden_layers[:-1]  # -> [64, 128] (using the predefined values)
        B = hidden_layers[1:]   # -> [128, 64]
        in_out = zip(A, B)      # -> [(64, 128), (128, 64)]
        self.value_hidden_layers.extend([nn.Linear(a, b) for a, b in in_out])
        self.advantage_hidden_layers.extend(
            [nn.Linear(a, b) for a, b in in_out])

        self.value_output_layer = nn.Linear(hidden_layers[-1], 1)
        self.advantage_output_layer = nn.Linear(hidden_layers[-1], action_size)

        # self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, x):
        V = x
        A = x.clone()
        for value_layer, advantage_layer in zip(self.value_hidden_layers, self.advantage_hidden_layers):
            V, A = value_layer(V), advantage_layer(A)
            V, A = F.relu(V), F.relu(A)
        V, A = self.value_output_layer(V), self.advantage_output_layer(A)
        return V + A - torch.mean(A, dim=1).unsqueeze(1)
