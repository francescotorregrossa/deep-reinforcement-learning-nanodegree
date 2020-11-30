import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64]):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
                
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])
        
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
        

    def forward(self, state):
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        state = self.output_layer(state)
        return state
