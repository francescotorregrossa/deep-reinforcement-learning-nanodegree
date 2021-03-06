import torch
import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):
    """Architecture for a critic network. Given a state and action, estimate its Q value.
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Doesn't contain convolutional layers."""

    def __init__(self, state_size, action_size, num_agents, hidden_layers=[512, 256]):
        """Create an instance of a critic network.

        Parameters
        ----------
        state_size : int
            The number of values in the input vector
        action_size : int
            The number of values in the output vector
        num_agents : int
            The number of agents in the environment
        hidden_layers : [int]
            Number of neurons in each hidden layer.
        """

        super(Critic, self).__init__()

        # prepare the first hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size * num_agents, hidden_layers[0])])

        # this makes room to concatenate the given action in the second layer
        edited_hidden_layers = [hl for hl in hidden_layers]
        edited_hidden_layers[0] = hidden_layers[0] + action_size * num_agents

        # prepare the rest of the hidden layers
        A = edited_hidden_layers[:-1]
        B = edited_hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        # the critic will output an estimate of Q(s, a)
        self.output_layer = nn.Linear(edited_hidden_layers[-1], 1)

    def forward(self, state, action):
        """Evaluate a batch of states and actions, estimating their corresponding Q values.

        Parameters
        ----------
        state : torch.tensor
            Size has to be [n * num_agents * state_size], where n is the batch size.
        action : torch.tensor
            Size has to be [n * num_agents * state_size], where n is the batch size.

        Returns
        -------
            A torch.tensor of size [n * num_agents * action_size]. For each sample in
            the batch, estimate its the state-action value.
            Can be used to perform backward() using an optimizer.
        """

        # the input to this network is managed in two steps:
        # - first step only considers the states
        state = self.hidden_layers[0](state)
        state = F.relu(state)

        # - second step adds the actions by concatenating
        #   them with the first hidden layer's output
        state = torch.cat((state, action), dim=2)

        for layer in self.hidden_layers[1:]:
            state = layer(state)
            state = F.relu(state)
        return self.output_layer(state)
