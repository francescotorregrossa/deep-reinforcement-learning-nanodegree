import torch.nn.functional as F
from torch import nn
import torch


class DuelingDQN(nn.Module):
    """Architecture for a Dueling Deep Q Network. Given a state, estimate 
       V and A for all actions, then merge them into a single Q vector.
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Doesn't contain convolutional layers."""

    def __init__(self, state_size, action_size, hidden_layers=[32, 128, 32]):
        """Create an instance of a Dueling DQN.
        The two streams V and A have the same shape.

        Parameters
        ----------
        state_size : int
            The number of values in the input vector
        action_size : int
            The number of values in the output vector
        hidden_layers : [int]
            Number of neurons in each hidden layer.
        """

        super(DuelingDQN, self).__init__()

        # prepare the first pair of hidden layers (will be two independent streams)
        self.value_hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])
        self.advantage_hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        # prepare the rest of the hidden layers
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.value_hidden_layers.extend(
            [nn.Linear(a, b) for a, b in zip(A, B)])
        self.advantage_hidden_layers.extend(
            [nn.Linear(a, b) for a, b in zip(A, B)])

        # prepare the output layers for the streams, which will be merged in the forward operation
        self.value_output_layer = nn.Linear(hidden_layers[-1], 1)
        self.advantage_output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, x):
        """Evaluate a batch of states, estimating their corresponding state-action values.

        Parameters
        ----------
        state : torch.tensor
            Size has to be [n * state_size], where n is the batch size.

        Returns
        -------
            A torch.tensor of size [n * action_size]. For each of the n samples in
            the batch, estimate the state-action value of every possible action.
            Can be used to perform backward() using an optimizer.
        """

        # the two streams receive the same input data x
        V = x
        A = x.clone()

        # as before, connect layers to each other and put relu activations between them
        for value_layer, advantage_layer in zip(self.value_hidden_layers, self.advantage_hidden_layers):
            V, A = value_layer(V), advantage_layer(A)
            V, A = F.relu(V), F.relu(A)
        V, A = self.value_output_layer(V), self.advantage_output_layer(A)

        # the output of the network is the special merging layer described in the dueling dqn paper
        return V + A - torch.mean(A, dim=1).unsqueeze(1)
