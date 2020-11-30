import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    """Architecture for a Deep Q Network. Given a state, estimate Q values for all actions.
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Doesn't contain convolutional layers."""

    def __init__(self, state_size, action_size, hidden_layers=[64, 128, 64]):
        """Create an instance of a DQN.

        Parameters
        ----------
        state_size : int
            The number of values in the input vector
        action_size : int
            The number of values in the output vector
        hidden_layers : [int]
            Number of neurons in each hidden layer.
        """

        super(DQN, self).__init__()

        # prepare the first hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])

        # prepare the rest of the hidden layers
        A = hidden_layers[:-1]
        B = hidden_layers[1:]
        self.hidden_layers.extend([nn.Linear(a, b) for a, b in zip(A, B)])

        # prepare the output layer
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
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
        # connect layers to each other and put relu activations between them
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        state = self.output_layer(state)
        return state
