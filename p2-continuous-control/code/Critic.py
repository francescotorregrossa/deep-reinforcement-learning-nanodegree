import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):
    """Architecture for a critic network. Given a state, estimate its V value.
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Doesn't contain convolutional layers."""

    def __init__(self, state_size, hidden_layers=[64, 128, 64]):
        """Create an instance of a critic network.

        Parameters
        ----------
        state_size : int
            The number of values in the input vector
        action_size : int
            The number of values in the output vector
        hidden_layers : [int]
            Number of neurons in each hidden layer.
        """

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
        """Evaluate a batch of states, estimating their corresponding V values.

        Parameters
        ----------
        state : torch.tensor
            Size has to be [n * num_agents * state_size], where n is the batch size.

        Returns
        -------
            A torch.tensor of size [n * num_agents * 1].
            Can be used to perform backward() using an optimizer.
        """

        # connect layers to each other and put relu activations between them
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        value = self.value_layer(state)
        return value
