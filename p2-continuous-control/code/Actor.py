import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    """Architecture for a policy actor network with continuous actions. 
       Given a state s, estimate the action a that will maximize V(s') for the next state s'.
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Doesn't contain convolutional layers."""

    def __init__(self, state_size, action_size, hidden_layers=[64, 128, 64]):
        """Create an instance of an actor network.

        Parameters
        ----------
        state_size : int
            The number of values in the input vector
        action_size : int
            The number of values in the output vector
        hidden_layers : [int]
            Number of neurons in each hidden layer.
        """

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
        """Evaluate a batch of states, estimating their corresponding best action.

        Parameters
        ----------
        state : torch.tensor
            Size has to be [n * num_agents * state_size], where n is the batch size.

        Returns
        -------
            A torch.tensor of size [n * num_agents * action_size].
            Can be used to perform backward() using an optimizer.
        """
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        mu = self.mu_layer(state)
        return mu, self.sigma
