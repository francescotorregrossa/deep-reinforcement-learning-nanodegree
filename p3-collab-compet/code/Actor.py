import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    """Architecture for a policy actor network with continuous actions. 
       Given a state, estimate the action a that will maximize Q(s, a).
       Input, output and hidden layers can be customized. ReLU is used between layers.
       Output uses tanh. Doesn't contain convolutional layers."""

    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
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
        """Evaluate a batch of states, estimating their corresponding best action.

        Parameters
        ----------
        state : torch.tensor
            Size has to be [n * num_agents * state_size], where n is the batch size.

        Returns
        -------
            A torch.tensor of size [n * num_agents * action_size]. For each state in
            the batch, estimate the action that will maximize its value.
            Can be used to perform backward() using an optimizer. Output action is
            limited to the range of -1 and 1 by a tanh activation function.
        """
        for layer in self.hidden_layers:
            state = layer(state)
            state = F.relu(state)
        out = self.output_layer(state)
        return F.tanh(out)
