import numpy as np
from numpy_ringbuffer import RingBuffer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class UniformReplayBuffer():
    """A buffer that stores tuples (s, a, r, ns, d) and allows to sample uniformly between them.
        - s is the state at the beginning of the timestep
        - a is the action that was taken
        - r is the reward obtained in the next timestep
        - ns is the state at the next timestep
        - d is a boolean value that determines if the episode ended
    """

    def __init__(self, capacity):
        """Initialize the replay buffer with a given capacity.

        Parameters
        ----------
        capacity : int
            Maximum capacity of the replay buffer. Old tuples will be deleted if necessary.
        """
        self.capacity = capacity
        self.reset()

    def reset(self):
        """Delete all the contents of the buffer. Capacity is maintained."""
        self.buff = RingBuffer(capacity=self.capacity, dtype=object)

    def sample(self, n, replace=True):
        """Sample a batch of n tuples (s, a, r, ns, d). Parameters are described at the top of the class.

            Parameters
            ----------
            n : int
                The number of observations to sample in this batch.
            replace : bool
                Whether or not a tuple can be extracted more than once in a batch. 
                See numpy.random.choice for more information.

            Returns
            -------
                A tuple of torch tensors ([s], [a], [r], [ns], [d]).
                The number of rows in each tensor is the same and equals n, the size of the batch.
        """
        samples = np.random.choice(np.array(self.buff), n, replace)

        s = torch.FloatTensor([sample[0] for sample in samples]).to(device)
        a = torch.FloatTensor([sample[1] for sample in samples]).to(device)
        r = torch.FloatTensor([sample[2] for sample in samples]).to(device)
        ns = torch.FloatTensor([sample[3] for sample in samples]).to(device)
        d = torch.FloatTensor([sample[4] for sample in samples]).to(device)

        return s, a, r, ns, d

    def add(self, observation):
        """Store a tuple (s, a, r, ns, d). Parameters are described at the top of the class.
           Adding a new tuple could result in the deletion of an old one.

            Parameters
            ----------
            observation : tuple
                Contains the observation to store.
        """
        s, a, r, ns, d = observation
        self.buff.append((s, a, r, ns, d))

    def size(self):
        """The number of elements currently stored."""
        return len(self.buff)
