import numpy as np
from numpy_ringbuffer import RingBuffer
import torch


class UniformReplayBuffer():
    """A buffer that stores tuples (s, a, r, ns, d) and allows to sample uniformly between them.
        - s is the state at the beginning of the timestep
        - a is the action that was taken
        - r is the reward obtained in the next timestep
        - ns is the state at the next timestep
        - d is a boolean value that determines if the episode ended
        There are extra parameters described in add() and sample().
    """

    def __init__(self, size):
        """Initialize the replay buffer with a given capacity.

        Parameters
        ----------
        size : int
            Maximum capacity of the replay buffer. Old tuples will be deleted if necessary.
        """
        self.buff = RingBuffer(capacity=size, dtype=object)

    def sample(self, n, replace=True):
        """Sample a batch of n tuples (s, a, r, ns, d, w) where:
            - s, a, r, ns, d are described at the top of the class.
            - w is the importance sampling weight. In the uniform case, it's just 1/n
              for every tuple. It is kept here because it will make a possible
              transition to the prioritized version easier.

            Parameters
            ----------
            n : int
                The number of observations to sample in this batch.
            replace : bool
                Whether or not a tuple can be extracted more than once in a batch. 
                See numpy.random.choice for more information.

            Returns
            -------
                A tuple of torch tensors ([s], [a], [r], [ns], [d], [w]).
                The number of rows in each tensor is the same and equals n, the size of the batch.
                The number of columns in s and ns depends on the shape of the state.
        """
        samples = np.random.choice(np.array(self.buff), n, replace)

        s = torch.FloatTensor([sample[0] for sample in samples])
        a = torch.LongTensor([sample[1] for sample in samples])
        r = torch.FloatTensor([sample[2] for sample in samples])
        ns = torch.FloatTensor([sample[3] for sample in samples])
        d = torch.FloatTensor([sample[4] for sample in samples])
        w = torch.ones(d.size()) / n

        return s, a, r, ns, d, w

    def add(self, observation):
        """Store a tuple (s, a, r, ns, d, p) where:
            - s, a, r, ns, d are described at the top of the class.
            - p is the priority of the tuple, needed but ignored in this function.
              It is kept because it will make a possible transition to the
              prioritized version easier. At the moment, however, every tuple will
              be considered with the max_priority() = 0 (not stored), regardless of p.
            Adding a new tuple could result in the deletion of an old one.

            Parameters
            ----------
            observation : tuple
                Contains the observation to store.
        """
        s, a, r, ns, d, _ = observation
        self.buff.append((s, a, r, ns, d))

    def size(self):
        """The number of elements currently stored."""
        return len(self.buff)

    def max_priority(self):
        """The priority of each tuple in the uniform case is the same and considered zero.
           This function is kept for easier implementation of the prioritized version.
        """
        return 0

    def update_priority(self, observations):
        """The priority of each tuple in the uniform case is the same and considered zero.
           This function is kept for easier implementation of the prioritized version.

            Parameters
            ----------
            observations : [tuple]
                A list of observations, each in the same shape as the one used in add().
        """
        pass
