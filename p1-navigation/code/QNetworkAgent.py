import numpy as np

import torch
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetworkAgent():
    """DQN agent. Keeps two networks, q_local (policy) and q_target (estimation of TD target).
       Learning is performed automatically after a certain number of calls to store function."""

    def __init__(self, QNetwork, state_size, action_size,
                 replay_buffer, Delta,
                 eps=1, eps_decay=0.9995, min_eps=0.0001,
                 gamma=0.99, alpha=0.001, tau=0.01,
                 update_every=15, batch_size=64, learning=True):
        """Create a new agent initialized with random values.

        Parameters
        ----------
            QNetwork : class
                One of DQN or DuelingDQN. Will be used to create two instances of the given class.
            state_size : int
                Number of parameters in the state.
            action_size : int
                Number of actions available.
            replay_buffer : UniformReplayBuffer
                Instance of UniformReplayBuffer. Will be used to store and sample experiences.
            Delta : function
                One of dt_dqn or dt_double_dqn. Will be used to calculate the error between
                the estimated TD target (by q_target) and the current value (by q_local).
                Will be used to train q_local using backpropagation.
            eps : float
                Initial value for the epsilon-greedy policy. The value needs to be in [0, 1], where 1
                means that every action taken by act() is random and 0 means that every action is greedy.
            eps_decay : float
                Used to decrease eps at every learning step. eps_decay needs to be in [0, 1].
            min_eps : float
                eps will not go below this value during training. Updates are performed according to
                eps = max(eps * eps_decay, min_eps).
            gamma : float
                Weight of the estimation of TD target, used in Delta. The value needs to be in [0, 1]
                (usually closer to 1) where 1 means that future rewards are as important as the immediate 
                reward and 0 means that future rewards are ignored.
            alpha : float
                Learning rate of the optimizer used for q_local. 
                The value needs to be in [0, 1] (usually closer to 0).
            tau : float
                We copy the parameters of q_local in q_target after a learning step. tau allows for 'soft'
                updates, where instead of a direct copy of q_local we interpolate towards it, according to
                q_target = (1 - tau) q_target + tau (q_local). The value needs to be in [0, 1] (usually closer to 0),
                where 1 means that we directly copy the parameters and 0 means that q_target stays constant.
            update_every : int
                Indicates how many experiences to store before performing a learning step.
                Note that new experiences aren't guaranteed to be sampled.
            batch_size : int
                Number of tuples to be sampled in a single learning step.
            learning : bool
                Determines if the policy used is eps-greedy (True) or greedy (False).
        """

        self.state_size, self.action_size = state_size, action_size
        self.original_eps = eps
        self.QNetwork = QNetwork
        self.replay_buffer = replay_buffer
        self.Delta = Delta
        self.learning = learning
        self.eps, self.eps_decay, self.min_eps = eps, eps_decay, min_eps
        self.gamma, self.alpha, self.tau = gamma, alpha, tau
        self.update_every, self.batch_size = update_every, batch_size
        self.reset()

    def reset(self):
        """Delete experiences, recreate q_local and q_target and reset eps to its original value."""

        self.replay_buffer.reset()
        self.eps = self.original_eps
        self.q_local = self.QNetwork(
            self.state_size, self.action_size).to(device)
        self.q_target = self.QNetwork(
            self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.alpha)
        self.update_i = 0

    def act(self, s):
        """Choose an action according to eps, learning, and q_local.

        Parameters
        ----------
        s : torch.tensor
            Tensor of size [state_size]

        Returns
        -------
            int representing the action that is likely to maximize cumulative rewards,
            according to eps-greedy/greedy policy estimated by q_local.
        """

        # eps-greedy policy: decide if the action should be random or greedy
        # (always greedy if agent.learning = False)
        if not self.learning or np.random.uniform() > self.eps:
            # greedy action, no need for autograd in this step. simply estimate
            # Q values for this state and return the action with the highest value
            with torch.no_grad():
                s = torch.FloatTensor(s).unsqueeze(0).to(device)
                return int(self.q_local(s).max(1)[1])
        else:
            # choose uniformly between all actions
            return np.random.randint(self.action_size)

    def store(self, s, a, r, ns, d):
        """Add a new experience to the replay buffer. Automatically calls learn() when needed.

        Parameters
        ----------
        s : torch.tensor
            Initial state of the environment
        a : int
            Action taken by the agent
        r : float
            Reward obtained for taking that action
        ns : torch.tensor
            State of the environment after taking that action
        d : bool
            Whether or not the episode is over
        """

        # store a new experience
        self.replay_buffer.add((s, a, r, ns, d))

        if self.update_i == 0 and self.replay_buffer.size() >= self.batch_size:
            # after you've stored enough new experiences update q_local and q_target
            # (note however that the batch used to learn might not contain the new experiences)
            self.learn()

            # also decrease epsilon
            self.eps = max(self.eps * self.eps_decay, self.min_eps)

        # keep track of how many new experiences we get
        self.update_i = (self.update_i + 1) % self.update_every

    def learn(self):
        """Sample a batch of experiences (based on batch_size), use Delta to estimate the error
        and take a gradient step to minimize it in q_local. Then use q_local to perform a soft 
        update of q_target. Note that this is called automatically by the agent."""

        # sample tuples of experiences from memory (each of these variables is a torch tensor)
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)

        # use the given function to calculate the difference between the TD target and our estimate
        td_delta = self.Delta(s, a, r, ns, d, self.q_local,
                              self.q_target, self.gamma)

        # use autograd to backpropagate, the error is MSE on td_delta
        self.optimizer.zero_grad()
        loss = torch.mean(td_delta ** 2)
        loss.backward()
        self.optimizer.step()

        # after updating q_local we also update q_target. the original paper makes a copy of the parameters,
        # however we can perform a 'soft' update by interpolating between the current parameters and the new ones
        with torch.no_grad():
            for local, target in zip(self.q_local.parameters(), self.q_target.parameters()):
                target.copy_(target + self.tau * (local - target))
