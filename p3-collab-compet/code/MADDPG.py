import torch
import torch.optim as optim
import numpy as np

from .OUNoise import OUNoise
from .Actor import Actor
from .Critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """Manage a pair of DDPG agents. Keeps four networks, actor (policy), critic (value), and the corresponding
       pair of target_actor and target_critic (for estimation of TD target).
       Learning is performed automatically after a certain number of calls to the store function."""

    def __init__(self, state_size, action_size, num_agents, replay_buffer,
                 update_every=1, batch_size=1024, alpha=0.0003, gamma=0.99, tau=0.001,
                 eps=3, min_eps=0.5, delta_eps=0.002, learning=True):
        """Create a new MADDPG instance with random values.

        Parameters
        ----------
            state_size : int
                Number of parameters in the state.
            action_size : int
                Number of actions available.
            num_agents : int
                Number of agents in the environment.
            replay_buffer : UniformReplayBuffer
                Instance of UniformReplayBuffer. Will be used to store and sample experiences.
            update_every : int
                Indicates how many experiences to store before performing a learning step.
                Note that new experiences aren't guaranteed to be sampled.
            batch_size : int
                Number of tuples to be sampled in a single learning step.
            alpha : float
                Learning rate of the optimizer used for actor and critic.
            gamma : float
                Weight of the estimation of TD target. The value needs to be in [0, 1]
                (usually closer to 1) where 1 means that future rewards are as important as the immediate 
                reward and 0 means that future rewards are ignored.
            tau : float
                Parameters of local networks are copied into target networks after a learning step. tau allows for 'soft'
                updates, where instead of a direct copy of the local parameters, we interpolate towards them, according to
                target = (1 - tau) target + tau (local). The value needs to be in [0, 1] (usually closer to 0),
                where 1 means that we directly copy the parameters and 0 means that the target network stays constant.
            eps : float
                Initial value for the amount of noise in the action.
            delta_eps : float
                Used to decrease eps at the end of every episode.
            min_eps : float
                eps will not go below this value during training. Updates are performed according to
                eps = max(eps - delta_eps, min_eps).            
            learning : bool
                Determines if actions are performed with noise (True) or without it (False).
        """

        self.state_size, self.action_size, self.num_agents = state_size, action_size, num_agents
        self.update_every, self.batch_size = update_every, batch_size
        self.replay_buffer = replay_buffer
        self.alpha, self.gamma, self.tau = alpha, gamma, tau
        self.original_eps, self.min_eps, self.delta_eps = eps, min_eps, delta_eps
        self.noise = OUNoise(size=(self.num_agents, self.action_size))
        self.learning = learning
        self.reset()

    def reset(self):
        """Delete experiences, recreate all networks and reset eps to its original value."""

        self.actor, self.target_actor = Actor(self.state_size, self.action_size).to(device), \
            Actor(self.state_size, self.action_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.alpha)
        with torch.no_grad():
            for local, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.copy_(local)

        self.critic, self.target_critic = Critic(self.state_size, self.action_size, self.num_agents).to(device), \
            Critic(self.state_size, self.action_size,
                   self.num_agents).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.alpha)
        with torch.no_grad():
            for local, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.copy_(local)

        self.replay_buffer.reset()
        self.noise.reset()
        self.update_i = 0
        self.eps = self.original_eps

    def act(self, state):
        """Choose an action according to eps, learning, and actor.

        Parameters
        ----------
        state : torch.tensor
            Tensor of size [state_size]

        Returns
        -------
            Numpy array of size [action_size] whose values are in [-1, 1],
            representing the action that is likely to maximize cumulative rewards.
        """

        state = torch.FloatTensor(state).to(device)

        # predict the action a that will maximize Q(s, a)
        action = self.actor(state)
        action = np.array(action.detach())

        # if the agent is learning, we also add some noise
        # to the network's output to favor exploration
        if self.learning:
            action += self.eps * self.noise.sample()

        # the environment only allows actions to be in the range of [-1, 1]
        return np.clip(action, -1, 1)

    def store(self, s, a, r, ns, d):
        """Add a new experience to the replay buffer. Automatically calls learn() when needed.

        Parameters
        ----------
        s : torch.tensor
            Initial state of the environment, per agent
        a : numpy array
            Action taken by each agent
        r : numpy array
            Reward obtained for taking that action, per agent
        ns : torch.tensor
            State of the environment after taking that action, per agent
        d : bool
            Whether or not the episode is over, per agent
        """

        # store a new experience
        self.replay_buffer.add((s, a, r, ns, d))

        if self.update_i == 0 and self.replay_buffer.size() >= self.batch_size:
            # after you've stored enough new experiences, update the networks
            self.learn()

        # keep track of how many new experiences we get
        self.update_i = (self.update_i + 1) % self.update_every

    def new_episode(self):
        """Reset the noise and decrease its intensity for the next episode."""
        self.noise.reset()
        self.eps = max(self.eps - self.delta_eps, self.min_eps)

    def flip(self, x):
        """Takes a vector x of shape [:, 2, :] and flips it along the second dimension."""
        return torch.cat((x[:, 1, :].unsqueeze(1), x[:, 0, :].unsqueeze(1)), dim=1)

    def partial_detach(self, x):
        """Takes a vector x of shape [:, 2, :] and detaches gradients at the index [:, 1, :]."""
        return torch.cat((x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1).detach()), dim=1)

    def preprocess(self, x, detach=False):
        """Takes a vector x of shape [:, 2, j] and uses flip to obtain another vector of
           size [:, 2, j]. Then, it concatenets them to a final shape of [:, 2, j * 2]. 
           If requested, some of the elements are detached. 
           Read section 5.2 in the report for a more detailed explaination."""
        x_flip = self.flip(x)
        if detach:
            x = self.partial_detach(x)
            x_flip = self.partial_detach(x_flip)
        return torch.cat((x, x_flip), dim=2)

    def learn(self):
        """Sample a batch of experiences (based on batch_size), estimate the TD error
        and take a gradient step to minimize it in critic. Then use the new critic to
        compute the gradients for the actor. Finally perform the soft update of target networks.
        Note that this is called automatically."""

        # sample tuples of experiences from memory (each of these variables is a torch tensor)
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)

        # Section 1: we use the target actor and target critic to estimate
        #  - the predicted best action na in the next state ns
        na = self.target_actor(ns)

        na_final = self.preprocess(na)
        ns_final = self.preprocess(ns)

        #  - the predicted value of Q(ns, na)
        target_values = self.target_critic(ns_final, na_final)
        #  - and finally the TD target
        targets = r.unsqueeze(
            2) + (self.gamma * target_values * (1 - d.unsqueeze(2)))

        s_final = self.preprocess(s)
        a_final = self.preprocess(a)

        # We also compute Q(s, a), but using the local critic this time
        # Note that we're not using the local actor here, since a comes from the replay buffer
        expected_values = self.critic(s_final, a_final)

        # Having the TD target and Q(s, a), we can now calculate the mean squared error
        critic_loss = torch.mean((targets.detach() - expected_values) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Section 2: we use our local actor and local critic to estimate
        #  - the predicted best action expected_a for the current state s
        expected_a = self.actor(s)

        expected_a_final = self.preprocess(expected_a, detach=True)

        #  - the predicted value of Q(s, expected_a)
        policy_values = self.critic(s_final, expected_a_final)

        #  - and, finally, we ask to maximize this last predicted value
        #    note: torch performs gradient descent, so we need a
        #    negative sign to indicate that we want gradient ascent
        actor_loss = -torch.mean(policy_values)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Section 3: we perform a soft update of the two target networks by
        # interpolating between the current parameters and the new ones
        with torch.no_grad():
            for local, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.copy_(target + self.tau * (local - target))
            for local, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.copy_(target + self.tau * (local - target))
