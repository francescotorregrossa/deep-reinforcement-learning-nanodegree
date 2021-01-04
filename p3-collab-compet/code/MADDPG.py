import torch
import torch.optim as optim
import numpy as np

from .OUNoise import OUNoise
from .Actor import Actor
from .Critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:

    # -- initialization -- #

    def __init__(self, state_size, action_size, num_agents, replay_buffer,
                 update_every=1, batch_size=1024, alpha=0.0003, gamma=0.99, tau=0.001,
                 eps=3, min_eps=0.5, delta_eps=0.002, learning=True):
        self.state_size, self.action_size, self.num_agents = state_size, action_size, num_agents
        self.update_every, self.batch_size = update_every, batch_size
        self.replay_buffer = replay_buffer
        self.alpha, self.gamma, self.tau = alpha, gamma, tau
        self.original_eps, self.min_eps, self.delta_eps = eps, min_eps, delta_eps
        self.noise = OUNoise(size=(self.num_agents, self.action_size))
        self.learning = learning
        self.reset()

    def reset(self):

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

    # -- initialization -- #

    def act(self, state):
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
        # store a new experience
        self.replay_buffer.add((s, a, r, ns, d))

        if self.update_i == 0 and self.replay_buffer.size() >= self.batch_size:
            # after you've stored enough new experiences, update the networks
            self.learn()

        # keep track of how many new experiences we get
        self.update_i = (self.update_i + 1) % self.update_every

    def new_episode(self):
        # reset and decrease the intensity of the noise in the next episode
        self.noise.reset()
        self.eps = max(self.eps - self.delta_eps, self.min_eps)

    def flip(self, x):
        return torch.cat((x[:, 1, :].unsqueeze(1), x[:, 0, :].unsqueeze(1)), dim=1)

    def partial_detach(self, x):
        return torch.cat((x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1).detach()), dim=1)

    def preprocess(self, x, detach=False):
        x_flip = self.flip(x)
        if detach:
            x = self.partial_detach(x)
            x_flip = self.partial_detach(x_flip)
        return torch.cat((x, x_flip), dim=2)

    def learn(self):
        # note that this is called automatically by the agent

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
