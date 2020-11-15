import time
import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy
env = gym.make('Blackjack-v0')

import matplotlib
matplotlib.use('TkAgg')  # it doesn't work on my mac otherwise

def get_greedy_action(Q, state):
    # if there are two or more actions for which Q[s][a] is maximized, choose uniformly between them
    greedy_actions = np.argwhere(Q[state] == np.amax(Q[state]))
    greedy_actions = greedy_actions.flatten()
    return np.random.choice(greedy_actions)


def generate_episode_eps_policy(env, Q, eps):
    nA = env.action_space.n
    episode = []
    state = env.reset()
    while True:

        # with probability eps choose random action, 1-eps greedy action
        action = np.random.choice(np.arange(nA)) if np.random.uniform() <= eps \
            else get_greedy_action(Q, state)

        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def improve_q_from_episode(Q, policy, episode, alpha, gamma=1.0, every_visit=False):

    is_first_visit = defaultdict(lambda: np.full(env.action_space.n, True))

    # walk backwards from the last action to the first
    G_t_1 = 0.0
    partial_returns = []
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]

        # calculate return G_t from this point
        G_t = reward + gamma * G_t_1
        G_t_1 = G_t
        partial_returns.insert(0, G_t)

    for t in range(len(episode)):
        state, action, reward = episode[t]
        G_t = partial_returns[t]

        # check for first-visit for this episode, if requested
        if every_visit or is_first_visit[state][action]:
            is_first_visit[state][action] = False

            # recalculate the average and update the policy
            Q[state][action] += alpha * (G_t - Q[state][action])
            policy[state] = get_greedy_action(Q, state)

    return Q, policy


def mc_control(env, num_episodes, alpha, gamma=1.0, eps=1, final_eps=0.1, stop_eps_after=0.5, every_visit=False):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    policy = defaultdict(lambda: np.choice(np.arange(nA)))

    # eps will decrease linearly and reach final_eps in episode stop_eps_at_episode
    stop_eps_at_episode = num_episodes * stop_eps_after - 1
    eps_delta = (eps - final_eps) / stop_eps_at_episode

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # generate episode with current policy and eps
        episode = generate_episode_eps_policy(env, Q, eps)
        eps -= eps_delta

        # for each state-action pair, get return and update q-table and policy
        Q, policy = improve_q_from_episode(
            Q, policy, episode, alpha, gamma, every_visit)

    return policy, Q


# obtain the estimated optimal policy and action-value function
start_time = time.time()
policy, Q = mc_control(env, 500000, 0.01)  # eps will go from 1 to .1 in 250000 episodes
end_time = time.time()
print('Time', end_time-start_time)

# obtain the corresponding state-value function
V = dict((k, np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

# plot the policy
plot_policy(policy)
