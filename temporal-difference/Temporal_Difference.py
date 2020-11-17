from plot_utils import plot_values
import check_test
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')  # it doesn't work on my computer otherwise

env = gym.make('CliffWalking-v0')


def get_greedy_action(Q, state):
    # if there are two or more actions for which Q[s][a] is maximized, choose uniformly between them
    greedy_actions = np.argwhere(Q[state] == np.amax(Q[state]))
    greedy_actions = greedy_actions.flatten()
    return np.random.choice(greedy_actions)


def get_random_action(Q, state):
    return np.random.choice(np.arange(Q[state].size))


def eps_greedy_policy(Q, state, eps):
    return get_random_action(Q, state) if np.random.uniform() <= eps \
        else get_greedy_action(Q, state)


def get_estimated_return_for_eps_greedy_policy(Q, state, eps):
    # pi(a|s) = eps/|A(s)|            else
    prob = np.ones(Q[state].shape) * (eps / Q[state].size)
    # pi(a|s) = 1 - eps + eps/|A(s)|  for greedy
    prob[np.argmax(Q[state])] = 1 - eps + eps / Q[state].size
    return np.dot(prob, Q[state])


def sarsa(env, num_episodes, alpha, gamma=1.0, eps=1, final_eps=0.1, stop_eps_after=0.5):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))

    # eps will decrease linearly and reach final_eps in episode stop_eps_at_episode
    final_eps = min(eps, final_eps)
    stop_eps_at_episode = num_episodes * stop_eps_after - 1
    eps_delta = (eps - final_eps) / stop_eps_at_episode

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        S_t = env.reset()  # get initial state S_t
        A_t = eps_greedy_policy(Q, S_t, eps)  # choose initial action A_t

        while True:

            # execute A_t, get R_t+1, S_t+1
            S_t1, R_t1, done, _ = env.step(A_t)

            if done:
                # if the episode is completed, update Q[S_t][A_t] using an estimated return of zero
                Q[S_t][A_t] += alpha * (R_t1 - Q[S_t][A_t])
                break

            A_t1 = eps_greedy_policy(Q, S_t1, eps)  # choose action A_t+1

            # update Q[S_t][A_t] using the estimated return from Q[S_t+1][A_t+1]
            Q[S_t][A_t] += alpha * (R_t1 + gamma * Q[S_t1][A_t1] - Q[S_t][A_t])

            S_t = S_t1
            A_t = A_t1

        eps -= eps_delta

    return Q


def q_learning(env, num_episodes, alpha, gamma=1.0, eps=1, final_eps=0.1, stop_eps_after=0.5):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))

    # eps will decrease linearly and reach final_eps in episode stop_eps_at_episode
    stop_eps_at_episode = num_episodes * stop_eps_after - 1
    eps_delta = (eps - final_eps) / stop_eps_at_episode

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        S_t = env.reset()  # get initial state S_t

        done = False
        while not done:

            # choose action A_t according to the behaviour policy
            A_t = eps_greedy_policy(Q, S_t, eps)
            # execute A_t, get R_t+1, S_t+1
            S_t1, R_t1, done, _ = env.step(A_t)

            # choose A_max according to the target policy
            A_max = get_greedy_action(Q, S_t1)
            # update Q[S_t][A_t] using the estimated return from Q[S_t+1][A_max]
            Q[S_t][A_t] += alpha * \
                (R_t1 + gamma * Q[S_t1][A_max] - Q[S_t][A_t])

            S_t = S_t1

        eps -= eps_delta

    return Q


def expected_sarsa(env, num_episodes, alpha, gamma=1.0, eps=1, final_eps=0.1, stop_eps_after=0.5):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))

    # eps will decrease linearly and reach final_eps in episode stop_eps_at_episode
    stop_eps_at_episode = num_episodes * stop_eps_after - 1
    eps_delta = (eps - final_eps) / stop_eps_at_episode

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        S_t = env.reset()  # get initial state S_t

        done = False
        while not done:

            # choose action A_t according to the behaviour policy
            A_t = eps_greedy_policy(Q, S_t, eps)
            # execute A_t, get R_t+1, S_t+1
            S_t1, R_t1, done, _ = env.step(A_t)

            G = get_estimated_return_for_eps_greedy_policy(Q, S_t1, eps)
            # update Q[S_t][A_t] using G
            Q[S_t][A_t] += alpha * (R_t1 + gamma * G - Q[S_t][A_t])

            S_t = S_t1

        eps -= eps_delta

    return Q


def test_sarsa():
    # obtain the estimated optimal policy and corresponding action-value function
    # eps=.1 safe path, eps = .01 optimal path
    Q_sarsa = sarsa(env, 5000, .01, eps=0.01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(
        Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = (
        [np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


def test_sarsamax():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01, eps=0.1)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(
        Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key])
                 if key in Q_sarsamax else 0 for key in np.arange(48)])


def test_expsarsa():
    # obtain the estimated optimal policy and corresponding action-value function
    # higher alpha but lower eps
    Q_expsarsa = expected_sarsa(env, 5000, 0.5, eps=0.01, final_eps=0.001)

    # print the estimated optimal policy
    policy_expsarsa = np.array([np.argmax(
        Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key])
                 if key in Q_expsarsa else 0 for key in np.arange(48)])


test_sarsa()
test_sarsamax()
test_expsarsa()
