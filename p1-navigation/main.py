import sys
import argparse

import numpy as np
from numpy_ringbuffer import RingBuffer
from scipy import signal

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from setup import unityagents
from unityagents import UnityEnvironment

from code.DQN import DQN
from code.DuelingDQN import DuelingDQN
from code.QNetworkAgent import QNetworkAgent
from code.UniformReplayBuffer import UniformReplayBuffer
from code.dt_estimators import dt_dqn, dt_double_dqn


parser = argparse.ArgumentParser(description='Train or execute a Dueling Double DQN agent in the Unity Banana environment.' +
                                 'Models are stored and loaded in the file final.pth.')
parser.add_argument('-t', '--train', dest='train_mode', action='store_true',
                    help='train a new model and store it as final.pth')
train_mode = parser.parse_args().train_mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# setup the environment
env = UnityEnvironment(file_name="setup/Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

# create a dueling double dqn agent
agent = QNetworkAgent(DuelingDQN, state_size, action_size,
                      UniformReplayBuffer(100_000), dt_double_dqn)


def execute_episode(agent, env, train_mode):
    # prepare the environment
    score = 0
    done = False
    env_info = env.reset(train_mode)[brain_name]

    # get the initial state
    state = env_info.vector_observations[0]
    while not done:

        # evaluate the current state
        action = agent.act(state)

        # execute the chosen action and get the outcome
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        # store the experience (also automatically learn, from time to time)
        agent.store(state, action, reward, next_state, done)

        # prepare for the next iteration
        state = next_state
        score += reward

    # return the total rewards obtained
    return score


def train(agent, env, episodes=700, consecutive_episodes=100, show_output=True, save_as=None):

    results = [None] * episodes

    # reset the agent to start learning from scratch
    agent.reset()
    for i in range(episodes):

        # execute all the episodes and store the results
        score = execute_episode(agent, env, train_mode=True)
        results[i] = score

        if show_output:
            print("\rEpisode: {}, Score: {}".format(i+1, score), end="")
            sys.stdout.flush()
    if show_output:
        print()

    # store the trained model if requested
    if save_as is not None:
        torch.save(agent.q_local.state_dict(), '{}.pth'.format(save_as))

    # use convolutions to calculate the mean, summarizing the training step
    results = np.array(results)
    mean = signal.convolve(results, np.ones(
        [consecutive_episodes]) / consecutive_episodes, mode='valid')
    return mean, results


if train_mode:
    # train
    mean, full_report = train(agent, env, save_as='final')

    if np.any(mean > 13):
        episode_solved = np.argmax(mean > 13) + 100
        print('Solved after {} episodes'.format(episode_solved))

    max_mean, max_mean_i = np.max(mean), np.argmax(mean)
    print('Best avg. score over 100 consecutive episodes: {} achieved during episodes {} ... {}'.format(
        max_mean, max_mean_i - 99, max_mean_i))
else:
    # play
    agent.q_local.load_state_dict(torch.load('final.pth', map_location='cpu'))
    agent.learning = False
    score = execute_episode(agent, env, train_mode=False)
    print('Score: {}'.format(score))

env.close()
