import sys
import platform
import argparse

import numpy as np
from numpy_ringbuffer import RingBuffer
import copy
from scipy import signal

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from setup import unityagents
from unityagents import UnityEnvironment

from code.A2CAgent import A2CAgent
from code.Actor import Actor
from code.Critic import Critic
from code.advantage_estimators import n_step, gae


parser = argparse.ArgumentParser(description='Train or execute an agent in A2C-GAE in the Unity Reacher environment.' +
                                 'Models are stored and loaded in the file final.pth.')
parser.add_argument('-t', '--train', dest='train_mode', action='store_true',
                    help='train a new model and store it as final.pth')
train_mode = parser.parse_args().train_mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# setup the environment
env = None
system = platform.system()
if system == 'Linux':
    env = UnityEnvironment(file_name="setup/Reacher_Linux/Reacher.x86_64")
elif system == 'Darwin':
    env = UnityEnvironment(file_name="setup/Reacher.app")
elif system == 'Windows':
    env = UnityEnvironment(
        file_name="setup/Reacher_Windows_x86_64/Reacher.exe")
else:
    print('Cannot find environment for this system.')
    exit(0)

# use the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]


# create an advantage actor critic agent
agent = A2CAgent(state_size, action_size, gae, n=8)


def execute_episode(agent, env, train_mode):
    # prepare the environment
    scores = np.zeros(num_agents)
    env_info = env.reset(train_mode)[brain_name]

    # get the initial state
    states = env_info.vector_observations
    while True:

        # evaluate the current state
        actions = agent.act(states)

        # execute the chosen action and get the outcome
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        # store the experience (also automatically learn if required)
        if train_mode:
            agent.store(states, actions, rewards, next_states, dones)

        # prepare for the next iteration
        states = next_states
        scores += rewards

        if np.any(dones):
            break

    # return the total rewards obtained
    return np.max(scores)


def train(agent, env, episodes=600, consecutive_episodes=100, show_output=True, save_as=None):

    results = [None] * episodes
    best_avg_score = 0
    current_avg_score = 0

    # reset the agent to start learning from scratch
    agent.reset()
    for i in range(episodes):

        # execute all the episodes and store the results
        score = execute_episode(agent, env, train_mode=True)
        results[i] = score

        # store the trained model if it is requested
        if i+1 >= 100 and save_as is not None:

            # but only if the model actually improved
            current_avg_score = np.mean(np.array(results[i-99:i+1]))
            if current_avg_score > best_avg_score:
                best_avg_score = current_avg_score
                torch.save(agent.actor.state_dict(), '{}.pth'.format(save_as))

        if show_output:
            print("\rEpisode: {}, Score: {:.2f}, Avg: {:.2f}".format(
                i+1, score, current_avg_score), end="")
            sys.stdout.flush()
    if show_output:
        print()

    # use convolutions to calculate the mean, summarizing the training step
    results = np.array(results)
    mean = signal.convolve(results, np.ones(
        [consecutive_episodes]) / consecutive_episodes, mode='valid')
    return mean, results


if train_mode:
    # train
    mean, full_report = train(agent, env, save_as='final')

    if np.any(mean > 30):
        episode_solved = np.argmax(mean > 30) + 100
        print('Solved after {} episodes'.format(episode_solved))

    max_mean, max_mean_i = np.max(mean), np.argmax(mean)
    print('Best avg. score over 100 consecutive episodes: {} achieved during episodes {} ... {}'.format(
        max_mean, max_mean_i - 99, max_mean_i))
else:
    # play
    agent.actor.load_state_dict(torch.load('final.pth', map_location='cpu'))
    agent.learning = False
    score = execute_episode(agent, env, train_mode=False)
    print('Score: {}'.format(score))

env.close()
