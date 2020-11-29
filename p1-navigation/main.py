from setup import unityagents
from unityagents import UnityEnvironment

import numpy as np

env = UnityEnvironment(file_name="setup/Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]
score = 0
while True:
    action = np.random.randint(action_size)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:
        break

print("Score: {}".format(score))

env.close()


"""
agent = QNetworkAgent(DQN, state_size, action_size, UniformReplayBuffer(100_000), dt_dqn)
agent.q_local.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))

env_info = env.reset(train_mode=False)[brain_name]
agent.learning = False
state = env_info.vector_observations[0]
score = 0
done = False
while not done:
    action = agent.act(state)

    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]

    agent.store(state, action, reward, next_state, done)
    score += reward
    state = next_state
print(score)
"""
