import numpy as np
import torch
from unityagents import UnityEnvironment
import time
from maddpg import MADDPG

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_trained(env_exe_path, n_steps=100, sleep_between_frames=0):
    # Load Environment
    env = UnityEnvironment(file_name=env_exe_path)
    brain_name = env.brain_names[0]

    print('Device', DEVICE)
    print('*' * 50)

    # Load trained agent
    ddpg_agents = MADDPG(env)
    # Load the weights from file
    ddpg_agents.load()
    # Reset the environment
    ddpg_agents.reset()
    # Set the noise factor to 0
    for ddpg_agent in ddpg_agents.ddpg_agents:
        ddpg_agent.noise.noise_factor = 0.0

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)


    for _ in range(n_steps):
        actions = ddpg_agents.select_action(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        scores += env_info.rewards
        states = next_states
        time.sleep(sleep_between_frames)

    final_reward = np.max(scores)

    print('*' * 50)
    print(f"Total Reward: {final_reward:.2f}")

    env.close()


def visualize_random(env_exe_path, n_steps=100, sleep_between_frames=0):
    # Load Environment
    env = UnityEnvironment(file_name=env_exe_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    n_actions = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    rewards = np.zeros(n_agents)

    for _ in range(n_steps):
        actions = np.random.randn(n_agents, n_actions)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        rewards += env_info.rewards[0]
        time.sleep(sleep_between_frames)

    final_reward = np.max(rewards)

    print('*' * 50)
    print(f"Total Reward: {final_reward:.3f}")

    env.close()