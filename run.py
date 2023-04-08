import numpy as np
from collections import deque

print_every = 100
decay_factor = 0.999


def ddpg_multi_agent(env, ddpg_agents, n_episodes=1000, save_every=100):
    """Deep Deterministic Policy Gradient with multiple agents."""

    scores = []
    scores_deque = deque(maxlen=100)
    time_step = 1

    # get the default brain
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    for i_episode in range(1, n_episodes + 1):

        agent_scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        ddpg_agents.reset()    # reset the noise

        while True:
            actions = ddpg_agents.select_action(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            ddpg_agents.step(states, actions, rewards, next_states, dones, time_step)
            states = next_states
            agent_scores += rewards
            time_step += 1

            if np.any(dones):
                break

        for agent in ddpg_agents.ddpg_agents:
            agent.noise.noise_factor *= decay_factor

        episode_rewards = np.max(agent_scores)
        scores.append(episode_rewards)
        scores_deque.append(episode_rewards)

        print(f'\rEpisode {i_episode:04}\t Score: {episode_rewards:.5f}\t Avg Score:{np.mean(scores_deque):.5f}', end='\r')

        if i_episode % save_every == 0:
            ddpg_agents.save()
            print(f'\rEpisode {i_episode:04}\t Score: Avg Score:{np.mean(scores_deque):.5f}\t\t\t\t    ')

        if np.mean(scores_deque) >= 0.5:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_deque):.5f}')
            ddpg_agents.save()
            break

    return scores