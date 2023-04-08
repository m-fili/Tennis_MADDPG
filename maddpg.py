import torch
from ddpg import DDPGAgent
from replay_buffer import ReplayBuffer


batch_size = 512
buffer_size = int(1e6)
local_update_freq = 1
target_update_freq = 10
random_seed = 72
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:


    def __init__(self, env):
        # brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        self.n_agents = len(env_info.agents)
        self.n_states = env_info.vector_observations.shape[1]
        self.n_actions = brain.vector_action_space_size
        # memory
        self.memory = ReplayBuffer(buffer_size, batch_size)
        # agents
        self.ddpg_agents = [DDPGAgent(self.n_states, self.n_actions) for _ in range(self.n_agents)]



    def step(self, states, actions, rewards, next_states, dones, time_step):
        # Save experience in replay memory
        self.memory.add_experience(states, actions, rewards, next_states, dones)

        if self.memory.ready_to_learn():
            if time_step % local_update_freq == 0:
                self.update_locals()  # update the local network

            if time_step % target_update_freq == 0:
                self.update_targets()  # soft update the target network towards the actual networks




    def select_action(self, states):
        actions = []
        for i, ddpg_agent in enumerate(self.ddpg_agents):
            actions.append(ddpg_agent.select_action(states[i]))
        return actions


    def update_locals(self):
        # learn from experiences in the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample_experience()

        for i, ddpg_agent in enumerate(self.ddpg_agents):
            ddpg_agent.update_local_networks(states[:,i], actions[:,i], rewards[:,i], next_states[:,i], dones[:,i])



    def update_targets(self):
        # soft update the target network towards the actual networks
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.update_target_networks()




    def reset(self):
        """Reset the noise process"""
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.reset()


    def save(self):
        """Save all agent's actor and critic parameters"""
        for i, ddpg_agent in enumerate(self.ddpg_agents):
            torch.save(ddpg_agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
            torch.save(ddpg_agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(i))


    def load(self):
        """Load all agent's actor and critic parameters"""
        for i, ddpg_agent in enumerate(self.ddpg_agents):
            ddpg_agent.actor_local.load_state_dict(torch.load('saved_models/checkpoint_actor_{}.pth'.format(i)))
            ddpg_agent.critic_local.load_state_dict(torch.load('saved_models/checkpoint_critic_{}.pth'.format(i)))
