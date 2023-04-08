import numpy as np
import torch
import torch.nn.functional as F
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise

#####################################
########## hyperparameters ##########
#####################################
gamma = 0.99
tau = 0.01
lr_actor = 0.0005
lr_critic = 0.0005
actions_lb = -1
actions_ub = +1
device = 'cpu'
random_seed = 72



class DDPGAgent:

    def __init__(self, n_states, n_actions):
        """
        Initialize an Agent object.
        :param env: Environment
        """

        # general
        self.action_lb = actions_lb
        self.action_ub = actions_ub
        self.n_states = n_states
        self.n_actions = n_actions

        # actor
        self.actor_local = Actor(self.n_states, self.n_actions, 256, 128, random_seed).to(device)
        self.actor_target = Actor(self.n_states, self.n_actions, 256, 128, random_seed).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # critic
        self.critic_local = Critic(self.n_states, self.n_actions, 256, 128, random_seed).to(device)
        self.critic_target = Critic(self.n_states, self.n_actions, 256, 128, random_seed).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        # noise
        self.noise = OrnsteinUhlenbeckNoise(size=self.n_actions, mu=0., theta=0.1, sigma=0.2, random_seed=random_seed)



    def select_action(self, state, add_noise=True):
        """
        Select an action from the input state.
        :param state:
        :param add_noise:
        :return:
        """
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            action += self.noise.generate()
        return np.clip(action, self.action_lb, self.action_ub)



    def update_local_networks(self, states, actions, rewards, next_states, dones):
        """
        Update the local networks (actor and critic) using the experiences sampled from memory.
        """
        # convert to tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        # update actor and critic networks
        self._update_actor(states)
        self._update_critic(states, actions, rewards, next_states, dones)

    def update_target_networks(self):
        """Soft update model parameters of the target network towards the local network."""
        self._soft_update(self.actor_local, self.actor_target)
        self._soft_update(self.critic_local, self.critic_target)

    def _update_actor(self, states):
        """
        Update the actor network using Q(s, mu(s)) as the loss function,
        where mu(s) is the output of the actor network (predicted action).
        """
        predicted_actions = self.actor_local(states)
        loss = - self.critic_local(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """ we want to minimize the loss function L = (Q(s,a) - y)**2
        where y = r + gamma * Q(s', mu(s')) * (1 - done)
        """
        # get the predicted next actions
        predicted_next_actions = self.actor_target(next_states)
        # get the predicted Q values for the next states and actions
        Q_targets_next = self.critic_target(next_states, predicted_next_actions)
        # compute the target Q values
        Q_targets = rewards.view(-1, 1)+ (gamma * Q_targets_next * (1 - dones).view(-1, 1))
        # get the predicted Q values for the current states and actions
        Q_expected = self.critic_local(states, actions)
        # compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _soft_update(self, local_network, target_network):
        """
        Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: The model that is being trained
        :param target_network: The model that is being updated
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()
