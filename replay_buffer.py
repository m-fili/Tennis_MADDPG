import numpy as np
import random
from collections import deque, namedtuple


class ReplayBuffer:

    def __init__(self, buffer_length, batch_size):
        self.buffer_length = buffer_length
        self.buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.experience_tuple = namedtuple("exp_tuple", field_names=["states", "actions", "rewards", "next_states", "dones"])

    def add_experience(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to memory.
        """
        experience = self.experience_tuple(states, actions, rewards, next_states, dones)
        self.buffer.append(experience)

    def sample_experience(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        sampled_tuples = random.sample(self.buffer, k=self.batch_size)
        states = np.stack([t.states for t in sampled_tuples if t is not None])
        actions = np.stack([t.actions for t in sampled_tuples if t is not None])
        rewards = np.stack([t.rewards for t in sampled_tuples if t is not None])
        next_states = np.stack([t.next_states for t in sampled_tuples if t is not None])
        dones = np.stack([t.dones for t in sampled_tuples if t is not None])
        return states, actions, rewards, next_states, dones

    def ready_to_learn(self):
        """
        Check if there are enough samples in memory to learn from.
        """
        return len(self.buffer) >= self.batch_size
