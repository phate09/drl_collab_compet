import numpy as np
from collections import deque
import torch
from utility.PrioritisedExperienceReplayBuffer_cython import PrioritizedReplayBuffer


class TDBuffer:
    def __init__(self, n_steps, gamma, memory: PrioritizedReplayBuffer, evaluate_fn, device):
        self.n_steps = n_steps
        self.gamma = gamma
        self.memory: PrioritizedReplayBuffer = memory
        self.evaluate_fn = evaluate_fn
        self.device = device
        self.buffer: deque = deque(maxlen=n_steps)

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) == self.n_steps:
            self._push_to_memory()

    def _push_to_memory(self):
        if len(self.buffer) == 0:
            return
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        initial_states, initial_actions, initial_rewards, next_states, dones = zip(*self.buffer[0])
        # initial_state = states
        # initial_action = actions
        total_reward = torch.zeros(1).to(self.device)
        for i, item in enumerate(self.buffer):
            states, actions, rewards, dones, next_states = zip(*item)
            total_reward += rewards[0]  # self.gamma ** i *
        # td_error = self.evaluate_fn(initial_state, initial_action, total_reward, next_states, dones)  # total_reward + self.gamma ** len(self.buffer) * self.evaluate_fn(next_states) - self.evaluate_fn(initial_state, initial_action)
        self.memory.add((list(initial_states), list(initial_actions), total_reward, list(next_states), list(dones)), 0)

    def flush(self):
        """When the episode is finished empties the buffer saving it to memory"""
        while len(self.buffer) > 0:
            self.buffer.popleft()
            self._push_to_memory()
