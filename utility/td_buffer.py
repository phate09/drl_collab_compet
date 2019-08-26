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
        if len(self.buffer)==0:
            return
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        states, actions, rewards, dones, next_states = self.buffer[0]
        initial_state = states
        initial_action = actions
        total_reward = torch.zeros(rewards.size()).to(self.device)
        for i, item in enumerate(self.buffer):
            states, actions, rewards, dones, next_states = item
            total_reward += rewards  # self.gamma ** i *
        td_error = self.evaluate_fn(initial_state, initial_action, total_reward, next_states, dones)  # total_reward + self.gamma ** len(self.buffer) * self.evaluate_fn(next_states) - self.evaluate_fn(initial_state, initial_action)
        for i in range(states.size()[0]):
            self.memory.add((initial_state[i], initial_action[i], total_reward[i], next_states[i], dones[i]), abs(td_error[i].item()))

    def flush(self):
        """When the episode is finished empties the buffer saving it to memory"""
        while len(self.buffer) > 0:
            self.buffer.popleft()
            self._push_to_memory()
