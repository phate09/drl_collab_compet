import numpy as np
from collections import deque
import torch
from utility.PrioritisedExperienceReplayBuffer_cython import PrioritizedReplayBuffer


class TDBuffer:
    def __init__(self, n_steps, gamma, memory: PrioritizedReplayBuffer, critic, target_critic, target_actor, device):
        self.n_steps = n_steps
        self.gamma = gamma
        self.memory: PrioritizedReplayBuffer = memory
        self.critic = critic
        self.target_critic = target_critic
        self.target_actor = target_actor
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
        states, actions, rewards, dones, next_states = self.buffer[0]
        initial_state = states
        initial_action = actions
        total_reward = torch.zeros(rewards.size()).to(self.device)
        for i, item in enumerate(self.buffer):
            states, actions, rewards, dones, next_states = item
            total_reward += rewards  # self.gamma ** i *
        # td_error = self.calculate_td_errors(initial_state, initial_action, total_reward, next_states, dones)  # total_reward + self.gamma ** len(self.buffer) * self.evaluate_fn(next_states) - self.evaluate_fn(initial_state, initial_action)
        # for i in range(states.size()[0]):
        self.memory.add((initial_state, initial_action, total_reward, next_states, dones), 0)#abs(td_error.item()))

    def flush(self):
        """When the episode is finished empties the buffer saving it to memory"""
        while len(self.buffer) > 0:
            self.buffer.popleft()
            self._push_to_memory()

    def calculate_td_errors(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        self.critic.eval()
        concat_states = torch.cat([states, actions], dim=0)
        suggested_next_action = self.target_actor(next_states.unsqueeze(dim=0))
        concat_next_states = torch.cat([next_states, suggested_next_action.squeeze(dim=0)], dim=0)
        dones = (1 - dones).float()
        target_Q1, target_Q2 = self.target_critic(concat_next_states.unsqueeze(dim=0))
        Q1, Q2 = self.critic(concat_states.unsqueeze(dim=0))
        y = rewards + np.power(self.gamma, self.n_steps) * torch.min(target_Q1, target_Q2) * dones
        td_errors = torch.min(y - Q1, y - Q2)
        self.critic.train()
        return td_errors  # calculate the td-errors, maybe use GAE
