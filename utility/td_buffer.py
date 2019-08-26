import numpy as np
from collections import deque


class TDBuffer:
    def __init__(self, n_steps, gamma, memory, evaluate_fn):
        self.n_steps = n_steps
        self.gamma = gamma
        self.memory = memory
        self.evaluate_fn = evaluate_fn
        self.buffer: deque = deque(maxlen=n_steps)

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer == self.n_steps):
            self._push_to_memory()

    def _push_to_memory(self):
        state, action, reward, next_state, done = self.buffer[0]
        initial_state = state
        initial_action = state
        total_reward = 0
        for i, item in enumerate(self.buffer):
            state, action, reward, next_state, done = item
            total_reward += self.gamma ** i * reward
        td_error = total_reward + self.gamma ** len(self.buffer) * self.evaluate_fn(next_state) - self.evaluate_fn(initial_state, initial_action)
        self.memory.add(initial_state, initial_action, total_reward, done, next_state, td_error)

    def flush(self):
        """When the episode is finished empties the buffer saving it to memory"""
        while len(self.buffer) > 0:
            self.buffer.popleft()
            self._push_to_memory()
