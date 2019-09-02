import random
from collections import deque

import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity,seed):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.seed = random.seed(seed)

    def add(self, transition, unused_td_error=None):
        self.memory.append(transition)

    def sample(self, batch_size, beta=None):
        return random.sample(self.memory, batch_size), np.ones(batch_size), None

    def __len__(self):
        return len(self.memory)
