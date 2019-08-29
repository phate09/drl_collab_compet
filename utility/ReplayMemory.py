import random
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, transition, unused_td_error=None):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size, beta=None):
        return random.sample(self.memory, batch_size), np.ones(batch_size), None

    def __len__(self):
        return len(self.memory)
