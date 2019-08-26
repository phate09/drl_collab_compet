import numpy as np


class Scheduler:
    def __init__(self, start: float, end: float, steps: int, warmup_steps: int = 0):
        """
        Initialise a noise scheduler for linearly decreasing the magnitude of the noise
        :param start: the starting noise value (usually 1.0)
        :param end:
        :param steps:
        """
        self.end = end
        self.start = start
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.progression = np.linspace(start, end, steps)

    def get(self, index: int):
        if index < self.steps:
            return self.progression[max(index - self.warmup_steps, 0)]
        else:
            return self.end
