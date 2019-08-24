import numpy as np


class Scheduler:
    def __init__(self, start: float, end: float, steps: int):
        """
        Initialise a noise scheduler for linearly decreasing the magnitude of the noise
        :param start: the starting noise value (usually 1.0)
        :param end:
        :param steps:
        """
        self.end = end
        self.start = start
        self.steps = steps
        self.progression = np.linspace(start, end, steps)

    def get(self, index: int):
        if index < self.steps:
            return self.progression[index]
        else:
            return self.end
