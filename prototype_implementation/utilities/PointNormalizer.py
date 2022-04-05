import numpy as np


class Normalizer:
    def __init__(self, points):
        self.points = points

    def normalize(self):
        deviations = self.points - np.min(self.points, axis=0)[None, :]
        interval_size = np.max(self.points, axis=0) - np.min(self.points, axis=0)
        self.points = -1 + 2 * deviations / interval_size[None, :]
        return self.points
