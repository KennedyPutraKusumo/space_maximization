from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt


class CvxHullVolCalculator:
    def __init__(self, points):
        self.points = points
        self.cvxhull = None
        self.vol = None

    def compute_volume(self):
        self.cvxhull = ConvexHull(self.points)
        self.vol = self.cvxhull.volume
