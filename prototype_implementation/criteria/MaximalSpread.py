from prototype_implementation.criteria.Criterion import DiscreteCriterion
from prototype_implementation.utilities.PointNormalizer import Normalizer
from scipy.spatial.distance import cdist
import cvxpy as cp
import numpy as np


class MaximalSpread(DiscreteCriterion):
    def __init__(self):
        super().__init__()
        self.name = "Maximal Spread"
        self.N = None
        self.n_dim = None
        self.pi_pj = None
        self.D = None
        self.y = None
        self.eta = None
        self.objective = None
        self.constraints = None

    def construct_cvxpy_problem(self):
        normalizer = Normalizer(self.points)
        self.points = normalizer.normalize()

        self.N, self.n_dim = self.determine_points_shape()
        self.pi_pj = cdist(
            self.points,
            self.points,
            metric="euclidean",
        )
        self.D = cp.max(self.pi_pj)
        self.y = cp.Variable(
            shape=self.N,
            boolean=True,
        )
        self.eta = cp.Variable()
        self.objective = cp.Maximize(self.eta)
        self.constraints = [
            self.eta <= self.pi_pj[i, j] + (1 - self.y[i]) * self.D + (1 - self.y[j]) * self.D for i in range(self.N) for j in range(self.N) if j > i
        ]
        self.constraints += [
            cp.sum(self.y) >= self.n_runs
        ]
        self.cvxpy_problem = cp.Problem(
            self.objective,
            self.constraints,
        )
        return self.cvxpy_problem

    def fix_y(self, y):
        self.constraints += [
            self.y == y
        ]
        self.cvxpy_problem = cp.Problem(
            self.objective,
            self.constraints,
        )
        return self.cvxpy_problem

    def evaluate_criterion(self, y, metric="euclidean"):
        spread = np.infty
        for point in self.points[y > 0]:
            distance_between_centroids = cdist([point], self.points[y > 0], metric=metric)
            spread = np.min(
                [spread, np.min(distance_between_centroids[distance_between_centroids > 0])]
            )
        return spread
