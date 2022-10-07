from prototype_implementation.criteria.Criterion import DiscreteCriterion
from prototype_implementation.utilities.PointNormalizer import Normalizer
from scipy.spatial.distance import cdist
import cvxpy as cp
import numpy as np


class MaximalCovering(DiscreteCriterion):
    def __init__(self):
        super().__init__()
        self.name = "Maximal Covering"
        self.y = None
        self.z = None
        self.eta = None
        self.objective = None
        self.constraints = None
        self.N = None
        self.n_dim = None
        self.pi_pj = None

    def construct_cvxpy_problem(self):
        normalizer = Normalizer(self.points)
        self.points = normalizer.normalize()

        self.pi_pj = cdist(
            self.points,
            self.points,
            metric="euclidean",
        )
        self.N, self.n_dim = self.determine_points_shape()

        self.y = cp.Variable(
            shape=self.N,
            boolean=True,
        )
        self.z = cp.Variable(
            shape=(self.N, self.N),
            boolean=True,
        )
        self.eta = cp.Variable()
        self.objective = cp.Minimize(self.eta)

        self.constraints = [
            self.eta >= cp.sum(self.z[:, j] * self.pi_pj[:, j]) for j in range(self.N)
        ]
        self.constraints += [
            cp.sum(self.z[:, j], axis=0) == 1 for j in range(self.N)
        ]
        self.constraints += [
            self.y[i] >= self.z[i, j] for i in range(self.N) for j in range(self.N)
        ]
        self.constraints += [
            self.n_runs >= cp.sum(self.y)
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
        coverage = -np.infty
        for point in self.points[y < 1]:
            distance_to_centroids = cdist([point], self.points[y > 0], metric=metric)
            distance_to_closest_centroid = np.min(distance_to_centroids)
            coverage = np.max([coverage, distance_to_closest_centroid])
        return coverage
