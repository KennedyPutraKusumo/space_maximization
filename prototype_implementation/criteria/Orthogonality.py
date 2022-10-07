from prototype_implementation.criteria.Criterion import DiscreteCriterion
from prototype_implementation.utilities.PointNormalizer import Normalizer
import numpy as np
import cvxpy as cp


class Orthogonality(DiscreteCriterion):
    def __init__(self):
        super().__init__()
        self.name = "Maximal Orthogonality"

    @staticmethod
    def obj_func(y, points):
        return np.linalg.det(points.T @ np.diag(y) @ points)

    def construct_cvxpy_problem(self):
        normalizer = Normalizer(self.points)
        points = normalizer.normalize()
        N, n_dim = self.determine_points_shape()

        y = cp.Variable(
            shape=N,
            boolean=True,
        )
        orthogonality = cp.log_det(points.T @ cp.diag(y) @ points)
        # orthogonality = cp.pnorm(points.T @ cp.diag(y) @ points, -4)
        # orthogonality = cp.sum_smallest(points.T @ cp.diag(y) @ points, n_dim)
        # orthogonality = cp.min(points.T @ cp.diag(y) @ points)
        # orthogonality = -cp.log_sum_exp(points.T @ cp.diag(y) @ points)
        # orthogonality = cp.sum(points, cp.diag(y))
        # orthogonality = -cp.trace(points.T @ cp.diag(y) @ points)
        objective = cp.Maximize(orthogonality)
        constraints = []
        constraints += [
            self.n_runs >= cp.sum(y)
        ]
        problem = cp.Problem(
            objective,
            constraints,
        )
        return problem
