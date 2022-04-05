from prototype_implementation.criteria.BasalCriterion import DiscreteBasalCriterion
from prototype_implementation.utilities.PointNormalizer import Normalizer
from scipy.spatial.distance import cdist
import cvxpy as cp


class MaximalSpread(DiscreteBasalCriterion):
    def __init__(self):
        super().__init__()

    def construct_cxvpy_problem(self):
        normalizer = Normalizer(self.points)
        points = normalizer.normalize()

        N, n_dim = self.determine_points_shape()
        pi_pj = cdist(
            points,
            points,
            metric="euclidean",
        )
        D = cp.max(pi_pj)
        y = cp.Variable(shape=N, boolean=True)
        eta = cp.Variable()
        objective = cp.Maximize(eta)
        constraints = [
            eta <= pi_pj[i, j] + (1 - y[i]) * D + (1 - y[j]) * D for i in range(N) for j in range(N) if j > i
        ]
        constraints += [
            cp.sum(y) >= self.n_runs
        ]
        problem = cp.Problem(
            objective,
            constraints,
        )
        return problem
