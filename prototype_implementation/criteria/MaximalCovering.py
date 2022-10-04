from prototype_implementation.criteria.Criterion import DiscreteCriterion
from prototype_implementation.utilities.PointNormalizer import Normalizer
from scipy.spatial.distance import cdist
import cvxpy as cp


class MaximalCovering(DiscreteCriterion):
    def __init__(self):
        super().__init__()
        self.name = "Maximal Covering"

    def construct_cxvpy_problem(self):
        normalizer = Normalizer(self.points)
        points = normalizer.normalize()

        pi_pj = cdist(
            points,
            points,
            metric="euclidean",
        )
        N, n_dim = self.determine_points_shape()

        y = cp.Variable(
            shape=N,
            boolean=True,
        )
        z = cp.Variable(
            shape=(N, N),
            boolean=True,
        )
        eta = cp.Variable()
        objective = cp.Minimize(eta)

        constraints = [
            eta >= cp.sum(z[:, j] * pi_pj[:, j]) for j in range(N)
        ]
        constraints += [
            cp.sum(z[:, j], axis=0) == 1 for j in range(N)
        ]
        constraints += [
            y[i] >= z[i, j] for i in range(N) for j in range(N)
        ]
        constraints += [
            self.n_runs >= cp.sum(y)
        ]
        problem = cp.Problem(
            objective,
            constraints,
        )
        return problem
