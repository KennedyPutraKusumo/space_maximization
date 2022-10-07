from prototype_implementation.solvers.EffortSolver import EffortSolver
from scipy.optimize import minimize
import numpy as np


class ScipySolver(EffortSolver):
    def __init__(self, bounds, criterion, n_runs, x0=None, verbose=False):
        self.bounds = bounds
        self.ndim = bounds.shape[0]
        self.criterion = criterion()
        self.n_runs = n_runs
        self.optimizer = "CPLEX"
        super().__init__()
        self.verbose = verbose
        self.x0 = x0

    def solve(self):
        n_dim = self.bounds.shape[0]

        if self.x0 is None:
            self.x0 = np.zeros(N)
        scipy_result = minimize(
            self.criterion.obj_func,
            x0=self.x0,
            method="SLSQP",
            args=self.points,
            constraints=[
                {"type": "ineq", "fun": lambda y: np.sum(y) - self.n_runs},
            ],
            bounds=[(0, 1) for _ in range(N)],
        )
        return scipy_result
