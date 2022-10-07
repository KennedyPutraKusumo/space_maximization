from prototype_implementation.solvers.EffortSolver import EffortSolver


class CvxpyEffortSolver(EffortSolver):
    """
    Input: 2D np.array with n_points X n_dim
    """
    def __init__(self, points, criterion, n_runs, verbose=True, fix_y=None):
        self.points = points
        self.n_points, self.n_dim = points.shape
        self.criterion = criterion()
        self.n_runs = n_runs
        self.problem = None
        self.optimizer = "CPLEX"
        super().__init__()
        self.verbose = verbose
        self.fix_y = fix_y

    def solve(self):
        self.criterion.points = self.points
        self.criterion.n_runs = self.n_runs
        self.problem = self.criterion.construct_cvxpy_problem()
        if self.fix_y is not None:
            self.problem = self.criterion.fix_y(self.fix_y)
        self.problem.solve(solver=self.optimizer, verbose=self.verbose)


if __name__ == '__main__':
    import numpy as np
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    from prototype_implementation.criteria.MaximalCovering import MaximalCovering
    from prototype_implementation.criteria.MaximalSpread import MaximalSpread

    grid_reso = 11j
    n_centroids = 4

    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    solver1 = CvxpyEffortSolver(
        points=Y,
        criterion=MaximalCovering,
        n_runs=n_centroids,
    )
    solver1.solve()
    print(solver1.problem.variables()[2].value)
    solver2 = CvxpyEffortSolver(
        points=Y,
        criterion=MaximalSpread,
        n_runs=n_centroids,
    )
    solver2.solve()
    print(solver2.problem.variables()[1].value)
