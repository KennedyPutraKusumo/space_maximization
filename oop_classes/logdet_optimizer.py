from oop_classes.point_normalizer import Normalizer
import numpy as np
import cvxpy as cp


class LogDetOptimizer:
    def __init__(self, points, criterion=None, solver="MOSEK"):

        self.points = points
        self.N_s = points.shape[0]
        self.solver = solver

        self.constraints = []
        self.objective = None
        self.problem = None

        if criterion is None:
            criterion = cp.log_det
        self.criterion = criterion

        self.r = cp.Variable(self.N_s, integer=False)
        self._points = None
        self.optimal_points = None

        # optional attributes
        self.points_2 = None
        self._points_2 = None

    def logdet_criterion(self, points, r):
        normalizer = Normalizer(points)
        _points = normalizer.normalize()
        _points = np.append(np.ones((points.shape[0], 1)), _points, axis=1)
        sign, logdet = np.linalg.slogdet(_points.T @ np.diag(r) @ _points)
        return sign, logdet

    def add_pareto_constraint(self, epsilon):
        normalizer = Normalizer(self.points_2)
        self._points_2 = normalizer.normalize()
        self.constraints.extend([
            self.criterion(self._points_2.T @ cp.diag(self.r) @ self._points_2) >= epsilon
        ])

    def optimize(self):
        normalizer = Normalizer(self.points)
        self._points = normalizer.normalize()

        self._points = np.append(np.ones((self.N_s, 1)), self._points, axis=1)
        self.constraints.extend([
            self.r >= 1e-8,
            self.r <= 1,
            cp.sum(self.r) <= 1,
        ])
        self.objective = cp.Maximize(self.criterion(self._points.T @ cp.diag(self.r) @ self._points))
        self.problem = cp.Problem(
            self.objective,
            self.constraints,
        )
        self.problem.solve(solver=self.solver)

        return self.r.value, self.objective.value

    def fetch_optimal_points(self, tol=1e-5):
        return self.r.value >= tol


if __name__ == '__main__':
    grid_reso = 11j

    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    opt1 = LogDetOptimizer(X)
    r, obj = opt1.optimize()
    print(obj)
    print(r)

    from matplotlib import pyplot as plt

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        x1,
        x2,
    )
    axes.scatter(
        x1,
        x2,
        s=1000 * r,
        marker="H",
        facecolor="none",
        edgecolor="tab:red",
    )
    fig.tight_layout()
    plt.show()
