from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np
import cvxpy as cp


class ConstraintDistanceOptimizer:
    def __init__(self, points, constraints, solver="MOSEK"):

        self.points = points
        self.g_vals = constraints
        self.N_s = points.shape[0]
        self.solver = solver

        self.constraints = []
        self.objective = None
        self.problem = None

        self.r = cp.Variable(self.N_s, integer=False)
        self.optimal_points = None

        self.criterion = cp.sum(self.r * cp.square(self.g_vals))

        # optional attributes
        self.points_2 = None
        self._points_2 = None

    def optimize(self):
        self.constraints.extend([
            self.r >= 1e-8,
            cp.sum(self.r) == 1,
        ])
        spread_term = cp.sum(distance_matrix(self.points, self.points))
        distance_term = self.r @ distance_matrix(self.points, self.points) @ self.r.T

        self.objective = cp.Minimize(self.criterion - 0.001 * cp.sum(cp.entr(self.r)) - distance_term)
        self.problem = cp.Problem(
            self.objective,
            self.constraints,
        )
        self.problem.solve(solver=self.solver)

        return self.r.value, self.objective.value

    def fetch_optimal_points(self, tol=1e-5):
        return self.r.value >= tol


if __name__ == '__main__':
    grid_reso = 31j

    x1, x2 = np.mgrid[-1:1:grid_reso, -1:1:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    backoff_epsilon = 0.0
    cons = np.array([
        -(X[:, 1] - 2.12 * X[:, 0] ** 2 - 0.67 * X[:, 0] + 0.22),
    ]).T
    opt1 = ConstraintDistanceOptimizer(X, cons - backoff_epsilon)
    r, obj = opt1.optimize()

    opt_r_idx = r >= 1e-6
    colors = ["tab:red", "tab:green", "tab:blue", "magenta"]
    hatch_styles = ["+"]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    contour = axes.tricontourf(
        X[:, 0],
        X[:, 1],
        cons[:, 0],
        levels=np.linspace(cons.min(), 0, 101),
        colors=colors[0],
        linestyles="dashed",
    )
    # hatches = axes.tricontourf(
    #     X[:, 0],
    #     X[:, 1],
    #     cons[:, 0],
    #     levels=np.linspace(cons.min(), 0, 5),
    #     colors="none",
    #     edgecolor=colors[0],
    #     hatches=hatch_styles[0],
    # )
    axes.scatter(
        X[:, 0],
        X[:, 1],
        c="tab:blue",
        alpha=0.5,
        s=5,
    )
    axes.scatter(
        X[opt_r_idx][:, 0],
        X[opt_r_idx][:, 1],
        c="none",
        marker="H",
        edgecolor="tab:green",
        s=100,
    )

    axes.set_xlabel("$x_1$")
    axes.set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig("constraint_distance_entropy.png", dpi=360)
    plt.show()

    print(obj)
    print(r)
