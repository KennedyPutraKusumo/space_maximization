from oop_classes.point_normalizer import Normalizer
from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp


class EntropyOptimizer:
    def __init__(self, points, entropic_scale, criterion=None, solver="MOSEK"):

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
        self.entropic_scale = entropic_scale
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
        self.constraints.extend([
            self.r >= 1e-8,
            self.r <= 1,
            cp.sum(self.r) <= 1,
        ])
        self.objective = cp.Maximize(cp.sum(self.r * cp.abs(self._points)) + self.entropic_scale * cp.sum(cp.entr(self.r)))
        self.problem = cp.Problem(
            self.objective,
            self.constraints,
        )
        self.problem.solve(solver=self.solver)

        return self.r.value, self.objective.value

    def fetch_optimal_points(self, tol=1e-5):
        return self.r.value >= tol


if __name__ == '__main__':
    n_cuts = 3
    cut_scheme = "parabolic"
    grid_reso = 101j
    seed = np.random.randint(1, 123456789, 1)
    np.random.seed(seed)
    colors = ["tab:red", "tab:green", "tab:blue", "magenta"]
    hatch_styles = ["o", ".", "-", "|"]
    save_X_points = True

    x1, x2 = np.mgrid[-1:1:grid_reso, -1:1:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T

    # line cuts
    if cut_scheme == "linear":
        low_param = [-5, -1]
        high_param = [5, 1]
        n_param = 2
    # parabolic cuts
    elif cut_scheme == "parabolic":
        low_param = [-3, -5, -1]
        high_param = [3, 5, 1]
        n_param = 3
    else:
        low_param = 0
        high_param = 1
        n_param = 0
        n_cuts = 0

    params = np.random.uniform(
        low=low_param,
        high=high_param,
        size=(n_cuts, n_param),
    )

    fig = plt.figure(figsize=(12, 9))
    axes = fig.add_subplot(111)
    title = f"Seed: {seed}. "

    for i, param in enumerate(params):
        if cut_scheme == "linear":
            constraint_val = X[:, 1] - param[0] * X[:, 0] - param[1]
            title += f"Cut {i}: $y +{-param[0]:.2f} x +{-param[1]:.2f}$. "
        if cut_scheme == "parabolic":
            constraint_val = X[:, 1] - param[0] * X[:, 0] ** 2 - param[1] * X[:, 0] - param[2]
            title += f"Cut {i}: $y + {-param[0]:.2f} x^2 + {-param[1]:.2f} x + {-param[2]:.2f}$. "
        else:
            constraint_val = -1 * np.ones_like(X)
        feasible_idx = constraint_val <= 0
        X_infeas = X[~feasible_idx]
        contour = axes.tricontour(
            X[:, 0],
            X[:, 1],
            constraint_val,
            levels=np.linspace(0, constraint_val.max(), 5),
            colors=colors[i],
            linestyles="dashed",
        )
        hatches = axes.tricontourf(
            X[:, 0],
            X[:, 1],
            constraint_val,
            levels=np.linspace(0, constraint_val.max(), 5),
            colors="none",
            edgecolor=colors[i],
            hatches=hatch_styles[i],
        )
        X = X[feasible_idx]

    axes.scatter(
        X[:, 0],
        X[:, 1],
        alpha=0.5,
    )

    opt1 = EntropyOptimizer(X, 0)
    r, obj = opt1.optimize()
    axes.scatter(
        X[:, 0],
        X[:, 1],
        c="none",
        edgecolor="tab:red",
        marker="H",
        s=r * 500,
    )
    axes.set_xlim([-1.05, 1.05])
    axes.set_ylim([-1.05, 1.05])

    axes.set_xlabel("$x$")
    axes.set_ylabel("$y$")

    fig.suptitle(title)
    fig.tight_layout()

    plt.show()
    nice = input(
        "Is the experimental space nice? Type 'N' if no, any other answer is treated as "
        "yes and the program will save the figure. \n "
    )
    if nice != "N":
        fig.savefig(f"seed_{seed}_n_cuts_{n_cuts}.png", dpi=360)
