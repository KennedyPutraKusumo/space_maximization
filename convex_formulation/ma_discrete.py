from michael_addition.model_scipy import multvar_sim_cqa
from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp


def normalize_to_minone_one(M):
    M_away = M - np.min(M, axis=0)[None, :]
    delta_M = np.max(M, axis=0) - np.min(M, axis=0)
    M_normed = -1 + 2 * M_away / delta_M[None, :]
    return M_normed

def compute_optimal_exp(
        objective,
        n_exp,
        relax_efforts,
        grid_reso=11j,
        solver="MOSEK",
        plot=True,
):
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    X_norm = normalize_to_minone_one(X)
    X_calc = np.append(np.ones((X_norm.shape[0], 1)), X_norm, axis=1)
    N_s = X.shape[0]
    if relax_efforts:
        r = cp.Variable(N_s, integer=False)         # repetitions
    else:
        r = cp.Variable(N_s, integer=True)          # repetitions
    cons = [
        r >= 1e-8,
        r <= 1,
        cp.sum(r) <= n_exp,
    ]

    Y = multvar_sim_cqa(X)
    if objective == "input":
        obj = cp.Maximize(cp.log_det(X_calc.T @ cp.diag(r) @ X_calc))
        # obj = cp.Maximize(cp.trace(X.T @ cp.diag(r) @ X))
        # obj = cp.Maximize(cp.sum(X.T @ cp.diag(r) @ X))
        # obj = cp.Maximize(cp.harmonic_mean(X.T @ cp.diag(r) @ X))
        # obj = cp.Maximize(cp.lambda_min(X.T @ cp.diag(r) @ X))
    else:
        Y_norm = normalize_to_minone_one(Y)
        Y_calc = np.append(np.ones((Y_norm.shape[0], 1)), Y_norm, axis=1)
        obj = cp.Maximize(cp.log_det(Y_calc.T @ cp.diag(r) @ Y_calc))

    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)

    if plot:
        fig = plt.figure()
        axes1 = fig.add_subplot(121)
        axes2 = fig.add_subplot(122)
        axes1.scatter(
            x1,
            x2,
            alpha=r.value,
        )
        axes2.scatter(
            Y[:, 0],
            Y[:, 1],
            alpha=r.value,
        )
        plt.show()

if __name__ == '__main__':
    n_exp = 4
    relax_efforts = True
    grid_reso = 11j
    solver = "MOSEK"
    objective = "output"
    plot = True

    compute_optimal_exp(
        objective=objective,
        n_exp=n_exp,
        relax_efforts=relax_efforts,
        grid_reso=grid_reso,
        solver=solver,
        plot=plot,
    )
