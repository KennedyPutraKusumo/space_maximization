from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np


def input_orthogonality(exp, bounds=None):
    if bounds is None:
        bounds = np.array([
            [10, 30],
            [400, 1400],
        ]).T
    if exp.ndim == 1:
        exp = exp.reshape(int(exp.shape[0]/2), 2)
    n_exp = exp.shape[0]
    n_dim = exp.shape[1]

    scaled_exp = exp - bounds[0, :][None, :]
    bound_ranges = (bounds[1, :] - bounds[0, :])[None, :]
    scaled_exp = scaled_exp / bound_ranges
    scaled_exp *= 2
    scaled_exp -= 1

    U = scaled_exp
    U = np.append(np.ones(U.shape[0])[:, None], U, axis=1)
    M = U.T @ U
    in_obj = np.linalg.det(M)

    return in_obj

def compute_optimal_input(n_exp, optimizer="l-bfgs-b", constraints=None):
    np.random.seed(12345)

    opt_init_x = np.random.uniform(
        low=[10, 400],
        high=[30, 1400],
        size=(n_exp, 2),
    )
    opt_bound = np.array([
        [10, 30],
        [400, 1400],
    ] * n_exp)
    if constraints is None:
        opt_result = minimize(
            lambda x: -input_orthogonality(x),
            x0=opt_init_x,
            bounds=opt_bound,
            method=optimizer,
            options={
                "disp": False,
            },
        )
    else:
        opt_result = minimize(
            lambda x: -input_orthogonality(x),
            x0=opt_init_x,
            bounds=opt_bound,
            method=optimizer,
            options={
                "disp": False,
            },
            constraints=constraints,
        )
    return opt_result

if __name__ == '__main__':
    n_exp = 5
    opt_exp = compute_optimal_input(
        n_exp,
        optimizer="slsqp",
    )
    opt_exp = opt_exp.x.reshape(n_exp, 2)

    print(input_orthogonality(opt_exp.flatten()))

    fig = plt.figure(figsize=(8, 5))
    axes = fig.add_subplot(111)
    axes.scatter(
        opt_exp[:, 0],
        opt_exp[:, 1],
        c="tab:red",
        label="Optimal Experiments",
        marker="H",
    )
    axes.set_xlabel("Feed Ratio (AH/B)")
    axes.set_ylabel("Residence Time (min)")
    print(opt_exp)
    fig.tight_layout()
    plt.show()
