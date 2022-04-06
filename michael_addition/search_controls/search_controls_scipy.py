from michael_addition.model_scipy import multvar_sim_cqa
from oop_classes.point_normalizer import Normalizer
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import numpy as np


def maximal_spread(x):
    x = x.reshape(int(x.size/2), 2)
    cqas = multvar_sim_cqa(x)
    pi_pj = cdist(cqas, cqas)
    spread = pi_pj[~np.eye(*pi_pj.shape, dtype=bool)].min()
    return -spread

def orthogonality(x):
    x = x.reshape(int(x.size/2), 2)
    cqas = multvar_sim_cqa(x)
    norm1 = Normalizer(cqas)
    cqas = norm1.normalize()
    # append column of 1
    cqas = np.append(np.ones((cqas.shape[0], 1)), cqas, axis=1)
    info_matrix = cqas @ cqas.T
    return -np.linalg.det(info_matrix)


if __name__ == '__main__':
    n_trials = 4
    n_seeds = 100
    spread_results = []
    orthogonality_results = []

    fig1 = plt.figure()
    fig2 = plt.figure()

    # multistart optimization
    for seed in range(n_seeds):
        print(f"Running seed {seed}")
        np.random.seed(seed)
        x0 = np.random.uniform(
            low=[10, 400],
            high=[30, 1400],
            size=(n_trials, 2),
        )
        # x0 = np.array([
        #     [10,  400],
        #     [10, 1400],
        #     [30,  400],
        #     [30, 1400],
        # ])
        # Maximal Spread Design
        opt_result = minimize(
            maximal_spread,
            x0=x0,
            method="l-bfgs-b",
            bounds=[
                (10, 30),
                (400, 1400),
            ] * n_trials,
            options={
                "disp": True,
            },
            tol=1e-8,
        )
        print("Maximal Spread")
        print(opt_result.x)
        print(opt_result.fun)
        spread_results.append([opt_result.x, -opt_result.fun])

        # Maximal Orthogonality Design
        opt_result2 = minimize(
            orthogonality,
            x0=x0,
            method="l-bfgs-b",
            bounds=[
                       (10, 30),
                       (400, 1400),
                   ] * n_trials,
            options={
                "disp": True,
            },
            tol=1e-8,
        )
        print("Maximal Orthogonality")
        print(opt_result2.x)
        print(opt_result2.fun)
        orthogonality_results.append(
            [opt_result2.x, -opt_result2.fun]
        )
    spread_results = np.array(spread_results)
    best_spread_design_idx = np.argmax(spread_results[:, 1], axis=0)
    x = opt_result.x.reshape(n_trials, 2)
    axes1 = fig1.add_subplot(121)
    axes1.scatter(
        x[:, 0],
        x[:, 1],
    )
    cqas = multvar_sim_cqa(x)
    axes2 = fig1.add_subplot(122)
    axes2.scatter(
        cqas[:, 0],
        cqas[:, 1],
    )

    x = opt_result2.x.reshape(n_trials, 2)
    axes3 = fig2.add_subplot(121)
    axes3.scatter(
        x[:, 0],
        x[:, 1],
    )
    cqas = multvar_sim_cqa(x)
    axes4 = fig2.add_subplot(122)
    axes4.scatter(
        cqas[:, 0],
        cqas[:, 1],
    )
    plt.show()
