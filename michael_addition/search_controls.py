from model_scipy import multvar_sim_cqa
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


if __name__ == '__main__':
    n_trials = 4
    x0 = np.random.uniform(
        low=[10, 400],
        high=[30, 1400],
        size=(n_trials, 2),
    )
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
    )
    print(opt_result.x)
    print(opt_result.fun)
    x = opt_result.x.reshape(n_trials, 2)
    fig1 = plt.figure()
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
    plt.show()
