from model_scipy import ds_simulate
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle


if __name__ == '__main__':
    reso = 31j
    n_scr = 100
    d1, d2 = np.mgrid[10:30:reso, 400:1400:reso]
    d1 = d1.flatten()
    d2 = d2.flatten()
    d = np.array([d1, d2]).T
    mp = np.random.multivariate_normal(
        mean=[49.7796, 8.9316, 1.3177, 0.3109, 3.8781],
        cov=np.array([
            [1.005, -3.428e-4, -1.006e-3, 1.523e-3, 2.718e-3],
            [-3.428e-4, 0.412, -7.951e-4, -3.937e-3, 2.364e-3],
            [-1.006e-3, -7.951e-4, 3.224e-3, 1.466e-3, -2.4e-3],
            [-1.523e-3, -3.937e-3, 1.466e-3, 2.746e-3, -4.102e-3],
            [2.718e-3, 2.364e-3, -2.4e-3, -4.102e-3, 7.148e-3],
        ]),
        size=n_scr,
    )
    feas_probs = []
    for d_point in d:
        p_feas = 0
        for p in mp:
            g = ds_simulate(d_point, p)
            if np.all(g >= 0) :
                p_feas += 1
        feas_probs.append(p_feas)
    feas_probs = np.array(feas_probs)
    fig = plt.figure()
    axes = fig.add_subplot(111)

    axes.scatter(
        d[:, 0],
        d[:, 1],
        c=feas_probs,
    )
    fig.savefig(f"ma_prob_ds_{reso}x{reso}_grid_{n_scr}_scr.png", dpi=180)

    solution = pd.DataFrame({
        "d1": d[:, 0],
        "d2": d[:, 1],
        "feas_probs": feas_probs,
    })
    with open(f"ma_prob_ds_{reso}x{reso}_grid_{n_scr}_scr.pkl", "wb") as file:
        pickle.dump(solution, file)
    plt.show()
