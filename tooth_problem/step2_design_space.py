from step1_model import sim
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def g_constraint(ti_controls, theta):
    g1 = sim(ti_controls, theta) - (-1)
    g2 = 1 - sim(ti_controls, theta)
    g = np.array([g1, g2])
    return np.squeeze(g)


if __name__ == '__main__':
    tic_reso = 11j
    n_scr = 1000
    theta_mean = [5.0, 0.6, 1.0, -2.2, -3.0]
    theta_cov = 0.50 * np.identity(5)
    np.random.seed(123)
    n_bins = 11

    tic = np.mgrid[-1.5:1.5:tic_reso, -1.5:1.5:tic_reso]
    tic = np.array([tic[0].flatten(), tic[1].flatten()])
    theta = np.random.multivariate_normal(theta_mean, theta_cov, n_scr)
    ys = []
    gs = []
    for th in theta:
        y = sim(tic, th)
        ys.append(y)
        g = g_constraint(tic, th)
        gs.append(g)
    ys = np.squeeze(ys)
    gs = np.array(gs)

    fig = plt.figure(figsize=(13, 5))
    cmap = np.linspace(0, 1, tic.shape[1])
    c = cm.gist_rainbow(cmap)
    axes = fig.add_subplot(121)
    axes.scatter(
        tic[0],
        tic[1],
        c=c,
    )
    axes2 = fig.add_subplot(122)
    axes2.hist(
        ys[:, 0],
        bins=n_bins,
    )
    axes2.axvline(
        ymin=0,
        ymax=1,
        x=-1,
    )

    axes2.axvline(
        ymin=0,
        ymax=1,
        x=1,
    )
    fig.tight_layout()
    plt.show()
