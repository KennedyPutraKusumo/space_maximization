from model import g
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    T_lb = 273.15
    T_ub = 500

    tau_lb = 1
    tau_ub = 72

    markersize = 10
    reso = 48j
    alpha = 1

    cpp = np.mgrid[T_lb:T_ub:reso, tau_lb:tau_ub:reso]
    T = cpp[0]
    T = T.flatten()
    tau = cpp[1]
    tau = tau.flatten()
    ca0 = 1
    t = np.linspace(np.zeros_like(tau), tau, 11).T
    mp = np.array([0.3, 5300, 0.5, 10000])
    gs = []
    for T_ind, t_ind in zip(T, t):
        g_val = g(t_ind, ca0, T_ind, mp)
        gs.append(g_val)
    gs = np.array(gs)
    feasibility = np.all(gs > 0, axis=1)
    c = np.where(feasibility, "tab:red", "tab:purple")
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        tau,
        T,
        c=c,
        s=markersize,
        alpha=alpha,
        marker="s",
    )
    plt.show()

