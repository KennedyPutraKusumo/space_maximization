from michael_addition.model_scipy import simulate_g
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    grid_reso = 41j

    R_grid, tau_grid = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    R_grid = R_grid.flatten()
    tau_grid = tau_grid.flatten()

    g_list = []
    for R, tau in zip(R_grid, tau_grid):
        g_res = simulate_g(d=[R, tau])
        if np.all(g_res <= 0):
            feasibility = 1
        else:
            feasibility = -1
        g_results = [g_res[0], g_res[1], feasibility]
        g_list.append(g_results)
    g_list = np.array(g_list)

    fig = plt.figure()
    axes = fig.add_subplot(111)

    design_points = np.array([R_grid, tau_grid]).T
    feasible_points = design_points[g_list[:, 2] >= 0]
    infeasible_points = design_points[g_list[:, 2] < 0]

    axes.scatter(
        feasible_points[:, 0],
        feasible_points[:, 1],
        c="tab:red",
        alpha=0.5,
        label="Feasible Points",
    )
    axes.scatter(
        infeasible_points[:, 0],
        infeasible_points[:, 1],
        c="tab:blue",
        alpha=0.5,
        label="Infeasible Points",
    )
    axes.set_xlabel("Feed Ratio (AH/B)")
    axes.set_ylabel("Residence Time (min)")
    axes.legend()
    plt.show()
