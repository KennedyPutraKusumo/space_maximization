from matplotlib import pyplot as plt
from michael_addition.model_scipy import simulate_cqa
from matplotlib import cm
from scipy.spatial import ConvexHull
import numpy as np
import pickle


if __name__ == '__main__':

    grid_points = False

    grid_reso = 41j

    R_grid, tau_grid = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    R_grid = R_grid.flatten()
    tau_grid = tau_grid.flatten()

    mp = {
        1: 49.7796,
        2: 8.9316,
        3: 1.3177,
        4: 0.3109,
        5: 3.8781,
    }
    cqa_list = []
    for R, tau in zip(R_grid, tau_grid):
        cqa, infodict, ier, mesg = simulate_cqa(d=[R, tau], p=mp, full_output=True)
        if ier != 1:
            print(f"Fsolve failed at R: {R:.10E} and tau: {tau:.10E}")
            cqa_list.append(np.full_like(cqa, fill_value=np.nan))
        else:
            cqa_list.append(cqa)
    cqa_list = np.array(cqa_list)

    # fig = plt.figure(figsize=(13, 5))
    fig = plt.figure(figsize=(8, 3))
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    cmap = np.linspace(0, 1, R_grid.shape[0])
    if grid_points:
        axes1.scatter(
            R_grid,
            tau_grid,
            c=cm.gist_rainbow(cmap),
            alpha=0.50,
        )
        axes2.scatter(
            cqa_list[:, 0],
            cqa_list[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=0.50,
        )
    axes1.set_xlabel("Feed Ratio (AH/B)")
    axes1.set_ylabel("Residence Time (min)")
    axes2.set_xlabel("Conversion of Feed C (mol/mol)")
    axes2.set_ylabel("Concentration of AC- (mol/L)")

    cqa_list_no_nan = cqa_list[~np.isnan(cqa_list).any(axis=1)]
    hull = ConvexHull(
        cqa_list_no_nan,
    )
    print(f"Feasible space convex hull is {hull.volume:.2E} in volume.")
    for simplex in hull.simplices:
        axes2.plot(
            cqa_list_no_nan[simplex, 0],
            cqa_list_no_nan[simplex, 1],
            c="k",
            ls="solid",
            linewidth=1,
            alpha=1.0,
            marker="None",
        )

    fig.tight_layout()

    with open("feasible_space.pkl", "wb") as file:
        pickle.dump(fig, file)

    plt.show()
