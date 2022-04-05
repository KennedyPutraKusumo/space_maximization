from matplotlib import pyplot as plt
from matplotlib import cm
from oop_classes.convex_hull_plotter import draw_cvxhull
import numpy as np


def plot_feasible_space(
        inputs,
        outputs,
        axes,
        show_scatter=False,
        hull_color="tab:red",
        hull_ls=(1, (1, 1)),
):
    cmap_blueprint = np.linspace(0, 1, inputs.shape[0])
    color = cm.gist_rainbow(cmap_blueprint)

    in_axis = axes[0]
    out_axis = axes[1]
    if show_scatter:
        in_axis.scatter(
            inputs[:, 0],
            inputs[:, 1],
            c=color,
        )
        out_axis.scatter(
            outputs[:, 0],
            outputs[:, 1],
            c=color,
        )
    feas_hull_lines, feas_hull = draw_cvxhull(outputs, [out_axis], color=hull_color, ls=hull_ls)

    return feas_hull_lines, feas_hull


if __name__ == '__main__':
    from michael_addition.model_scipy import multvar_sim_cqa
    grid_reso = 11j
    in1, in2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    in1 = in1.flatten()
    in2 = in2.flatten()
    inputs = np.array([in1, in2]).T
    fig = plt.figure(figsize=(13, 5))
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    axes = [axes1, axes2]

    outputs = multvar_sim_cqa(
        inputs,
    )
    plot_feasible_space(
        inputs,
        outputs,
        axes,
    )
    fig.tight_layout()
    plt.show()
