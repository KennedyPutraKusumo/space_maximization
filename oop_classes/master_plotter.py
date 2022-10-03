from matplotlib import pyplot as plt
from matplotlib import cm
from oop_classes.convex_hull_plotter import draw_cvxhull
from michael_addition.model_scipy import multvar_sim_cqa
from oop_classes.feasible_space_plotter import plot_feasible_space
import numpy as np


def plot_feasible_spaces(
        g_model,
        bounds,
        axes,
        grid_reso=11j,
        input_axes_labels=None,
        output_axes_labels=None,
        show_scatter=False,
        hull_color="k",
        hull_ls="solid",
):

    x_grid, y_grid = np.mgrid[bounds[0][0]:bounds[0][1]:grid_reso, bounds[1][0]:bounds[1][1]:grid_reso]
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    inputs = np.array([x_grid, y_grid]).T
    outputs = g_model(inputs)

    axis1 = axes[0]
    axis2 = axes[1]

    hull_line, hull = plot_feasible_space(
        inputs,
        outputs,
        axes,
        show_scatter=show_scatter,
        hull_color=hull_color,
        hull_ls=hull_ls,
    )
    if input_axes_labels is not None:
        axis1.set_xlabel(input_axes_labels[0])
        axis1.set_ylabel(input_axes_labels[1])

    if output_axes_labels is not None:
        axis2.set_xlabel(output_axes_labels[0])
        axis2.set_ylabel(output_axes_labels[1])

    return [axis1, axis2]

def plot_results(
        inputs,
        outputs,
        repetitions=None,
        cmap=None,
        marker_alpha=0.5,
        input_markersize=100,
        output_markersize=100,
        convex_hull_color="tab:red",
        convex_hull_ls=(1, (1, 1)),
        figsize=(13, 5),
        input_axes_labels=None,
        output_axes_labels=None,
        suptitle=None,
        axestitle=None,
        annotate=None,
        hexagon_size=10,
):
    if cmap is None:
        cmap_blueprint = np.linspace(0, 1, inputs.shape[0])
        cmap = cm.gist_rainbow(cmap_blueprint)
    if input_axes_labels is None:
        input_axes_labels = ["$x_1$", "$x_2$"]
    if output_axes_labels is None:
        output_axes_labels = ["$y_1$", "$y_2$"]

    fig = plt.figure(figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle)
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    if axestitle is not None:
        axes1.set_title(axestitle[0])
        axes2.set_title(axestitle[1])

    axes1.scatter(
        inputs[:, 0],
        inputs[:, 1],
        c=cmap,
        s=input_markersize,
        alpha=marker_alpha,
    )
    if repetitions is not None:
        axes1.scatter(
            inputs[:, 0],
            inputs[:, 1],
            c="none",
            s=input_markersize * hexagon_size * repetitions,
            marker="H",
            edgecolor=cmap,
        )
    axes1.set_xlabel(input_axes_labels[0])
    axes1.set_ylabel(input_axes_labels[1])

    axes2.scatter(
        outputs[:, 0],
        outputs[:, 1],
        c=cmap,
        s=output_markersize,
        alpha=marker_alpha,
    )
    if repetitions is not None:
        axes2.scatter(
            outputs[:, 0],
            outputs[:, 1],
            c="none",
            s=input_markersize * hexagon_size * repetitions,
            marker="H",
            edgecolor=cmap,
        )
    axes2.set_xlabel(output_axes_labels[0])
    axes2.set_ylabel(output_axes_labels[1])

    draw_cvxhull(
        outputs,
        [axes2],
        color=convex_hull_color,
        ls=convex_hull_ls,
    )

    fig.tight_layout()

    return fig, [axes1, axes2]

if __name__ == '__main__':
    inputs = np.array([
            [10, 400],
            [10, 1400],
            [30, 400],
            [30, 1400],
    ])
    output = multvar_sim_cqa(
        inputs,
    )
    fig, axes = plot_results(
        inputs,
        output,
        input_axes_labels=["Feed Ratio (AH/B)", "Residence Time (min)"],
        output_axes_labels=["Conversion of Feed C (mol/mol)", "Concentration of AC- (mol/L)"],
    )
    axes = plot_feasible_spaces(
        multvar_sim_cqa,
        bounds=[
            [10, 30],
            [400, 1400],
        ],
        axes=axes,
    )

    plt.show()
