from matplotlib import pyplot as plt
import numpy as np


def plot_pareto(
        pareto,
        figsize=None,
        annotate=None,
):
    pareto = np.asarray(pareto)
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    axes.scatter(
        pareto[:, 0],
        pareto[:, 1],
    )
    if annotate is not None:
        for anno in annotate:
            axes.annotate(
                text=anno[0],
                xy=anno[1],
            )
    axes.set_xlabel("Input Orthogonality")
    axes.set_ylabel("Output Orthogonality")

    fig.tight_layout()
    return fig, axes
