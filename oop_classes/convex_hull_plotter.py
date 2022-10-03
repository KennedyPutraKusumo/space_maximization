from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np


def draw_cvxhull(points, axes, color="tab:red", ls=(1, (1, 1))):
    hull = ConvexHull(points)

    for axis in axes:
        for simplex in hull.simplices:
            hull_lines = axis.plot(
                points[simplex, 0],
                points[simplex, 1],
                c=color,
                ls=ls,
            )

    return hull_lines, hull
