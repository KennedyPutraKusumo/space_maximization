from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np


def intra_distances(points):
    points = points.reshape(int(points.size / 2), 2)
    dist = distance_matrix(points, points)
    return -np.sum(dist)


if __name__ == '__main__':
    samples_1 = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ])
    print(intra_distances(samples_1))

    n_exp = 6
    x0 = np.random.uniform(
        low=[-1, -1],
        high=[1, 1],
        size=(n_exp, 2),
    )
    print(intra_distances(x0))
    x1 = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
        [ 0,  0],
    ])
    print(intra_distances(x1))
    opt_result = minimize(
        fun=intra_distances,
        x0=x0,
        bounds=np.array([[-1,  1]] * x0.size),
        options={
            "disp": True,
        },
        method="l-bfgs-b",
        # constraints=[
        #     {"type": "eq", }
        # ],
    )
    x_final = opt_result.x.reshape(n_exp, 2)

    print(x_final)
    print(opt_result.fun)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        x_final[:, 0],
        x_final[:, 1],
    )
    plt.show()
