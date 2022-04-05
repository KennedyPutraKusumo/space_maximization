from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np


def centroid_distances(points):
    points = points.reshape(int(points.size / 2), 2)
    centroid = np.mean(points, axis=0)
    dist = distance_matrix(points, centroid[None, :])
    return -np.sum(dist)


if __name__ == '__main__':
    samples_1 = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ])
    print(centroid_distances(samples_1))

    res_points_2 = np.array([
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
    ])
    print(centroid_distances(res_points_2))

    n_exp = 8
    x0 = np.random.uniform(
        low=[-1, -1],
        high=[1, 1],
        size=(n_exp, 2),
    )
    print(centroid_distances(x0))
    x1 = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
        [ 0,  0],
    ])
    print(centroid_distances(x1))
    opt_input = np.array([
        [16.5, 32.7],
        [18.7, 20.1],
        [83.4, 16.1],
        [80.1, 18.9],
    ])
    print(opt_input)
    print(centroid_distances(opt_input))
    opt_output = np.array([
        [50.1, 87.2],
        [11.9, 70.4],
        [18.7, 20.1],
        [83.6, 16.1],
    ])
    print(opt_output)
    print(centroid_distances(opt_output))
    opt_result = minimize(
        fun=centroid_distances,
        x0=x0,
        bounds=np.array([[-1,  1]] * x0.size),
        options={
            "disp": True,
        },
        method="l-bfgs-b",
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
