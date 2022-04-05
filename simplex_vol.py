import numpy as np


def vol_simplex(vertices):
    n = vertices.shape[0]
    v0 = vertices[0]  # the (arbitrarily chosen) anchor
    v = vertices - v0
    v = v[1:]
    vol = 1 / np.math.factorial(n) * np.abs(np.linalg.det(v))
    return vol

if __name__ == '__main__':

    res_points_1 = np.array([
        [20, 20],
        [85, 17],
        [11.5, 70],
        [50, 87],
    ])
    vol = vol_simplex(res_points_1)
