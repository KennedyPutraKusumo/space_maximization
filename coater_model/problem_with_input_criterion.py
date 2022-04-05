import numpy as np


u_1 = np.array([
    [ 1, -1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1, -1, -1],
])
m1 = u_1.T @ u_1
det_m1 = np.linalg.det(m1)
print(det_m1)

u_2 = np.array([
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
])
m2 = u_2.T @ u_2
det_m2 = np.linalg.det(m2)
print(det_m2)
