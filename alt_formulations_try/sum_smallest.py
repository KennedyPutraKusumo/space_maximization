from oop_classes.logdet_optimizer import LogDetOptimizer
import numpy as np
import cvxpy as cp


grid_reso = 11j
x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
x1 = x1.flatten()
x2 = x2.flatten()
X = np.array([x1, x2]).T
input_opt = LogDetOptimizer(X, criterion=lambda x: cp.sum_smallest(x, 5))
r, optimal_input = input_opt.optimize()
print(r)
