from sympy.abc import x, y, i, C, p, V
from sympy import Sum, Indexed, IndexedBase
from sympy import init_printing, degree_list

n_points = 3

init_printing()
xi = Indexed(x, i)
yi = Indexed(y, i)
pi = Indexed(p, i)
centroid_x = Sum(pi * xi, (i, 1, n_points)) / Sum(pi, (i, 1, n_points))
centroid_y = Sum(pi * yi, (i, 1, n_points)) / Sum(pi, (i, 1, n_points))
vol = Sum(pi * ((xi - centroid_x) ** 2 + (yi - centroid_y) ** 2), (i, 1, n_points))
print(vol)
print(vol.doit())
print(vol.doit().expand())
print(degree_list(vol.doit().expand()))
