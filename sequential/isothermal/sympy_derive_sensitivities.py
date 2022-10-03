from sympy import *
import numpy as np


ca, cb, cc, theta1, theta2, ca0, t = symbols("ca cb cc theta1 theta2 ca0 t")
init_printing(use_unicode=True)

""" Sensitivities of [A] """
ca = ca0 * exp(-theta1 * t)
dca_d1 = diff(ca, theta1)
dca_d2 = diff(ca, theta2)
print(dca_d1)
print(dca_d2)

""" Sensitivities of [B] """
cb = theta1 * ca0 / (theta2 - theta1) * (exp(-theta1 * t) - exp(-theta2 * t))
dcb_d1 = diff(cb, theta1)
dcb_d2 = diff(cb, theta2)
print(dcb_d1)
print(dcb_d2)

""" Sensitivities of [C] """
cc = ca0 - cb - ca
dcc_d1 = diff(cc, theta1)
dcc_d2 = diff(cc, theta2)
print(dcc_d1)
print(dcc_d2)
