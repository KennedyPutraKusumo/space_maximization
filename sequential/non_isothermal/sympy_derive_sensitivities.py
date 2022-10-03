from sympy import *
import numpy as np

ca, cb, cc, k1, e1, k2, e2, ca0, t, T, R = symbols("ca cb cc k1 e1 k2 e2 ca0 t T R")
init_printing(use_unicode=True)

response_name = ["[A]", "[B]", "[C]"]
mp_name = ["k1", "e1", "k2", "e2"]

replacements = {
    "exp": "np.exp",
    "k1": "theta[0]",
    "e1": "theta[1]",
    "k2": "theta[2]",
    "e2": "theta[3]",
}
replace = True

""" Sensitivities of [A] """
ca = ca0 * exp(-k1 * exp(-e1 / (R * T)) * t)
dca_d1 = str(diff(ca, k1))
dca_d2 = str(diff(ca, e1))
dca_d3 = str(diff(ca, k2))
dca_d4 = str(diff(ca, e2))

""" Sensitivities of [B] """
cb = k1 * exp(-e1 / (R * T)) * ca0 / (k2 * exp(-e2 / (R * T)) - k1 * exp(-e1 / (R * T))) * (exp(-k1 * exp(-e1 / (R * T)) * t) - exp(-k2 * exp(-e2 / (R * T)) * t))
dcb_d1 = str(diff(cb, k1))
dcb_d2 = str(diff(cb, e1))
dcb_d3 = str(diff(cb, k2))
dcb_d4 = str(diff(cb, e2))

""" Sensitivities of [C] """
cc = ca0 - cb - ca
dcc_d1 = str(diff(cc, k1))
dcc_d2 = str(diff(cc, e1))
dcc_d3 = str(diff(cc, k2))
dcc_d4 = str(diff(cc, e2))

sensitivities = [
    [
        dca_d1,
        dca_d2,
        dca_d3,
        dca_d4,
    ],
    [
        dcb_d1,
        dcb_d2,
        dcb_d3,
        dcb_d4,
    ],
    [
        dcc_d1,
        dcc_d2,
        dcc_d3,
        dcc_d4,
    ],
]
if replace:
    for key, value in replacements.items():
        for i, sens in enumerate(sensitivities):
            for j, s in enumerate(sens):
                sensitivities[i][j] = s.replace(key, value)

width = 100
for i, sens in enumerate(sensitivities):
    print(f"Sensitivitity of {response_name[i]}".center(width, "="))
    for j, s in enumerate(sens):
        print(f"w.r.t. {mp_name[j]}:")
        print(s)
        print("".center(width, "-"))
