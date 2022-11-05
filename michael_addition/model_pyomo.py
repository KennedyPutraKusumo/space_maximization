from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np

def simulate(d):
    R = d[0]
    tau = d[1]

    model = po.ConcreteModel()

    model.tau = po.Var(bounds=(0, None))

    # set of chemical species (in order of appearance in Laky 2018 paper)
    model.i = po.Set(initialize=([
        "AH",
        "B",
        "A-",
        "BH+",
        "C",
        "AC-",
        "P",
    ]))
    # set of chemical reactions (in order of appearance in Laky 2018 paper)
    model.j = po.Set(initialize=([
        1,
        2,
        3,
        4,
        5,
    ]))

    model.c = po.Var(model.i, within=po.NonNegativeReals)
    model.c_in = po.Var(model.i)

    k = {
        1: 49.7796,
        2:  8.9316,
        3:  1.3177,
        4:  0.3109,
        5:  3.8781,
    }
    model.k = po.Param(model.j, initialize=k)

    nu = {
        ("AH", 1): -1,
        ("AH", 2):  0,
        ("AH", 3):  0,
        ("AH", 4): -1,
        ("AH", 5):  0,

        ("B", 1): -1,
        ("B", 2):  0,
        ("B", 3):  0,
        ("B", 4):  0,
        ("B", 5):  1,

        ("A-", 1):  1,
        ("A-", 2): -1,
        ("A-", 3):  1,
        ("A-", 4):  1,
        ("A-", 5):  0,

        ("BH+", 1):  1,
        ("BH+", 2):  0,
        ("BH+", 3):  0,
        ("BH+", 4):  0,
        ("BH+", 5): -1,

        ("C", 1):  0,
        ("C", 2): -1,
        ("C", 3):  1,
        ("C", 4):  0,
        ("C", 5):  0,

        ("AC-", 1):  0,
        ("AC-", 2):  1,
        ("AC-", 3): -1,
        ("AC-", 4): -1,
        ("AC-", 5): -1,

        ("P", 1):  0,
        ("P", 2):  0,
        ("P", 3):  0,
        ("P", 4):  1,
        ("P", 5):  1,
    }
    model.nu = po.Param(model.i, model.j, initialize=nu)

    def _mass_bal(m, i):
        r = {
            1: m.k[1] * m.c["AH"] * m.c["B"],
            2: m.k[2] * m.c["A-"] * m.c["C"],
            3: m.k[3] * m.c["AC-"],
            4: m.k[4] * m.c["AC-"] * m.c["AH"],
            5: m.k[5] * m.c["AC-"] * m.c["BH+"],
        }
        return m.c_in[i] - m.cqa[i] + m.tau * sum(m.nu[i, j] * r[j] for j in m.j) == 0
    model.mass_bal = po.Constraint(model.i, rule=_mass_bal)

    def _dummy_obj(m):
        return 0
    model.dummy_obj = po.Objective(rule=_dummy_obj, sense=po.maximize)

    model.tau.fix(tau)
    model.c_in["AH"].fix(0.3955)
    model.c_in["B"].fix(0.3955 / R)
    model.c_in["C"].fix(0.25)
    model.c_in["BH+"].fix(0)
    model.c_in["A-"].fix(0)
    model.c_in["AC-"].fix(0)
    model.c_in["P"].fix(0)

    solver = po.SolverFactory("ipopt")
    solver.design_experiments(model)

    return np.array([
        -po.value(model.c["C"] + model.c["AC-"] - 0.1 * model.c_in["C"]),
        0.002 - po.value(model.c["AC-"]),
    ])

if __name__ == '__main__':
    grid_reso = 41j

    R_grid, tau_grid = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    R_grid = R_grid.flatten()
    tau_grid = tau_grid.flatten()
    g_list = []
    for R, tau in zip(R_grid, tau_grid):
        g = simulate([R, tau])
        if np.all(g >= 0):
            feasibility = 1
        else:
            feasibility = -1
        d = [g[0], g[1], feasibility]
        g_list.append(d)
    g_list = np.array(g_list)

    fig = plt.figure()
    axes = fig.add_subplot(111)

    design_points = np.array([R_grid, tau_grid]).T
    feasible_points = design_points[g_list[:, 2] >= 0]
    infeasible_points = design_points[g_list[:, 2] < 0]

    axes.scatter(
        feasible_points[:, 0],
        feasible_points[:, 1],
        c="tab:red",
    )
    axes.scatter(
        infeasible_points[:, 0],
        infeasible_points[:, 1],
        c="tab:purple",
    )
    plt.show()
