from pyomo import environ as po
from oop_classes.point_normalizer import Normalizer
import numpy as np


def create_model(n_trials):
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

    model.trials = po.Set(
        initialize=np.arange(0, n_trials)
    )

    model.c = po.Var(model.trials, model.i, within=po.NonNegativeReals)
    model.c_in = po.Var(model.trials, model.i)

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

    def _mass_bal(m, trials, i):
        r = {
            1: m.k[1] * m.c[trials, "AH"] * m.c[trials, "B"],
            2: m.k[2] * m.c[trials, "A-"] * m.c[trials, "C"],
            3: m.k[3] * m.c[trials, "AC-"],
            4: m.k[4] * m.c[trials, "AC-"] * m.c[trials, "AH"],
            5: m.k[5] * m.c[trials, "AC-"] * m.c[trials, "BH+"],
        }
        return m.c_in[i] - m.cqa[i] + m.tau * sum(m.nu[i, j] * r[j] for j in m.j) == 0
    model.mass_bal = po.Constraint(model.trials, model.i, rule=_mass_bal)

    return model

def maximal_spread(x):
    n_trials = int(x.size/2)
    x = x.reshape((n_trials, 2))

    model = create_model()

    model.j = po.Set(np.arange(0, n_trials))

    model.c_in["AH"].fix(0.3955)
    model.c_in["C"].fix(0.25)
    model.c_in["BH+"].fix(0)
    model.c_in["A-"].fix(0)
    model.c_in["AC-"].fix(0)
    model.c_in["P"].fix(0)

    cqa1 = po.Var(model.i, model.i)
    cqa2 = po.Var(model.i, model.i)

    eta = po.Var()

    def _cqa1_def(m):
        return m.cqa1 == m.c["C"] + m.c["AC-"] - 0.1 * m.c_in["C"]
    cqa1_def = po.Constraint(rule=_cqa1_def)

    def _cqa2_def(m):
        return m.cqa2 == 0.002 - po.value(model.c["AC-"])
    cqa2_def = po.Constraint(rule=_cqa2_def)

    def _minimal_distance_def(m, i, j):
        if i >= j:
            return m.eta <= (m.cqa1[i, j] - m.cqa2[i, j]) ** 2
        else:
            po.Constraint.Skip()
    minimal_distance_def = po.Constraint(model.j, model.j, rule=_minimal_distance_def)

    for run in x:
        tau = x[0]
        R = x[1]
        model.tau.fix(tau)
        model.c_in["B"].fix(0.3955 / R)

    return
