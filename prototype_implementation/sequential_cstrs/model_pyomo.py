import numpy as np
from pyomo import environ as po

if __name__ == '__main__':
    model = po.ConcreteModel()

    model.i = po.Set(initialize=["A", "B", "C", "D"])
    model.j = po.Set(initialize=[1, 2])
    model.k = po.Set(initialize=["Reactor 1", "Reactor 2"])

    model.q = po.Var(bounds=(0, None))
    model.C = po.Var(model.i, model.k, bounds=(0, None))
    model.C_in = po.Var(model.i, bounds=(0, None))

    # stoichiometric ratios
    nu = {
        # rxn 1
        ("A", 1): -1,
        ("D", 1): -1,
        ("B", 1): 1,
        # rxn 2
        ("B", 2): -1,
        ("C", 2): 1,
    }
    model.nu = po.Param(model.i, model.j, initialize=nu)
    
    model.k0 = po.Param(model.j)
    model.ea = po.Param(model.j)
    model.T = po.Var(model.k, bounds=(0, None))
    
    model.V = po.Var(model.k, bounds=(0, None))
    
    # model equations
    def _material_bal_reactor(m, i, k):
        r11 = m.k0[1] * po.exp(-m.ea[1]) / (m.T + 273) * m.C["A", "Reactor 1"] * m.C["D", "Reactor 1"]
        r21 = m.k0[2] * po.exp(-m.ea[2]) / (m.T + 273) * m.C["B", "Reactor 1"]
        rj1 = {
            1: r11,
            2: r21,
        }
        return 0 == m.q * m.C_in[i] - m.q * m.C[i, "Reactor 1"] + sum(m.nu[i, j] * rj1[j] * m.V["Reactor 1"] for j in m.j)
    model.mat_bal_r1 = po.Constraint(model.i, rule=_material_bal_reactor)
    
    def _material_bal_reactor_2(m, i, k):
        r12 = m.k0[1] * po.exp(-m.ea[1]) / (m.T + 273) * m.C["A", "Reactor 2"] * m.C["D", "Reactor 2"]
        r22 = m.k0[2] * po.exp(-m.ea[2]) / (m.T + 273) * m.C["B", "Reactor 2"]
        rj2 = {
            1: r12,
            2: r22,
        }
        return 0 == m.q * m.C[i, "Reactor 1"] - m.q * m.C[i, "Reactor 2"] + sum(m.nu[i, j] * rj2[j] * m.V["Reactor 2"] for j in m.j)
    model.mat_bal_r2 = po.Constraint(model.i, rule=_material_bal_reactor_2)

    def _dummy_obj(m):
        return m.V["Reactor 1"]
    dummy_obj = po.Objective(rule=_dummy_obj, sense=po.minimize)
    
    # fixing variables
    model.T.fix(22.0)
    model.V["Reactor 1"].fix(0.729)
    model.V["Reactor 2"].fix(2.000)
    model.q.fix(0.025)
    model.C_in["A", ]
    
    solver = po.SolverFactory("ipopt")
    solver.solve(model)
