from os import getcwd
from pyomo import environ as po
from duu import DUU
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

# non pyomo model
if False:
    class Model:
        def __init__(self):
            # reaction constants
            self.k1 = None
            self.k2 = None
            # reaction rates
            self.r1 = None
            self.r2 = None
            # outlet molar concentrations
            self.ca_out = None
            self.cb_out = None
            self.cc_out = None
            self.cd_out = None
            self.ce_out = None
            # residence time
            self.tau = None
            # inlet molar concentrations
            self.ca_in = None
            self.cb_in = None

        @staticmethod
        def rxn_rate_1(k1, ca, cb):
            return k1 * ca * cb

        @staticmethod
        def rxn_rate_2(k2, cc):
            return k2 * cc

        @staticmethod
        def kinetic_model_1(k1, ca, cb):
            return k1 * ca * cb

        @staticmethod
        def kinetic_model_2(k2, cc):
            return k2 * cc

        @staticmethod
        def mass_balance_1(ca_in, ca_out, tau, r1):
            return ca_in - ca_out - tau * r1

        @staticmethod
        def mass_balance_2(cb_in, cb_out, tau, r1):
            return cb_in - cb_out - tau * r1

        @staticmethod
        def mass_balance_3(cc_in, cc_out, tau, r1, r2):
            return cc_in - cc_out + tau * (r1 - r2)

        @staticmethod
        def mass_balance_4(cd_in, cd_out, tau, r2):
            return cd_in - cd_out + tau * r2

        @staticmethod
        def mass_balance_5(ce_in, ce_out, tau, r2):
            return ce_in - ce_out + tau * r2

# define pyomo model build function
if True:
    def build_model(set_species, set_reaction, stoic_info):
        model = po.ConcreteModel()

        # sets
        model.i = po.Set(initialize=set_species)  # species set
        model.j = po.Set(initialize=set_reaction)  # reaction set

        # known model parameters
        model.nu = po.Param(model.i, model.j, initialize=stoic_info)  # stoichiometry

        # uncertain model parameters
        model.k = po.Var(model.j)  # reaction constants

        # state variables
        model.r = po.Var(model.j)  # reaction rates
        model.c_in = po.Var(model.i)  # inlet molar concentrations
        model.c_out = po.Var(model.i)  # outlet molar concentrations
        model.tau = po.Var()  # residence time

        # feasibility indicators of interest
        model.g_1 = po.Var()  # feasibility indicator 1
        model.g_2 = po.Var()  # feasibility indicator 2

        def _mass_balance(m, i):
            return m.c_in[i] - m.c_out[i] + \
                   m.tau * sum(m.nu[i, j] * m.r[j] for j in m.j) == 0
        model.mass_balance = po.Constraint(model.i, rule=_mass_balance)

        def _rate_of_reaction_1(m):
            return m.r['r1'] == m.k['r1'] * m.c_out['a'] * m.c_out['b']
        model.rate_of_reaction_1 = po.Constraint(rule=_rate_of_reaction_1)

        def _rate_of_reaction_1(m):
            return m.r['r2'] == m.k['r2'] * m.c_out['c']
        model.rate_of_reaction_1 = po.Constraint(rule=_rate_of_reaction_1)

        def _feasibility_1(m):
            return - m.g_1 == 0.9 * m.c_in['a'] - 0.9 * m.c_out['a'] - m.c_out['d']
        model.g1 = po.Constraint(rule=_feasibility_1)

        def _feasibility_2(m):
            return - m.g_2 == m.c_out['a'] + m.c_out['b'] + m.c_out['c'] - 5 * m.c_out['d']
        model.g2 = po.Constraint(rule=_feasibility_2)

        return model

# declare the feasibility function
if True:
    def g(m, solver, d, p):

        # identifying problem scale
        num_of_d, dims_of_d = np.shape(d)
        num_of_p, dims_of_p = np.shape(p)

        # fixing variables for pyomo model and solving
        m.c_in['a'].fix(0.53)
        m.c_in['c'].fix(0)
        m.c_in['d'].fix(0)
        m.c_in['e'].fix(0)
        n_g1_list = []
        n_g2_list = []
        m_by_n_by_g_matrix = []
        evaluations = 0
        d_count = 0
        for n_d in range(num_of_d):
            d_count += 1
            p_count = 0
            # print('d_count' + str(d_count))
            m.c_in['b'].fix(0.53 * d[n_d, 0])
            m.tau.fix(d[n_d, 1])
            for n_p in range(num_of_p):
                p_count += 1
                # print('p_count' + str(p_count))
                m.k['r1'].fix(p[n_p, 0])
                m.k['r2'].fix(p[n_p, 1])
                results = solver.solve(m)
                evaluations += 1
                g1 = po.value(m.g_1)
                g2 = po.value(m.g_2)
                n_g1_list.append(g1)
                n_g2_list.append(g2)
                n_by_g_matrix = np.array([g1, g2]).transpose()
                m.k['r1'].unfix()
                m.k['r2'].unfix()
            m_by_n_by_g_matrix.append(n_by_g_matrix)
            m.c_in['b'].unfix()
            m.tau.unfix()
        print('Total evaluations: ' + str(evaluations))
        return m_by_n_by_g_matrix

spc = ['a', 'b', 'c', 'd', 'e']
rxn = ['r1', 'r2']
nu = {
    # reaction 1
    ('a', 'r1'): -1,
    ('b', 'r1'): -1,
    ('c', 'r1'): 1,
    ('d', 'r1'): 0,
    ('e', 'r1'): 0,
    # reaction 2
    ('a', 'r2'): 0,
    ('b', 'r2'): 0,
    ('c', 'r2'): -1,
    ('d', 'r2'): 1,
    ('e', 'r2'): 1
      }

my_model = build_model(set_species=spc, set_reaction=rxn, stoic_info=nu)
solver = po.SolverFactory('ipopt')


def g_alt(d, p):
    return g(my_model, solver, d, p)


with open('d_laky_case_1_pe' + '/' + 'results.pkl', 'rb') as file:
    pe_results = pickle.load(file)
my_p_samples = [sample for sample in pe_results["samples"] if sample["w"] > 0.0]

# activity form declaration
if True:
    an_activity_form = {
        "activity_type": "ds",

        "activity_settings": {
            "case_name": "d_laky_case_1",
            "case_path": getcwd(),
            "resume": False,
            "save_period": 10
        },

        "problem": {
            "goal": "discover",
            "ds_definition": 1,
            "constraints": g_alt,
            "parameters_samples": my_p_samples,
            "design_variables": [
                {"r_ab": [4, 6]},
                {"tau": [350, 550]}
            ]
        },

        "solver": {
            "name": "ds-ns",
            "settings": {
                "phi_resolution": 0,
                "stop_criteria": [
                    {"inside_fraction": 1.0}
                ],
                "debug_level": 0,
                "monitor_performance": False
            },
            "algorithms": {
                "sampling": {
                    "algorithm": "mc_sampling-ns_global",
                    "settings": {
                         "nlive": 10,
                         "nreplacements": 5,
                         "prng_seed": 1989,
                         "f0": 0.3,
                         "alpha": 0.2,
                         "stop_criteria": [
                             {"max_iterations": 100000}
                         ],
                         "debug_level": 0,
                         "monitor_performance": False
                     },
                    "algorithms": {
                        "replacement": {
                            "sampling": {
                                "algorithm": "suob-box"
                            }
                        }
                    }
                }
            }
        }
    }

cs_path = an_activity_form["activity_settings"]["case_path"]
cs_name = an_activity_form["activity_settings"]["case_name"]

if True:
    the_duu = DUU(an_activity_form)

    t0 = time.time()
    the_duu.solve()
    cpu_time = time.time() - t0
    print('CPU seconds', cpu_time)

    writer = open(cs_path + '/' + cs_name + 'cpu_time.txt', 'w')
    writer.write(cs_name + ' took ' + str(cpu_time) + ' seconds to complete.')
    writer.close()

if True:
    with open(cs_path + '/' + cs_name + '/' + 'results.pkl', 'rb') \
            as file:
        results = pickle.load(file)

    samples = results["samples"]
    inside_threshold = 0.20
    cat_1 = 0.40
    cat_2 = 0.55
    cat_3 = 0.65
    inside_samples_coords = np.empty((0, 2))
    cat_l_samples_coords = np.empty((0, 2))
    cat_2_samples_coords = np.empty((0, 2))
    cat_3_samples_coords = np.empty((0, 2))
    outside_samples_coords = np.empty((0, 2))
    for i, sample in enumerate(samples):
        if sample["phi"] < inside_threshold:
            outside_samples_coords = np.append(outside_samples_coords,
                                               [sample["c"]], axis=0)
        elif inside_threshold <= sample["phi"] < cat_1:
            cat_l_samples_coords = np.append(cat_l_samples_coords,
                                             [sample["c"]], axis=0)
        elif cat_1 <= sample["phi"] < cat_2:
            cat_2_samples_coords = np.append(cat_2_samples_coords,
                                             [sample["c"]], axis=0)
        elif cat_2 <= sample["phi"] < cat_3:
            cat_3_samples_coords = np.append(cat_3_samples_coords,
                                             [sample["c"]], axis=0)
        else:
            inside_samples_coords = np.append(inside_samples_coords,
                                              [sample["c"]], axis=0)

    fig1 = plt.figure(figsize=[6, 4.5])
    marker_size = 7.5

    # reaction time vs equivalent of catalyst
    axes1 = fig1.add_subplot(1, 1, 1)
    x = outside_samples_coords[:, 0]
    y = outside_samples_coords[:, 1]
    axes1.scatter(x, y, s=marker_size, c='blue', alpha=0.5,
                  label=r'$\alpha \leq $' + "{:.2f}".format(inside_threshold))

    x = cat_l_samples_coords[:, 0]
    y = cat_l_samples_coords[:, 1]
    axes1.scatter(x, y, s=marker_size, c='green', alpha=0.5,
                  label="{:.2f}".format(
                      inside_threshold) + '$ \leq \alpha \leq $' + "{:.2f}".format(
                      cat_1))

    x = cat_2_samples_coords[:, 0]
    y = cat_2_samples_coords[:, 1]
    axes1.scatter(x, y, s=marker_size, c='gold', alpha=0.5,
                  label="{:.2f}".format(
                      cat_1) + r'$ \leq \alpha \leq $' + "{:.2f}".format(cat_2))

    x = cat_3_samples_coords[:, 0]
    y = cat_3_samples_coords[:, 1]
    axes1.scatter(x, y, s=marker_size, c='orange', alpha=0.5,
                  label="{:.2f}".format(
                      cat_2) + r'$ \leq \alpha \leq $' + "{:.2f}".format(cat_3))

    x = inside_samples_coords[:, 0]
    y = inside_samples_coords[:, 1]
    axes1.scatter(x, y, s=marker_size, c='red', alpha=0.05,
                  label=r'$\alpha \geq $' + "{:.2f}".format(cat_3))

    axes1.set_xlabel('Batch Time (min)')
    axes1.set_ylabel('Temperature (K)')
    axes1.legend(framealpha=1)
    axes1.set_xlim([250, 350])
    axes1.set_ylim([250, 300])
    plt.show()
