import numpy as np
from scipy.optimize import fsolve

import abc


class Model:
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_controls(self, u):
        raise NotImplementedError

    @abc.abstractmethod
    def set_fixed_parameters(self, fp):
        raise NotImplementedError

    @abc.abstractmethod
    def set_estimated_parameters(self, p):
        raise NotImplementedError

    @abc.abstractmethod
    def get_states_by_name(self, names):
        raise NotImplementedError

    @abc.abstractmethod
    def ss_eqns(self):
        raise NotImplementedError

    @abc.abstractmethod
    def dyn_eqns(self):
        raise NotImplementedError


class DesignScenario:
    def __init__(self, process):
        self.process = process
        self.scenario = {
            'steady': True,
            'u': {},
            'fp': {},
            'p': {}
        }
        self.guess = None

    @abc.abstractmethod
    def set_design_variables(self, d):
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(self, p):
        raise NotImplementedError

    @abc.abstractmethod
    def get_constraints(self, d, p):
        raise NotImplementedError


class Process:
    def __init__(self, model):
        self.model = model
        self.sim_sol = None

    def apply_scenario(self, scenario):
        self.set_controls(scenario['u'])
        self.set_fixed_parameters(scenario['fp'])
        self.set_estimated_parameters(scenario['p'])

    def set_controls(self, u):
        self.model.set_controls(u)

    def set_fixed_parameters(self, fp):
        self.model.set_fixed_parameters(fp)

    def set_estimated_parameters(self, p):
        self.model.set_estimated_parameters(p)

    def get_states_by_name(self, names):
        return self.model.get_states_by_name(names)

    def simulate(self, scenario, s0):
        self.apply_scenario(scenario)
        if scenario['steady'] is True:
            guess = s0
            self.sim_sol = fsolve(self.model.ss_eqns, guess)
        else:
            assert False, \
                "The simulation of dynamic models is not implemented yet."


class LakySimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.controls = {}
        self.fixed_parameters = {}
        self.estimated_parameters = {}
        self.state = {}

    def set_controls(self, u):
        for name, value in u.items():
            self.controls[name] = value

    def set_fixed_parameters(self, fp):
        for name, value in fp.items():
            self.fixed_parameters[name] = value

    def set_estimated_parameters(self, p):
        for name, value in p.items():
            self.estimated_parameters[name] = value

    def get_states_by_name(self, names):
        out = {}
        for name in names:
            out[name] = self.state[name]
        return out

    def ss_eqns(self, s):
        tau = self.controls['tau']
        cA_in = self.controls['cA_in']
        cB_in = self.controls['cB_in']
        cC_in = self.controls['cC_in']
        cD_in = self.controls['cD_in']
        cE_in = self.controls['cE_in']

        niu = self.fixed_parameters['niu']

        k1 = self.estimated_parameters['k1']
        k2 = self.estimated_parameters['k2']

        cA, cB, cC, cD, cE = s

        r1 = k1*cA*cB
        r2 = k2*cC

        dcA_dt = cA_in - cA + tau*(niu['r1']['A']*r1 +
                                  niu['r2']['A']*r2)

        dcB_dt = cB_in - cB + tau*(niu['r1']['B']*r1 +
                                  niu['r2']['B']*r2)

        dcC_dt = cC_in - cC + tau*(niu['r1']['C']*r1 +
                                  niu['r2']['C']*r2)

        dcD_dt = cD_in - cD + tau*(niu['r1']['D']*r1 +
                                  niu['r2']['D']*r2)

        dcE_dt = cE_in - cE + tau*(niu['r1']['E']*r1 +
                                  niu['r2']['E']*r2)
        dc_dt = np.array(
            [dcA_dt, dcB_dt, dcC_dt, dcD_dt, dcE_dt]
        )
        return dc_dt

    def dyn_eqns(self):
        pass


class LSMDesignScenario(DesignScenario):
    def __init__(self, process):
        super().__init__(process)
        self.scenario = {
            'steady': True,
            'u': {
                'tau': None,
                'cA_in': 0.53,
                'cB_in': None,
                'cC_in': 0.0,
                'cD_in': 0.0,
                'cE_in': 0.0
            },
            'fp': {
                'niu': {
                    'r1': {'A': -1, 'B': -1, 'C': 1, 'D': 0, 'E': 0},
                    'r2': {'A': 0, 'B': 0, 'C': -1, 'D': 1, 'E': 1}
                }
            },
            'p': {
                'k1': None,
                'k2': None
            }
        }
        self.guess = [0.53, 0.53, 0.0, 0.0, 0.0]

    def set_design_variables(self, d):
        r_ba, tau = d
        self.scenario['u']['tau'] = tau
        self.scenario['u']['cB_in'] = 0.53*r_ba

    def set_parameters(self, p):
        k1, k2 = p
        self.scenario['p']['k1'] = k1
        self.scenario['p']['k2'] = k2

    def get_constraints(self, d, p):
        self.set_design_variables(d)
        self.set_parameters(p)

        self.process.simulate(self.scenario, self.guess)

        ca, cb, cc, cd, ce = self.process.sim_sol
        ca0 = self.scenario['u']['cA_in']

        cqa1 = cd/(ca0 - ca)
        cqa2 = cd/(ca + cb + cc)

        g1 = cqa1 - 0.9
        g2 = cqa2 - 0.2
        constraints = [g1, g2]

        return constraints


def pydex_simulate_2(ti_controls, model_parameters):

    return

if __name__ == '__main__':
    a_lsm = LakySimpleModel()

    d1 = 4.0
    d2 = 350.0
    the_scenario = {
        'steady': True,
        'u': {
            'tau': d2,
            'cA_in': 0.53,
            'cB_in': 0.53 * d1,
            'cC_in': 0.0,
            'cD_in': 0.0,
            'cE_in': 0.0
        },
        'fp': {
            'niu': {
                'r1': {'A': -1, 'B': -1, 'C': 1, 'D': 0, 'E': 0},
                'r2': {'A': 0, 'B': 0, 'C': -1, 'D': 1, 'E': 1}
            }
        },
        'p': {
            'k1': 0.31051,
            'k2': 0.026650
        }
    }
    print(a_lsm.__dict__)

    a_lsm_process = Process(a_lsm)
    lsm_solution_guess = [0.53, 0.53 * d1, 0.0, 0.0, 0.0]
    a_lsm_process.simulate(the_scenario, lsm_solution_guess)
    solution = a_lsm_process.sim_sol
    print('c =\n', solution)
    print('dc_dt=\n', a_lsm.ss_eqns(solution))
