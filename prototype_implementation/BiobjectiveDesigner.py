from abc import ABC, abstractmethod
import numpy as np


class BiObjectiveDesigner(ABC):
    def __init__(self):
        self.single_objective_solver = None
        pass

    @abstractmethod
    def solve(self):
        pass


class EpsilonConstraintSolver(BiObjectiveDesigner, ABC):
    def __init__(self, bound, n_epsilon):
        super().__init__()
        self.bound = bound
        self.n_epsilon = n_epsilon
        self.epsilons = np.linspace(bound[0], bound[1], self.n_epsilon)

    def solve(self):
        pass


class CvxpyEpsilonConstraintSolver(EpsilonConstraintSolver, ABC):
    def __init__(self, bound, n_epsilon):
        super().__init__(bound, n_epsilon)

    def solve(self):
        pass
