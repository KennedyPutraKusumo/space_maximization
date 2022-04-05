from abc import ABC
from prototype_implementation.solvers.Solver import Solver


class EffortSolver(Solver, ABC):
    def __init__(self):
        super().__init__()
        self.efforts = None
        self.atomics = None
