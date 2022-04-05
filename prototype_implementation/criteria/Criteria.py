import numpy as np
from abc import ABC, abstractmethod

# modifier
# type
class CriterionType(ABC):
    def __init__(self):
        self.criterion_value = None
        return

    @abstractmethod
    def criterion_function(self):
        pass

    @abstractmethod
    def primal_function(self):
        pass


class SamplingAverageApproximation(CriterionType, ABC):
    def __init__(self):
        super().__init__()
        self.scr_mp = None
        self.scr_weights = None
        self.scr_criterion = None
        self.n_scr = None
        return

    @abstractmethod
    def criterion_function(self):
        self.scr_criterion = []
        for scr, mp in enumerate(self.scr_mp):
            self.scr_criterion.append(self.primal_function())
        pass


class PseudoBayesian(SamplingAverageApproximation, ABC):
    def __init__(self):
        super().__init__()

    def criterion_function(self):
        super().criterion_function()
        return np.sum(self.scr_criterion)


class CVaR(SamplingAverageApproximation, ABC):
    def __init__(self, beta, rounding=None):
        self.beta = beta
        self.rounding = rounding
        super().__init__()

    def criterion_function(self):
        super().criterion_function()
        scr_criterion_sorted = np.sort(self.scr_criterion)
        if self.rounding == "up":
            cvar_idx = np.ceil(self.n_scr * self.beta)
        else:
            cvar_idx = np.floor(self.n_scr * self.beta)
        return scr_criterion_sorted[cvar_idx]


class MaxiMin(SamplingAverageApproximation, ABC):
    def __init__(self):
        super().__init__()

    def criterion_function(self):
        super().criterion_function()
        return np.min(self.scr_criterion)
