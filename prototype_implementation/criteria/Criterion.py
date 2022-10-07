from abc import abstractmethod, ABC


class Criterion:
    def __init__(self):
        self.criterion_value = None
        self.model_parameter = None
        self.points = None
        self.N = None
        self.n_dim = None
        self.name = None
        self.cvxpy_problem = None

    def determine_points_shape(self):
        self.N, self.n_dim = self.points.shape
        return self.N, self.n_dim

    @abstractmethod
    def construct_cvxpy_problem(self):
        pass


class DiscreteCriterion(Criterion, ABC):
    def __init__(self):
        super().__init__()
        self.n_runs = None
