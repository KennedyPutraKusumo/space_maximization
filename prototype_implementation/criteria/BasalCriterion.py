class BasalCriterion:
    def __init__(self):
        self.criterion_value = None
        self.model_parameter = None
        self.points = None
        self.N = None
        self.n_dim = None

    def determine_points_shape(self):
        self.N, self.n_dim = self.points.shape
        return self.N, self.n_dim


class DiscreteBasalCriterion(BasalCriterion):
    def __init__(self):
        super().__init__()
        self.n_runs = None


class MaximalSpreadCriterion(DiscreteBasalCriterion):
    def __init__(self):
        super().__init__()


class MaximalCoveringCriterion(DiscreteBasalCriterion):
    def __init__(self):
        super().__init__()
