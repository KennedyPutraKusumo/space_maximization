class Designer:
    def __init__(self):
        self.criteria = None
        self.model_parameters = None
        self.candidate_experiments = None

    def initialize(self):
        pass

    def design_experiments(self):
        pass

if __name__ == '__main__':
    designer1 = Designer()
    campaign1 = Enumerate(
        bounds=[],
        levels=[],
        switching_times=[],
    )
    designer1.candidate_experiments = campaign1
    model_parameter1 = [
        49.7796,
        8.9316,
        1.3177,
        0.3109,
        3.8781,
    ]
    designer1.model_parameters = model_parameter1
    criterion1 = CVaR(D_optimal)
    criterion2 = PseudoBayesian(D_optimal)
    criterion3 = Maximin(MaximalSpread)
    designer1.criteria = [criterion1, criterion2]
    designer1.initialize()
    designer1.design_experiments()
