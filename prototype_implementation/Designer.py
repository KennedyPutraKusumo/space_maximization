from solvers.CvxpyEffortSolver import CvxpyEffortSolver
from plotters.BispacePlotter import BispacePlotter

class Designer:
    def __init__(self):
        self.criterion = None
        self.candidate_experiments = None
        self.n_run = None
        """
        Candidate experiments are expected to be a 2-element np.array containing:
        } The first entry being a 2D npointsXndim np.array of the input points
        } The second entry being a 2D npointsXndim np.array of the output points
        """

        self.solver = None
        self.plotter = None
        self.verbosity = 0

        self.package = "cvxpy"
        self.optimizer = "CPLEX"

    def initialize(self, verbose=0):
        if self.criterion is None:
            print("[REMINDER]: the criteri(a/on) is still not declared")
        if self.candidate_experiments is None:
            print("[REMINDER]: the candidate experiments is still not declared")
        if self.n_run is None:
            print("[REMINDER]: the number of runs is still not declared")
        self.verbosity = verbose

    def design_experiments(self, method):
        if "effort" in method:
            if self.verbosity >= 2:
                verbose = True
            else:
                verbose = False
            self.solver = CvxpyEffortSolver(
                points=self.candidate_experiments[1],
                criterion=self.criterion,
                n_runs=self.n_run,
                verbose=verbose,
            )
            self.solver.solve()

    def plot_results(self, title=None, in_labels=None, out_labels=None):
        if isinstance(self.criterion(), MaximalCovering):
            y = self.solver.problem.variables()[2].value
        elif isinstance(self.criterion(), MaximalSpread):
            y = self.solver.problem.variables()[1].value
        self.plotter = BispacePlotter(
            self.candidate_experiments[0],
            self.candidate_experiments[1],
            y,
            title=title + f". {self.criterion().name} Design, Objective: {self.solver.problem.objective.value:.3e} Number of Runs: {self.n_run}, Number of Candidates: {self.candidate_experiments[0].shape[0]}",
            in_labels=in_labels,
            out_labels=out_labels,
        )
        self.plotter.plot()

    def show_plots(self):
        self.plotter.show_plots()


if __name__ == '__main__':
    from criteria.MaximalSpread import MaximalSpread
    from criteria.MaximalCovering import MaximalCovering
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    import numpy as np

    grid_reso = 11j
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    designer1 = Designer()
    designer1.candidate_experiments = np.array([
        X,
        Y,
    ])
    criterion1 = MaximalCovering
    criterion2 = MaximalSpread
    criteria = [criterion1, criterion2]
    n_runs = [4, 8]
    for criterion in criteria:
        for n_run in n_runs:
            designer1.package = "cvxpy"
            designer1.optimizer = "CPLEX"
            designer1.criterion = criterion
            designer1.n_run = n_run
            designer1.initialize(verbose=2)
            designer1.design_experiments(
                method="effort",
            )
            designer1.plot_results(
                title="Michael Addition",
                in_labels=[
                    "Feed Ratio (AH/B)",
                    "Residence Time (min)",
                ],
                out_labels=[
                    "Conversion of Feed C (mol/mol)",
                    "Concentration of AC- (mol/L)",
                ],
            )
    designer1.show_plots()
