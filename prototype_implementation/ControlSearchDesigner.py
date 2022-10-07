from solvers.ScipySolver import ScipySolver
from plotters.BispacePlotter import BispacePlotter
from criteria.Orthogonality import Orthogonality
from time import time
import datetime
import numpy as np
import pandas as pd


class EffortDesigner:
    def __init__(self):
        self.in_bounds = None
        self.out_bounds = None
        self.criterion = None
        self.n_run = None
        """
        Candidate experiments are expected to be a 2-element np.array containing:
        } The first entry being a 2D npointsXndim np.array of the input points
        } The second entry being a 2D npointsXndim np.array of the output points
        """

        self.solver = None
        self.opt_result = None
        self.plotter = None
        self.verbosity = 0
        self.status = 0

        self.indim = None
        self.outdim = None

        self.start_time = None
        self.solution_time = None
        self.end_time = None

        self.package = "scipy"
        self.optimizer = "SLSQP"

        self.objective_value = None

    def initialize(self, verbose=0):
        if self.criterion is None:
            print("[REMINDER]: the criteri(a/on) is still not declared")
            self.status += 1
        if self.n_run is None:
            print("[REMINDER]: the number of runs is still not declared")
            self.status += 1
        self.indim = self.in_bounds.shape[0]
        self.outdim = self.out_bounds.shape[0]
        self.verbosity = verbose
        self.start_time = time()
        if self.status <= 0:
            if self.verbosity >= 0:
                print(f"".center(100, "="))
                print(f"[{f'{time() - self.start_time:.2f}':>10} s]: Initialization completed successfully. Start time recorded. Status: READY")
                if self.verbosity >= 1:
                    print(f"".center(100, "-"))
                    print(f"{'Number of runs':<40}: {self.n_run}")
                    print(f"{'Design Criterion':<40}: {self.criterion().name}")
                    print(f"{'Number of input dimension':<40}: {self.indim} ")
                    print(f"{'Number of output dimension':<40}: {self.outdim} ")
                    print(f"{'Package':<40}: {self.package}")
                    print(f"{'Optimizer':<40}: {self.optimizer}")
                print(f"".center(100, "="))

    def design_experiments(self):
        if self.verbosity >= 0:
            print(f"".center(100, "="))
            print(
                f"[{f'{time() - self.start_time:.2f}':>10} s] Computing the "
                f"{self.criterion().name} design using {self.optimizer} interfaced via "
                f"{self.package}"
            )
            print(f"".center(100, "-"))
        self._effort_design_solve()
        if self.verbosity >= 0:
            if self.verbosity >= 1:
                print(f"".center(100, "-"))
                print(f"[{f'{time() - self.start_time:.2f}':>10} s] Completed after {time() - self.start_time:.2f} s, end time recorded.")
            print(f"".center(100, "="))
        self.end_time = time()

    def _effort_design_solve(self):
        if self.verbosity >= 2:
            verbose = True
        else:
            verbose = False
        if self.package == "scipy":
            self.solver = ScipySolver(
                bounds=self.in_bounds,
                criterion=self.criterion,
                n_runs=self.n_run,
                verbose=verbose,
            )
            self.solver.optimizer = self.optimizer
            self.opt_result = self.solver.solve()
            self.objective_value = self.opt_result.fun
            self.X = self.opt_result.x
        else:
            print(f"[WARNING]: unrecognized solution package: '{self.package}', reverting "
                  f"to the default: 'cvxpy'.")
            self.package = "cvxpy"
            self._effort_design_solve()

    def plot_results(self, title=None, in_labels=None, out_labels=None, marker_labels=None):
        if self.pd_df is None:
            self.get_optimal_candidates()
        if marker_labels is None:
            marker_labels = self.pd_df.index
        self.plotter = BispacePlotter(
            self.candidate_experiments[0],
            self.candidate_experiments[1],
            self.y,
            title=title + f". {self.criterion().name} Design, Objective: {self.objective_value:.3e} Number of Runs: {self.n_run}, Number of Candidates: {self.candidate_experiments[0].shape[0]}",
            in_labels=in_labels,
            out_labels=out_labels,
            marker_labels=marker_labels.values + 1,
        )
        self.plotter.plot()

    def show_plots(self):
        self.plotter.show_plots()

    def get_optimal_candidates(self):
        if self.y is None:
            raise SyntaxError(f"[WARNING]: please solve an experimental design problem first!")
        data = []
        cols = []
        for i in range(self.indim):
            data.append(self.candidate_experiments[0][:, i])
            cols.append(f"Input {i+1}")
        for j in range(self.outdim):
            data.append(self.candidate_experiments[1][:, j])
            cols.append(f"Output {j+1}")
        data.append(self.y)
        cols.append("Repetitions")
        data = np.array(data).T
        self.pd_df = pd.DataFrame(
            data=data,
            columns=cols,
        )
        self.opt_candidates = self.pd_df.query("Repetitions >= 1")

    def print_results(self):
        if self.pd_df is None:
            self.get_optimal_candidates()
        print(f"[{self.criterion().name} Design]".center(100, "="))
        print(f"{'Obtained on':<40}: {datetime.datetime.utcfromtimestamp(self.end_time)}")
        print(f"{'Criterion Value':<40}: {self.objective_value}")
        print(f"{'Package':<40}: {self.package}")
        print(f"{'Optimizer':<40}: {self.optimizer}")
        print(f"{'Number of candidate points':<40}: {self.npoints}")
        print(f"{'Number of input dimensions':<40}: {self.indim}")
        print(f"{'Number of output dimensions':<40}: {self.outdim}")
        print(f"{'Number of runs':<40}: {self.n_run}")
        for i, (j, candidate) in enumerate(self.opt_candidates.iterrows()):
            print(f"[Candidate {j+1} - (Run {i+1}/{self.n_run} Runs)]".center(100, "-"))
            print(f"Input Coordinates:")
            print(candidate.filter(regex="Input").to_string(index=True))
            print(f"".center(100, "."))
            print(f"Output Coordinates:")
            print(candidate.filter(regex="Output").to_string(index=True))

        print(f"".center(100, "="))


if __name__ == '__main__':
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    import numpy as np

    grid_reso = 11j
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    designer1 = EffortDesigner()
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
            designer1.optimizer = "GUROBI"
            designer1.criterion = criterion
            designer1.n_run = n_run
            designer1.initialize(verbose=2)
            designer1.design_experiments(
                method="effort",
            )
            designer1.get_optimal_candidates()
            designer1.print_results()
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
