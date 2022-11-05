from prototype_implementation.solvers.CvxpyEffortSolver import CvxpyEffortSolver
from prototype_implementation.plotters.BispacePlotter import BispacePlotter
from prototype_implementation.criteria.MaximalCovering import MaximalCovering
from prototype_implementation.criteria.MaximalSpread import MaximalSpread
from time import time
import datetime
import numpy as np
import pandas as pd


class EffortDesigner:
    def __init__(self):
        self.criterion = None
        self.input_points = None
        self.output_points = None
        self.n_runs = None
        """
        Candidate experiments are expected to be a 2-element np.array containing:
        } The first entry being a 2D npointsXndim np.array of the input points
        } The second entry being a 2D npointsXndim np.array of the output points
        """

        self.solver = None
        self.plotter = None
        self.verbosity = 0
        self.status = 0

        self.npoints = None
        self.ndim = None

        self.start_time = None
        self.solution_time = None
        self.end_time = None

        self.package = "cvxpy"
        self.optimizer = "CPLEX"

        self.objective_value = None
        self.y = None
        self.pd_df = None
        self.opt_candidates = None

        self.space_of_interest = None

        self._fixed_y = None
        self.y_is_fixed = False

    @property
    def fixed_y(self):
        return self._fixed_y

    @fixed_y.setter
    def fixed_y(self, y):
        if y is None:
            self._fixed_y = None
            self.y_is_fixed = False
        else:
            self._fixed_y = y
            self.y_is_fixed = True

    def initialize(self, verbose=0):
        if self.criterion is None:
            print("[REMINDER]: the criteri(a/on) is still not declared")
            self.status += 1
        if self.input_points is None and "input" in self.space_of_interest:
            print("[REMINDER]: the input points is still not declared")
            self.status += 1
        if self.output_points is None and "output" in self.space_of_interest:
            print("[REMINDER]: the output points is still not declared")
            self.status += 1
        if self.n_runs is None:
            print("[REMINDER]: the number of runs is still not declared")
            self.status += 1
        if self.space_of_interest is None:
            print(f"[REMINDER]: the space of interest is still not declared")
            self.status += 1
        if "input" in self.space_of_interest:
            in_shape = self.input_points.shape
            self.npoints, self.indim = in_shape
        if "output" in self.space_of_interest:
            out_shape = self.output_points.shape
            self.npoints, self.outdim = out_shape
        if "input" in self.space_of_interest and "output" in self.space_of_interest:
            assert in_shape[0] == out_shape[0], \
                f"Number of points in the input points ({in_shape[0]}) is different from " \
                f"the number of points in the output points ({out_shape}), please check " \
                f"the provided input and output points."
        self.verbosity = verbose
        self.start_time = time()
        if self.status <= 0:
            if self.verbosity >= 0:
                print(f"".center(100, "="))
                print(f"[{f'{time() - self.start_time:.2f}':>10} s]: Initialization completed successfully. Start time recorded. Status: READY")
                if self.verbosity >= 1:
                    print(f"".center(100, "-"))
                    print(f"{'Number of points':<40}: {self.npoints} ")
                    print(f"{'Number of runs':<40}: {self.n_runs}")
                    print(f"{'Design Criterion':<40}: {self.criterion().name}")
                    print(f"{'Space of interest':<40}: {self.space_of_interest}")
                    if "input" in self.space_of_interest:
                        print(f"{'Number of input dimension':<40}: {self.indim} ")
                    if "output" in self.space_of_interest:
                        print(f"{'Number of output dimension':<40}: {self.outdim} ")
                    print(f"{'Package':<40}: {self.package}")
                    print(f"{'Optimizer':<40}: {self.optimizer}")
                print(f"".center(100, "="))
        else:
            raise SyntaxError(
                f"[ERROR]: please ensure that all required components to run a design "
                f"has been declared correctly, terminating."
            )

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
        self.get_optimal_candidates()
        return self.y, self.objective_value

    def _effort_design_solve(self):
        if self.verbosity >= 2:
            verbose = True
        else:
            verbose = False
        if self.package == "cvxpy":
            if self.space_of_interest == "input":
                points = self.input_points
            elif self.space_of_interest == "output":
                points = self.output_points
            else:
                print(
                    f"[WARNING]: unrecognized space of interest, reverting to default "
                    f"settings of 'output space'."
                )
                self.space_of_interest = "output"
                points = self.output_points
            self.solver = CvxpyEffortSolver(
                points=points,
                criterion=self.criterion,
                n_runs=self.n_runs,
                verbose=verbose,
                fix_y=self.fixed_y,
            )
            self.solver.optimizer = self.optimizer
            self.solver.solve()
            if self.criterion.__name__ == "MaximalCovering":
                self.y = self.solver.problem.variables()[2].value
            elif self.criterion.__name__ == "MaximalSpread":
                self.y = self.solver.problem.variables()[1].value
            self.objective_value = self.solver.problem.objective.value
        else:
            print(
                f"[WARNING]: unrecognized solution package: '{self.package}', reverting "
                f"to the default: 'cvxpy'."
            )
            self.package = "cvxpy"
            self._effort_design_solve()

    def plot_results(self, title=None, in_labels=None, out_labels=None, marker_labels=None):
        if self.pd_df is None:
            self.get_optimal_candidates()
        if marker_labels is None:
            marker_labels = self.pd_df.index
        if title is None:
            title = ""
        self.plotter = BispacePlotter(
            self.input_points,
            self.output_points,
            self.y,
            title=title + f". {self.criterion().name} Design in the {self.space_of_interest} Space, Objective: {self.objective_value:.3e} Number of Runs: {self.n_runs}, Number of Candidates: {self.input_points.shape[0]}",
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
        if "input" in self.space_of_interest:
            for i in range(self.indim):
                data.append(self.input_points[:, i])
                cols.append(f"Input {i+1}")
        if "output" in self.space_of_interest:
            for j in range(self.outdim):
                data.append(self.output_points[:, j])
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
        print(f"{'Space of interest':<40}: {self.space_of_interest}")
        print(f"{'Package':<40}: {self.package}")
        print(f"{'Optimizer':<40}: {self.optimizer}")
        print(f"{'Number of candidate points':<40}: {self.npoints}")
        if "input" in self.space_of_interest:
            print(f"{'Number of input dimensions':<40}: {self.indim}")
        if "output" in self.space_of_interest:
            print(f"{'Number of output dimensions':<40}: {self.outdim}")
        print(f"{'Number of runs':<40}: {self.n_runs}")
        for i, (j, candidate) in enumerate(self.opt_candidates.iterrows()):
            print(f"[Candidate {j+1} - (Run {i+1}/{self.n_runs} Runs)]".center(100, "-"))
            print(f"Input Coordinates:")
            print(candidate.filter(regex="Input").to_string(index=True))
            print(f"".center(100, "."))
            print(f"Output Coordinates:")
            print(candidate.filter(regex="Output").to_string(index=True))

        print(f"".center(100, "="))


if __name__ == '__main__':
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    import numpy as np

    grid_reso = 5j
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    designer1 = EffortDesigner()
    designer1.input_points = X
    designer1.output_points = Y
    designer1.package = "cvxpy"
    designer1.optimizer = "GUROBI"
    criterion1 = MaximalCovering
    criterion2 = MaximalSpread
    criteria = [criterion1, criterion2]
    n_runs = [4]
    spaces_of_interest = ["input", "output"]
    for criterion in criteria:
        for n_run in n_runs:
            for space in spaces_of_interest:
                designer1.fixed_y = np.zeros(X.shape[0])
                designer1.fixed_y[5] = 1
                designer1.fixed_y[11] = 1
                designer1.fixed_y[14] = 1
                designer1.fixed_y[15] = 1
                designer1.criterion = criterion
                designer1.n_runs = n_run
                designer1.space_of_interest = space
                designer1.initialize(verbose=2)
                designer1.design_experiments()
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
