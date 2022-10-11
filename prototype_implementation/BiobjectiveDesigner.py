from EffortDesigner import EffortDesigner
from plotters.BispacePlotter import BispacePlotter
from time import time
from matplotlib import pyplot as plt
import cvxpy as cp
import numpy as np


class BiObjectiveDesigner:
    def __init__(self):
        self.designers = None
        self.pareto_points = None
        self.n_epsilon_points = None
        self.n_runs = None
        self.input_points = None
        self.output_points = None

        self.method = "epsilon-constraint"
        self.status = 0
        self.verbosity = 0

        self.start_time = None
        self.solution_time = None
        self.end_time = None

        self.alpha_problem = None
        self.alpha_levels = None

        self.combined_problem_cvxpy = None
        self.combined_variables = None
        self.combined_constraints = None
        self.objective = None
        self.objective_designer = 0

        self.in_labels = None
        self.out_labels = None

    def initialize(self, verbose=0):
        self.verbosity = verbose
        if self.input_points is None:
            print("[REMINDER]: the input points is still not declared")
            self.status += 1
        if self.output_points is None:
            print("[REMINDER]: the output points is still not declared")
            self.status += 1
        if self.designers is None:
            print(
                f"[REMINDER]: the designers is still not declared"
            )
            self.status += 1
        if self.n_epsilon_points is None and self.method == "epsilon-constraint":
            print(
                f"[REMINDER]: the number of alpha points to generate is still not "
                f"declared"
            )
        if self.n_runs is None:
            print("[REMINDER]: the number of runs is still not declared")
            self.status += 1
        if self.status <= 0:
            self.start_time = time()
            print(f"".center(100, "+"))
            print(
                f"[{f'{time() - self.start_time:.2f}':>10} s]: Bi-objective designer initialized successfully. Overall start time recorded. Status: READY")
            if self.verbosity >= 1:
                for designer in self.designers:
                    designer.n_runs = self.n_runs
                    designer.input_points = self.input_points
                    designer.output_points = self.output_points
                    designer.initialize(verbose=self.verbosity)
            print(f"Individual single-objective designer are successfully initialized")
            print(f"".center(100, "+"))
        else:
            raise SyntaxError(
                f"[ERROR]: please ensure that all required components to run a design "
                f"has been declared correctly, terminating."
            )

    def design_experiments(self, plot=True, print_results=True):
        if self.verbosity >= 1:
            print(f"".center(100, "+"))
            print(
                f"Solving the end-points of the Pareto Frontier by solving two single-"
                f"objective optimization problems"
            )
        self.pareto_points = []
        if self.method == "epsilon-constraint":
            # eval the extreme pareto points
            print(f"Designer 1 is running...")
            print(f"".center(100, "+"))
            y0, obj0 = self.designers[0].design_experiments()
            if print_results:
                self.designers[0].print_results()
            print(f"Designer 2 is running...")
            print(f"".center(100, "+"))
            y1, obj1 = self.designers[1].design_experiments()
            if print_results:
                self.designers[1].print_results()
            obj01 = self.designers[1].solver.criterion.evaluate_criterion(y0)
            obj10 = self.designers[0].solver.criterion.evaluate_criterion(y1)
            if self.verbosity >= 1:
                print(
                    f"Single-objective Optimizations at the end-points are completed "
                    f"successfully, with the following criterion values:"
                )
                print(f"".center(100, "-"))
                print(
                    f"{self.designers[0].solver.criterion.name} Design of "
                    f"{self.designers[0].space_of_interest} Space:"
                )
                print(
                    f"{self.designers[0].space_of_interest} "
                    f"{self.designers[0].solver.criterion.name}: {obj0:.3f}"
                )
                print(
                    f"{self.designers[1].space_of_interest} "
                    f"{self.designers[1].solver.criterion.name}: {obj01:.3f}"
                )
                print(
                    f"{self.designers[1].solver.criterion.name} Design of "
                    f"{self.designers[1].space_of_interest} Space:"
                )
                print(
                    f"{self.designers[0].space_of_interest} "
                    f"{self.designers[0].solver.criterion.name}: {obj10:.3f}"
                )
                print(
                    f"{self.designers[1].space_of_interest} "
                    f"{self.designers[1].solver.criterion.name}: {obj1:.3f}"
                )
            if plot:
                plotter = BispacePlotter(
                    self.designers[0].input_points,
                    self.designers[0].output_points,
                    y0,
                    title=f"Mode {self.designers[0].solver.criterion.name} Design of {self.designers[0].space_of_interest} Space: Input {self.designers[0].solver.criterion.name} {obj0:.3f} Output {self.designers[1].solver.criterion.name} Criterion: {obj01:.3f} Number of Runs: {self.designers[0].n_runs}, Number of Candidates: {self.designers[0].input_points.shape[0]}",
                    in_labels=self.in_labels,
                    out_labels=self.out_labels,
                    marker_labels=self.designers[0].pd_df.index.values + 1,
                )
                plotter.plot()
                plotter = BispacePlotter(
                    self.designers[1].input_points,
                    self.designers[1].output_points,
                    y1,
                    title=f"Mode {self.designers[1].solver.criterion.name} Design of {self.designers[1].space_of_interest} Space: Input {self.designers[0].solver.criterion.name} {obj10:.3f} Output {self.designers[1].solver.criterion.name} Criterion: {obj1:.3f} Number of Runs: {self.designers[0].n_runs}, Number of Candidates: {self.designers[0].input_points.shape[0]}",
                    in_labels=self.in_labels,
                    out_labels=self.out_labels,
                    marker_labels=self.designers[0].pd_df.index.values + 1,
                )
                plotter.plot()
            # generating the epsilon constraints
            obj = [
                [obj0, obj01],
                [obj10, obj1],
            ]
            self.pareto_points = np.array(obj)
            self.alpha_levels = np.linspace(self.pareto_points[0, :], self.pareto_points[1, :], self.n_epsilon_points + 2)
            self.alpha_levels = self.alpha_levels[1:-1, :]

            # combining constraints from both problems
            self.combined_constraints = []
            self.combined_constraints += self.designers[0].solver.problem.constraints
            self.combined_constraints += self.designers[1].solver.problem.constraints
            # settings the two y variables from the designers to be equal to each other
            if self.designers[0].criterion().name == "Maximal Covering":
                y1var = self.designers[0].solver.problem.variables()[2]
            elif self.designers[0].criterion().name == "Maximal Spread":
                y1var = self.designers[0].solver.problem.variables()[1]
            else:
                raise SyntaxError(
                    f"Unrecognized criterion for designer 1: "
                    f"{self.designers[0].criterion().name}, terminating."
                )
            if self.designers[1].criterion().name == "Maximal Covering":
                y2var = self.designers[1].solver.problem.variables()[2]
            elif self.designers[1].criterion().name == "Maximal Spread":
                y2var = self.designers[1].solver.problem.variables()[1]
            else:
                raise SyntaxError(
                    f"Unrecognized criterion for designer 2: "
                    f"{self.designers[1].criterion().name}, terminating."
                )
            y_ids = [
                y1var,
                y2var,
            ]
            self.combined_constraints += [
                y2var == y1var
            ]
            self.objective = self.designers[self.objective_designer].solver.problem.objective
            for k, epsilon in enumerate(self.alpha_levels):
                if self.objective_designer == 0:
                    idx = 1
                elif self.objective_designer == 1:
                    idx = 0
                else:
                    print(
                        f"[WARNING]: unrecognized designer id that is to be used as "
                        f"objective in the epsilon-constraint solution; reverting to the "
                        f"default of '0' i.e., the first designer is used."
                    )
                    self.objective_designer = 0
                    idx = 1
                if self.designers[idx].solver.problem.objective.NAME == "minimize":
                    temp_cons = [
                        self.designers[idx].solver.problem.objective.expr <= epsilon[idx]
                    ]
                else:
                    temp_cons = [
                        self.designers[idx].solver.problem.objective.expr >= epsilon[idx]
                    ]
                temp_prob = cp.Problem(
                    self.objective,
                    self.combined_constraints + temp_cons,
                )
                print(f"".center(100, "+"))
                print(f"[{time() - self.start_time:.2f} s - Pareto Point {k + 1}/{self.n_epsilon_points}]".center(100))
                print(
                    f"Running optimization using designer {self.objective_designer + 1}'s "
                    f"criterion: {self.designers[self.objective_designer].criterion().name} "
                    f"while constraining \ndesigner {idx + 1}'s criterion: {self.designers[idx].criterion().name} "
                    f"to be at least as good as the given epsilon: {epsilon[idx]:.3e}"
                )
                print(f"".center(100, "+"))
                temp_prob.solve(
                    verbose=True,
                )
                if print_results:
                    self.print_results()
                par_obj1 = temp_prob.objective.value
                assert np.array_equal(y_ids[0].value, y_ids[1].value), "y from designer 1 and 2 don't match!"
                par_y = y_ids[1]
                par_obj2 = self.designers[idx].solver.criterion.evaluate_criterion(par_y.value)
                pareto_point = [par_obj1, par_obj2]
                print(f"".center(100, "-"))
                print(f"Computed objectives of Pareto point {k + 1}: {par_obj1:.3e}, {par_obj2:.3e}")
                if idx == 0:
                    pareto_point = [par_obj2, par_obj1]
                self.pareto_points = np.vstack((
                    self.pareto_points,
                    [pareto_point],
                ))
                if plot:
                    plotter = BispacePlotter(
                        self.designers[0].input_points,
                        self.designers[0].output_points,
                        par_y.value,
                        title=f"Pareto Point {k+1}/{self.n_epsilon_points} Input {self.designers[0].solver.criterion.name} Criterion: {par_obj1:.3f} Output {self.designers[1].solver.criterion.name} Criterion: {par_obj2:.3f} Number of Runs: {self.designers[0].n_runs}, Number of Candidates: {self.designers[0].input_points.shape[0]}",
                        in_labels=self.in_labels,
                        out_labels=self.out_labels,
                        marker_labels=self.designers[0].pd_df.index.values + 1,
                    )
                    plotter.plot()
            print(f"".center(100, "+"))
            print(f"Completed bi-objective optimization for {self.n_epsilon_points} Pareto points after {time() - self.start_time:.2f} seconds")
            print(f"".center(100, "+"))
        else:
            print(
                f"[WARNING]: unrecognized method {self.method}, reverting to default "
                f"'epsilon-constraint'."
            )

    def plot_pareto_frontier(self):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(
            self.pareto_points[:, 0],
            self.pareto_points[:, 1],
        )

        def _annotate_sense(designer):
            if designer.solver.problem.objective.NAME == "maximize":
                s = f"higher is better"
            else:
                s = f"lower is better"
            return s
        s0 = _annotate_sense(self.designers[0])
        s1 = _annotate_sense(self.designers[1])
        axes.set_xlabel(f"{self.designers[0].solver.criterion.name} ({s0})")
        axes.set_ylabel(f"{self.designers[1].solver.criterion.name} ({s1})")
        fig.tight_layout()
        return fig

    @staticmethod
    def show_plots():
        plt.show()

    def print_results(self):
        for designer in self.designers:
            designer.print_results()


if __name__ == '__main__':
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    from criteria.MaximalSpread import MaximalSpread
    from criteria.MaximalCovering import MaximalCovering
    import numpy as np

    grid_reso = 11j
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    designer1 = EffortDesigner()
    designer1.package = "cvxpy"
    designer1.optimizer = "GUROBI"
    designer1.criterion = MaximalSpread
    # designer1.criterion = MaximalCovering
    designer1.space_of_interest = "input"

    designer2 = EffortDesigner()
    designer2.package = "cvxpy"
    designer2.optimizer = "CPLEX"
    designer2.criterion = MaximalSpread
    # designer2.criterion = MaximalCovering
    designer2.space_of_interest = "output"

    biobj_designer1 = BiObjectiveDesigner()
    biobj_designer1.input_points = X
    biobj_designer1.output_points = Y
    biobj_designer1.designers = [designer1, designer2]
    biobj_designer1.in_labels = [
        "Feed Ratio (AH/B)",
        "Residence Time (min)",
    ]
    biobj_designer1.out_labels = [
        "Conversion of Feed C (mol/mol)",
        "Concentration of AC- (mol/L)",
    ]
    biobj_designer1.n_runs = 4
    biobj_designer1.n_epsilon_points = 10
    biobj_designer1.initialize(verbose=2)
    biobj_designer1.design_experiments()
    biobj_designer1.plot_pareto_frontier()
    biobj_designer1.show_plots()
