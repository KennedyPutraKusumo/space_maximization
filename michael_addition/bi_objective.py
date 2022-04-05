from matplotlib import pyplot as plt
from matplotlib import cm
from maximum_input import input_orthogonality, compute_optimal_input
from maximum_output import output_coverage, compute_optimal_output
from michael_addition.model_scipy import multvar_sim_cqa
from time import time
from scipy.spatial import ConvexHull
from oop_classes.logger import Logger
import sys
import numpy as np
import pickle
import os
import datetime


def biobjective(n_exp, n_pareto_points, basis="input", optimizer="slsqp", random_seed=12345, plot=True, write=True):
    """  """

    """ Computing Optimal Input Design """
    start = time()
    print(f"".center(100, "="))
    print(f"[{f'{time() - start:.2f}':>10} s]: Computing the optimal input orthogonality design...")
    opt_input = compute_optimal_input(
        n_exp=n_exp,
        optimizer=optimizer,
    )
    opt_input_exp = opt_input.x.reshape((n_exp, 2))
    opt_input_output_obj = output_coverage(opt_input_exp)
    print(f"[{f'{time() - start:.2f}':>10} s]: Complete.")
    print(f"[{f'{time() - start:.2f}':>10} s]: Complete. Maximal Input Design:")
    print(f"[{'Input Orthogonality':>30}]: {-opt_input.fun}")
    print(f"[{'Output Coverage':>30}]: {opt_input_output_obj:.4E}")
    print(f"[{'Experimental Design':>30}]:")

    """ Computing Optimal Output Design """
    print(f"".center(100, "="))
    print(f"[{f'{time() - start:.2f}':>10} s]: Computing the optimal output coverage design...")
    opt_output = compute_optimal_output(
        n_exp=n_exp,
        optimizer="slsqp",
        random_seed=random_seed,
    )
    opt_output_exp = opt_output.x.reshape((n_exp, 2))
    opt_output_input_obj = input_orthogonality(opt_output_exp)
    print(f"[{f'{time() - start:.2f}':>10} s]: Complete. Maximal Output Design:")
    print(f"[{'Input Orthogonality':>30}]: {opt_output_input_obj}")
    print(f"[{'Output Coverage':>30}]: {-opt_output.fun:.4E}")
    print(f"[{'Experimental Design':>30}]:")

    """ Computing Pareto Points """
    print(f"".center(100, "="))
    print(f"[{f'{time() - start:.2f}':>10} s]: Checking validity of extreme points...")

    # check the unexpected: if optimal input design has higher output objective than optimal output design
    if -opt_output.fun <= opt_input_output_obj:
        print(
            f"[WARNING]: the output coverage of the optimal input design "
            f"({opt_input_output_obj}) is greater than the optimal output design "
            f"({-opt_output.fun})"
        )
    # check the unexpected: if optimal output design has higher input objective than optimal input design
    if -opt_input.fun <= opt_output_input_obj:
        print(
            f"[WARNIG]: the input orthogonality of the optimal output design "
            f"({opt_output_input_obj}) is greater than the optimal input design "
            f"({-opt_input.fun})"
        )

    if basis == "output":
        epsilon_values = np.linspace(opt_input_output_obj, -opt_output.fun, n_pareto_points)
    else:
        if basis != "input":
            print(
                f"The chosen basis {basis} is unrecognized. Defaulting to input basis "
                f"to generate the epsilon values."
            )
        epsilon_values = np.linspace(opt_output_input_obj, -opt_input.fun, n_pareto_points+2)
    epsilon_values = epsilon_values[1:-1]
    print(f"[{f'{time() - start:.2f}':>10} s]: Complete, generating epsilon values of {basis} objective values.")
    print(epsilon_values)

    pareto_results = []
    for i, eps in enumerate(epsilon_values):
        print(f"".center(100, "="))
        print(f"[{f'{time() - start:.2f}':>10} s]: Computing Pareto Point {i+1}...")
        if basis == "output":
            pareto_result = compute_optimal_input(
                n_exp=n_exp,
                optimizer=optimizer,
                constraints=[
                    {"type": "ineq", "fun": lambda x: output_coverage(x) - eps},
                ],
            )
            pareto_result_input_obj = -pareto_result.fun
            pareto_result_output_obj = output_coverage(pareto_result.x)
        else:
            pareto_result = compute_optimal_output(
                n_exp=n_exp,
                optimizer=optimizer,
                constraints=[
                    {"type": "ineq", "fun": lambda x: input_orthogonality(x) - eps},
                ],
            )
            pareto_result_input_obj = input_orthogonality(pareto_result.x)
            pareto_result_output_obj = -pareto_result.fun
        pareto_results.append({
            "input_objective": pareto_result_input_obj,
            "output_objective": pareto_result_output_obj,
            "experimental_design": pareto_result.x.reshape((n_exp, 2)),
        })
        print(f"[{f'{time() - start:.2f}':>10} s]: Complete, Pareto Point {i+1}:")
        print(f"[{'Input Orthogonality':>30}]: {pareto_result_input_obj}")
        print(f"[{'Output Coverage':>30}]: {pareto_result_output_obj:.4E}")
        print(f"[{'Experimental Design':>30}]:")
        print(f"{pareto_result.x.reshape((n_exp, 2))}")

    if plot:
        fig2 = plt.figure(figsize=(15, 5))
        axes3 = fig2.add_subplot(121)
        axes4 = fig2.add_subplot(122)
        axes3.set_title("Unnormalized Pareto Frontier")
        axes3.set_xlabel("Input Orthogonality")
        axes3.set_ylabel("Output Coverage")
        axes4.set_title("Normalized Pareto Frontier")
        axes4.set_xlabel("Input Orthogonality")
        axes4.set_ylabel("Output Coverage")
        # plotting extreme points
        axes3.scatter(
            -opt_input.fun,
            opt_input_output_obj,
            c="tab:blue",
            alpha=0.5,
        )
        axes4.scatter(
            -opt_input.fun / -opt_input.fun,
            opt_input_output_obj / -opt_output.fun,
            c="tab:blue",
            alpha=0.5,
        )
        axes3.scatter(
            opt_output_input_obj,
            -opt_output.fun,
            c="tab:blue",
            alpha=0.5,
        )
        axes4.scatter(
            opt_output_input_obj / -opt_input.fun,
            -opt_output.fun / -opt_output.fun,
            c="tab:blue",
            alpha=0.5,
        )
        """ Plotting the experimental designs """
        # the extreme points
        cmap = np.linspace(0, 1, n_exp)

        # Optimal Input Design
        with open("feasible_space.pkl", "rb") as file:
            fig1 = pickle.load(file)
            axes1, axes2 = fig1.get_axes()
        axes1.scatter(
            opt_input_exp[:, 0],
            opt_input_exp[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=0.5,
        )
        opt_input_output = multvar_sim_cqa(opt_input_exp)
        axes2.scatter(
            opt_input_output[:, 0],
            opt_input_output[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=0.5,
        )
        opt_input_hull = ConvexHull(opt_input_output)
        for simplex in opt_input_hull.simplices:
            axes2.plot(
                opt_input_output[simplex, 0],
                opt_input_output[simplex, 1],
                ls="--",
                c="tab:red",
                alpha=0.5,
            )

        # Optimal Output Design
        with open("feasible_space.pkl", "rb") as file:
            fig1 = pickle.load(file)
            axes1, axes2 = fig1.get_axes()
        axes1.scatter(
            opt_output_exp[:, 0],
            opt_output_exp[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=0.5,
        )
        opt_output_output = multvar_sim_cqa(opt_output_exp)
        axes2.scatter(
            opt_output_output[:, 0],
            opt_output_output[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=0.5,
        )
        opt_output_hull = ConvexHull(opt_output_output)
        for simplex in opt_output_hull.simplices:
            axes2.plot(
                opt_output_output[simplex, 0],
                opt_output_output[simplex, 1],
                ls="--",
                c="tab:red",
                alpha=0.5,
            )

        # the pareto points
        for i, pareto_point in enumerate(pareto_results):
            with open("feasible_space.pkl", "rb") as file:
                fig1 = pickle.load(file)
                axes1, axes2 = fig1.get_axes()

            fig1.suptitle(f"Pareto Point {i+1}")

            axes1.scatter(
                pareto_point["experimental_design"][:, 0],
                pareto_point["experimental_design"][:, 1],
                c=cm.gist_rainbow(cmap),
                alpha=0.5,
            )
            pareto_output = multvar_sim_cqa(pareto_point["experimental_design"])

            axes2.scatter(
                pareto_output[:, 0],
                pareto_output[:, 1],
                c=cm.gist_rainbow(cmap),
                alpha=0.5,
            )

            paretoHull = ConvexHull(pareto_output)
            for simplex in paretoHull.simplices:
                axes2.plot(
                    pareto_output[simplex, 0],
                    pareto_output[simplex, 1],
                    ls="--",
                    c="tab:red",
                    alpha=0.5,
                )

            axes3.scatter(
                pareto_point["input_objective"],
                pareto_point["output_objective"],
                c="tab:red",
                alpha=0.5,
            )
            axes4.scatter(
                pareto_point["input_objective"] / -opt_input.fun,
                pareto_point["output_objective"] / -opt_output.fun,
                c="tab:red",
                alpha=0.5,
            )
            fig1.tight_layout()
            fig1.savefig(f"results/{datetime.datetime.now().date()}/{datetime.datetime.now().strftime('%H_%M_%S')}_pareto_point_{i+1}.png", dpi=360)
        fig2.tight_layout()
        fig2.savefig(f"results/{datetime.datetime.now().date()}/{datetime.datetime.now().strftime('%H_%M_%S')}_pareto_frontiers.png", dpi=360)
        plt.show()
    print(f"".center(100, "="))

    return

if __name__ == '__main__':
    fp = f"results/{datetime.datetime.now().date()}/"
    if not os.path.exists(fp):
        os.makedirs(fp)
    log_file = fp + f"{datetime.datetime.now().strftime('%H_%M_%S')}_log.txt"
    sys.stdout = Logger(file_path=log_file)

    """ Output Basis Benchmark """
    if True:
        biobjective(
            n_exp=4,                    # benchmark: 4
            n_pareto_points=10,         # benchmark: 10
            basis="output",             # benchmark: input
            optimizer="slsqp",          # benchmark: slsqp
            random_seed=123,            # benchmark: 123
            plot=True,
        )
    """ Input Basis Benchmark """
    if False:
        biobjective(
            n_exp=4,                    # benchmark: 4
            n_pareto_points=10,         # benchmark: 10
            basis="input",              # benchmark: input
            optimizer="slsqp",          # benchmark: slsqp
            random_seed=123,            # benchmark: 123
            plot=True,
        )
    sys.stdout = sys.__stdout__
