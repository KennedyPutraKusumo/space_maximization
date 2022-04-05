from michael_addition.model_scipy import multvar_sim_cqa
from oop_classes.logdet_optimizer import LogDetOptimizer
from oop_classes.master_plotter import plot_results
from oop_classes.pareto_plotter import plot_pareto
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from time import time
import numpy as np


if __name__ == '__main__':
    start = time()

    grid_reso = 21j
    no_pareto_points = 10
    basis = "input"

    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    pareto_frontier = []
    pareto_frontier_2 = []
    annotate = []
    annotate_2 = []
    """ Optimal input """
    input_opt = LogDetOptimizer(X)
    r_input_opt, optimal_input_x = input_opt.optimize()
    sign, optimal_input_y = input_opt.logdet_criterion(Y, r_input_opt)
    pareto_frontier.append([optimal_input_x, optimal_input_y])

    optimal_only_idx = input_opt.fetch_optimal_points()
    annotate.append([
        "Opt-Input",
        [optimal_input_x, optimal_input_y],
    ])
    fig1, axes1 = plot_results(
        X[optimal_only_idx],
        Y[optimal_only_idx],
        repetitions=r_input_opt[optimal_only_idx],
        suptitle=f"Optimally Orthogonal Input Design",
        axestitle=[
            f"Input Orthogonality: {optimal_input_x:.2f}",
            f"Output Orthogonality: {optimal_input_y:.2f}",
        ],
    )

    # compute convex hull volume of output space
    opt_input_hull = ConvexHull(Y[optimal_only_idx])
    pareto_frontier_2.append([optimal_input_x, opt_input_hull.volume])
    annotate_2.append([
        "Opt-Input",
        [optimal_input_x, opt_input_hull.volume],
    ])
    fig1.savefig("orthogonal_input.png", dpi=360)

    """ Optimal output """
    output_opt = LogDetOptimizer(Y)
    r_output_opt, optimal_output_y = output_opt.optimize()
    sign, optimal_output_x = input_opt.logdet_criterion(X, r_output_opt)
    pareto_frontier.append([optimal_output_x, optimal_output_y])

    optimal_only_idx = output_opt.fetch_optimal_points()
    annotate.append([
        "Opt-Output",
        [optimal_output_x, optimal_output_y],
    ])
    fig2, axes2 = plot_results(
        X[optimal_only_idx],
        Y[optimal_only_idx],
        repetitions=r_output_opt[optimal_only_idx],
        suptitle=f"Optimally Orthogonal Output Design",
        axestitle=[
            f"Input Orthogonality: {optimal_output_x:.2f}",
            f"Output Orthogonality: {optimal_output_y:.2f}",
        ],
    )

    # compute convex hull volume of output space
    opt_output_hull = ConvexHull(Y[optimal_only_idx])
    pareto_frontier_2.append([optimal_output_x, opt_output_hull.volume])
    annotate_2.append([
        "Opt-Output",
        [optimal_output_x, opt_output_hull.volume],
    ])
    fig2.savefig("orthogonal_output.png", dpi=360)

    # OBJECTIVE: output criterion
    if basis == "input":
        epsilons = np.linspace(
            start=optimal_output_x,
            stop=optimal_input_x,
            num=no_pareto_points,
        )
        pareto_opt = LogDetOptimizer(Y)
        for i, epsilon in enumerate(epsilons):
            pareto_opt.points_2 = X
            pareto_opt.add_pareto_constraint(epsilon)
            r_pareto, pareto_y = pareto_opt.optimize()
            sign, pareto_x = pareto_opt.logdet_criterion(X, r_pareto)
            pareto_frontier.append([pareto_x, pareto_y])

            pareto_optimal_idx = pareto_opt.fetch_optimal_points()
            annotate.append([
                f"Pareto {i+1}",
                [pareto_x, pareto_y],
            ])
            fig_pareto, axes_pareto = plot_results(
                X[pareto_optimal_idx],
                Y[pareto_optimal_idx],
                repetitions=r_pareto[pareto_optimal_idx],
                suptitle=f"Pareto point {i+1}",
                axestitle=[
                    f"Input Orthogonality: {pareto_x:.2f}",
                    f"Output Orthogonality: {pareto_y:.2f}",
                ],
            )

            # compute convex hull volume of output space
            pareto_hull = ConvexHull(Y[pareto_optimal_idx])
            pareto_frontier_2.append([pareto_x, pareto_hull.volume])
            annotate_2.append([
                f"Pareto {i+1}",
                [pareto_x, pareto_hull.volume],
            ])
            fig_pareto.savefig(f"Pareto point {i+1}.png", dpi=360)
    # OBJECTIVE: input criterion
    else:
        pass
    frontier_fig, frontier_axes = plot_pareto(
        pareto_frontier,
        annotate=annotate,
    )
    frontier_fig.savefig("ma_pareto.png", dpi=360)

    frontier_2_fig, frontier_2_axes = plot_pareto(
        pareto_frontier_2,
        annotate=annotate_2,
    )
    frontier_2_fig.savefig("ma_pareto_2.png", dpi=360)

    print(f"Completing the runs took a total of {time() - start:.2f} seconds.")

    plt.show()
