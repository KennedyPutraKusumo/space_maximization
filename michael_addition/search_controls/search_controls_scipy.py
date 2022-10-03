from michael_addition.model_scipy import multvar_sim_cqa
from oop_classes.point_normalizer import Normalizer
from oop_classes.master_plotter import plot_results, plot_feasible_spaces
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from time import time
import logging
import sys
import numpy as np


def maximal_spread(x):
    x = x.reshape(int(x.size/2), 2)
    cqas = multvar_sim_cqa(x)
    pi_pj = cdist(cqas, cqas)
    spread = pi_pj[~np.eye(*pi_pj.shape, dtype=bool)].min()
    return -spread


def orthogonality(x):
    x = x.reshape(int(x.size/2), 2)
    cqas = multvar_sim_cqa(x)
    norm1 = Normalizer(cqas)
    cqas = norm1.normalize()
    # append column of 1
    cqas = np.append(np.ones((cqas.shape[0], 1)), cqas, axis=1)
    info_matrix = cqas @ cqas.T
    return -np.linalg.det(info_matrix)


if __name__ == '__main__':
    # n_trials_list = [3]
    n_trials_list = [4, 5, 6, 7, 8, 9, 10]
    n_seeds_list = [100]
    # n_seeds_list = [5, 10, 20, 50, 100]
    optimizer_list = ["l-bfgs-b"]
    # optimizer_list = ["l-bfgs-b", "nelder-mead", "tnc", "slsqp", "powell", "trust-constr"]
    for n_trials in n_trials_list:
        for n_seeds in n_seeds_list:
            for optimizer in optimizer_list:
                handler = logging.StreamHandler(stream=sys.stdout)
                handler.setLevel(logging.DEBUG)
                sys.stdout = open(f"search_controls_scipy_{optimizer}_{n_trials}_trials_{n_seeds}_sites.txt", "w")

                spread_results = []
                orthogonality_results = []
                # multistart optimization
                overall_spread_design_duration = 0
                overall_orthogonal_design_duration = 0
                for seed in range(n_seeds):
                    seed_tweaker = 621995
                    print(f"Running seed {seed_tweaker + seed}")
                    np.random.seed(seed_tweaker + seed)
                    x0 = np.random.uniform(
                        low=[10, 400],
                        high=[30, 1400],
                        size=(n_trials, 2),
                    )
                    # x0 = np.array([
                    #     [10,  400],
                    #     [10, 1400],
                    #     [30,  400],
                    #     [30, 1400],
                    # ])
                    spread_design_start = time()
                    # Maximal Spread Design
                    opt_result = minimize(
                        maximal_spread,
                        x0=x0,
                        method=optimizer,
                        bounds=[
                            (10, 30),
                            (400, 1400),
                        ] * n_trials,
                        options={
                            "disp": True,
                        },
                        tol=1e-8,
                    )
                    overall_spread_design_duration += time() - spread_design_start
                    print("Maximal Spread")
                    print(opt_result.x)
                    print(opt_result.fun)
                    spread_results.append([opt_result.x, -opt_result.fun])

                    # Maximal Orthogonality Design
                    orthogonal_design_start = time()
                    opt_result2 = minimize(
                        orthogonality,
                        x0=x0,
                        method=optimizer,
                        bounds=[
                                   (10, 30),
                                   (400, 1400),
                               ] * n_trials,
                        options={
                            "disp": True,
                        },
                        tol=1e-8,
                    )
                    overall_orthogonal_design_duration += time() - orthogonal_design_start
                    print("Maximal Orthogonality")
                    print(opt_result2.x)
                    print(opt_result2.fun)
                    orthogonality_results.append(
                        [opt_result2.x, -opt_result2.fun]
                    )
                spread_results = np.array(spread_results)
                best_spread_design_idx = np.argmax(spread_results[:, 1], axis=0)
                x = spread_results[best_spread_design_idx, 0].reshape(n_trials, 2)
                y = multvar_sim_cqa(x)
                fig1, axes1 = plot_results(
                    x,
                    y,
                    input_axes_labels=["Feed Ratio (AH/B)", "Residence Time (min)"],
                    output_axes_labels=["Conversion of Feed C (mol/mol)", "Concentration of AC- (mol/L)"],
                    suptitle=f"Maximal Spread Design: (i) {n_trials} Experiments (ii) multi-start sites: {n_seeds} (iii) solver: {optimizer} (iv) time: {overall_spread_design_duration:.2f} s",
                )
                axes1 = plot_feasible_spaces(
                    multvar_sim_cqa,
                    bounds=[
                        [10, 30],
                        [400, 1400],
                    ],
                    axes=axes1,
                )

                orthogonality_results = np.array(orthogonality_results)
                best_orthogonality_design_idx = np.argmax(orthogonality_results[:, 1], axis=0)
                print(f"Best orthogonal design has criterion value of {orthogonality_results[best_orthogonality_design_idx][1]}")
                x = orthogonality_results[best_orthogonality_design_idx, 0].reshape(n_trials, 2)
                y = multvar_sim_cqa(x)
                fig2, axes2 = plot_results(
                    x,
                    y,
                    input_axes_labels=["Feed Ratio (AH/B)", "Residence Time (min)"],
                    output_axes_labels=["Conversion of Feed C (mol/mol)", "Concentration of AC- (mol/L)"],
                    suptitle=f"Maximal Orthogonality Design: (i) {n_trials} Experiments (ii) multi-start sites: {n_seeds} (iii) solver: {optimizer} (iv) time: {overall_orthogonal_design_duration:.2f} s",
                )
                axes2 = plot_feasible_spaces(
                    multvar_sim_cqa,
                    bounds=[
                        [10, 30],
                        [400, 1400],
                    ],
                    axes=axes2,
                )
                print(f"{n_seeds} Multi-start optimization with {optimizer} for the Maximal Orthogonal Designs with {n_trials} Experiments took {overall_orthogonal_design_duration:.2f} s")
                print(f"{n_seeds} Multi-start optimization with {optimizer} for the Maximal Spread Designs with {n_trials} Experiments took {overall_spread_design_duration:.2f} s")
                fig1.savefig(f"search_controls_scipy_maximal_spread_{optimizer}_{n_trials}_trials_{n_seeds}_sites.png")
                fig2.savefig(f"search_controls_scipy_orthogonal_{optimizer}_{n_trials}_trials_{n_seeds}_sites.png")
                sys.stdout.close()
    plt.show()
