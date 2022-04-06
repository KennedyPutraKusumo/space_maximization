from michael_addition.model_scipy import multvar_sim_cqa
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from matplotlib import cm
from oop_classes.point_normalizer import Normalizer
from time import time
import numpy as np
import cvxpy as cp
import sys

if __name__ == '__main__':

    M_list = [4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
    grid_reso_list = [5j, 6j, 7j, 8j, 9j, 10j, 11j, 21j]
    for M in M_list:
        for grid_reso in grid_reso_list:
            solvers_list = [
                "CPLEX",    # MIP
                "GUROBI",   # MIP
                "CBC",      # MIP
                "MOSEK",    # MICvxP (except MISDP with exponential cone)
                "SCIP",     # MINLP
                "GLPK_MI",
            ]
            solver = "GUROBI"
            demo = False

            x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
            x1 = x1.flatten()
            x2 = x2.flatten()
            X = np.array([x1, x2]).T
            Y = multvar_sim_cqa(X)

            if demo:
                Y = X
                sys.stdout = open(
                    f"demo_maximal_covering_{grid_reso}x{grid_reso}_{solver}_{M}_trials.txt", "w")
            else:
                sys.stdout = open(
                    f"maximal_covering_{grid_reso}x{grid_reso}_{solver}_{M}_trials.txt", "w")

            norm_1 = Normalizer(Y)
            Y = norm_1.normalize()

            pi_pj = cdist(
                Y,
                Y,
                metric="euclidean",
            )
            N = Y.shape[0]

            y = cp.Variable(
                shape=N,
                boolean=True,
            )
            z = cp.Variable(
                shape=(N, N),
                boolean=True,
            )
            eta = cp.Variable()
            obj = cp.Minimize(eta)
            ####################################
            # new formulation
            ####################################
            if True:
                cons = [
                    eta >= cp.sum(z[:, j] * pi_pj[:, j]) for j in range(N)
                ]
                cons += [
                    cp.sum(z[:, j], axis=0) == 1 for j in range(N)
                ]
                cons += [
                    y[i] >= z[i, j] for i in range(N) for j in range(N)
                ]
                cons += [
                    M >= cp.sum(y)
                ]
            ####################################
            # old formulation
            ####################################
            if False:
                cons = [
                    eta >= cp.sum(z[i, :] * pi_pj[i, :]) for i in range(N)
                ]
                cons += [
                    y[i] == cp.sum(z[i, :]) for i in range(N)
                ]
                cons += [
                    cp.sum(y) == M
                ]
            max_cover_prob = cp.Problem(
                objective=obj,
                constraints=cons,
            )
            start = time()
            max_cover_prob.solve(
                solver=solver,
                verbose=True,
            )
            print(f"Optimization took {time() - start:.2f} seconds.")

            fig1 = plt.figure(figsize=(13, 5))

            cmap = cm.gist_rainbow(np.linspace(0, 1, X.shape[0]))

            axes1 = fig1.add_subplot(121)
            axes1.scatter(
                X[:, 0],
                X[:, 1],
                c=cmap,
            )
            axes2 = fig1.add_subplot(122)
            axes2.scatter(
                Y[:, 0],
                Y[:, 1],
                c=cmap,
            )
            axes1.scatter(
                X[:, 0],
                X[:, 1],
                edgecolor="tab:red",
                facecolor="none",
                marker="H",
                s=500 * y.value,
            )
            axes2.scatter(
                Y[:, 0],
                Y[:, 1],
                edgecolor="tab:red",
                facecolor="none",
                marker="H",
                s=500 * y.value,
            )
            fig1.tight_layout()
            if demo:
                fig1.savefig(f"demo_maximal_covering_{grid_reso}x{grid_reso}_{solver}_{M}_trials.png", dpi=180)
            else:
                fig1.savefig(f"maximal_covering_{grid_reso}x{grid_reso}_{solver}_{M}_trials.png", dpi=180)
            sys.stdout.close()
    plt.show()
