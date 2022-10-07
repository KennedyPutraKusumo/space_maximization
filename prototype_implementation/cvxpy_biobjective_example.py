def coverage(points, y, metric="euclidean"):
    coverage = -np.infty
    for point in points[y < 1]:
        distance_to_centroids = cdist([point], points[y > 0], metric=metric)
        distance_to_closest_centroid = np.min(distance_to_centroids)
        coverage = np.max([coverage, distance_to_closest_centroid])
    return coverage

def spread(points, y, metric="euclidean"):
    spread = np.infty
    for point in points[y > 0]:
        distance_between_centroids = cdist([point], points[y > 0], metric=metric)
        spread = np.min(distance_between_centroids[distance_between_centroids > 0])
    return spread


if __name__ == '__main__':
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    from plotters.BispacePlotter import BispacePlotter
    from utilities.PointNormalizer import Normalizer
    from scipy.spatial.distance import cdist
    from matplotlib import pyplot as plt
    import numpy as np
    import cvxpy as cp

    grid_reso = 11j
    n_runs = 4
    n_pareto_points = 10

    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)

    # Components Shared Between Input and Output
    normalizer_input = Normalizer(X)
    X_norm = normalizer_input.normalize()

    pi_pj_input = cdist(
        X_norm,
        X_norm,
        metric="euclidean",
    )
    normalizer_output = Normalizer(Y)
    Y_norm = normalizer_output.normalize()
    pi_pj_output = cdist(
        Y_norm,
        Y_norm,
        metric="euclidean",
    )

    N, indim = X.shape
    _, outdim = Y.shape
    y_input = cp.Variable(
        N,
        boolean=True,
    )
    y_output = cp.Variable(
        N,
        boolean=True,
    )
    # Input Maximal Spread
    if True:
        D = cp.max(pi_pj_input)
        eta_input = cp.Variable()
        constraints_input = []
        constraints_input += [
            eta_input <= pi_pj_input[i, j] + (1 - y_input[i]) * D + (1 - y_input[j]) * D for i in range(N) for j in range(N) if j > i
        ]
        constraints_input += [
            cp.sum(y_input) >= n_runs,
        ]
        obj_input = cp.Maximize(eta_input)

    # Output Maximal Covering
    if True:
        z = cp.Variable(
            shape=(N, N),
            boolean=True,
        )
        eta_output = cp.Variable()
        constraints_output = []
        constraints_output += [
            eta_output >= cp.sum(z[:, j] * pi_pj_output[:, j]) for j in range(N)
        ]
        constraints_output += [
            cp.sum(z[:, j], axis=0) == 1 for j in range(N)
        ]
        constraints_output += [
            y_output[i] >= z[i, j] for i in range(N) for j in range(N)
        ]
        constraints_output += [
            cp.sum(y_output) <= n_runs,
        ]

        obj_output = cp.Maximize(-eta_output)

    if True:
        pareto_points = []

        print(
            f"".center(100, "=")
        )
        print(
            f"[SOLVING INPUT MAXIMAL SPREAD]"
        )
        print(
            f"".center(100, "=")
        )
        combined_obj1 = cp.Maximize(eta_input)
        combined_constraints = constraints_input + constraints_output
        combined_constraints += [y_input == y_output]
        combined_prob1 = cp.Problem(
            combined_obj1,
            combined_constraints,
        )
        combined_prob1.solve(verbose=True)
        input_optimal_obj1 = eta_input.value
        input_optimal_obj1_alt = spread(X_norm, y_input.value)
        input_optimal_obj2 = coverage(Y_norm, y_input.value)
        pareto_points.append([input_optimal_obj1, input_optimal_obj2])

        plotter = BispacePlotter(
            X,
            Y,
            y_input.value,
        )
        plotter.plot()
        print(f"Model Spread Objective: {input_optimal_obj1}, Function: {input_optimal_obj1_alt}")
        print(f"Model Coverage Objective: {input_optimal_obj2}")
        print(
            f"".center(100, "=")
        )
        print(
            f"[SOLVING OUTPUT MAXIMAL COVERING]"
        )
        print(
            f"".center(100, "=")
        )
        combined_obj2 = cp.Maximize(-eta_output)
        combined_prob2 = cp.Problem(
            combined_obj2,
            combined_constraints,
        )
        combined_prob2.solve(verbose=True)
        output_optimal_obj1 = spread(X_norm, y_input.value)
        output_optimal_obj2 = eta_output.value
        output_optimal_obj2_alt = coverage(Y_norm, y_input.value)
        pareto_points.append([output_optimal_obj1, output_optimal_obj2])
        print(f"Model Coverage Objective: {output_optimal_obj2}, Function: {output_optimal_obj2_alt}")
        print(f"Model Spread Objective: {output_optimal_obj1}")
        print(
            f"".center(100, "=")
        )

        plotter = BispacePlotter(
            X,
            Y,
            y_input.value,
        )
        plotter.plot()

        pareto_levels = np.linspace(np.min(pareto_points, axis=0), np.max(pareto_points, axis=0), n_pareto_points+2)
        for p in range(n_pareto_points):
            print(
                f"".center(100, "=")
            )
            print(
                f"[SOLVING PARETO NUMBER {p+1}/{n_pareto_points}]"
            )
            print(
                f"".center(100, "=")
            )
            combined_prob3 = cp.Problem(
                combined_obj1,
                combined_constraints + [eta_output <= pareto_levels[1 + p, 1]]
            )
            combined_prob3.solve(verbose=True)
            pareto_optimal_obj1 = eta_input.value
            pareto_optimal_obj2 = coverage(Y_norm, y_input.value)
            pareto_optimal_obj2_alt = eta_output.value
            print(f"Pareto level: {pareto_levels[1 + p, 1]}, obj2: {pareto_optimal_obj2:.3f}, alt: {pareto_optimal_obj2_alt:.3f}")
            pareto_points.append([pareto_optimal_obj1, pareto_optimal_obj2])
            print(f"Pareto: {[pareto_optimal_obj1, pareto_optimal_obj2]}")
            print(
                f"".center(100, "=")
            )

            plotter = BispacePlotter(
                X,
                Y,
                y_input.value,
                title=f"Pareto Point {p+1}",
            )
            plotter.plot()
        pareto_points = np.array(pareto_points)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(
            pareto_points[:, 0],
            pareto_points[:, 1],
        )
        axes.set_xlabel("Input Maximal Spread (HIGHER IS BETTER)")
        axes.set_ylabel("Output Maximal Covering (LOWER IS BETTER)")
        fig.tight_layout()

    plotter.show_plots()
