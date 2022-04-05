import scipy.spatial.qhull
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, distance_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from time import time
import numpy as np


""" Function which computes input space's objective value for given experiment and bounds """
def input_objective(exp, bounds):
    assert isinstance(exp, np.ndarray), "The given experiment must be a numpy array"
    assert isinstance(bounds, np.ndarray), "The given bound must be a numpy array"
    # check if dimension on input is consistent between the given exp and bounds
    assert exp.shape[1] == bounds.shape[0], f"The length of axis 1 from given exp ({exp.shape[1]}) does not match the length of axis 0 from given bounds ({bounds.shape[0]})"
    n_in = exp.shape[1]

    # scale the experiment to the range -1 and 1
    scaled_exp = exp - bounds.T[0, :][None, :]
    bound_ranges = (bounds.T[1, :] - bounds.T[0, :])[None, :]
    scaled_exp = scaled_exp / bound_ranges
    scaled_exp *= 2
    scaled_exp -= 1

    in_mat = scaled_exp.T @ scaled_exp
    in_obj_val = np.linalg.slogdet(in_mat)
    if in_obj_val[0] == 0:
        return 0
    else:
        in_obj_val = in_obj_val[0] * np.exp(in_obj_val[1])

    return in_obj_val

def coater_model(exp):
    T_air_in = exp[:, 0]
    M_coat = exp[:, 1]
    Q_air = exp[:, 2]

    m_air_in = Q_air * (273 / (273 + T_air_in)) * (28.3 * 29.0 / 22.4)

    cp_air = 0.238
    cp_w = 1.0
    x_w = 0.8
    h_vap_w = 540
    T_amb = 20
    P_amb = 1.01325
    hlf = 0
    T_coat = 20
    percent_rh_in = 15

    U = m_air_in * cp_air + x_w * M_coat * cp_w + hlf
    T_air_out = (m_air_in * cp_air * T_air_in + x_w * M_coat * cp_w * T_coat - x_w * M_coat * h_vap_w + hlf * T_amb) / U

    # Antoine coefficients for water
    A = 4.6543
    B = 1435.264
    C = -64.848

    log_p_vap_w_out = A - B / (T_air_out + 273 + C)
    log_p_vap_w_in = A - B / (T_air_in + 273 + C)
    p_vap_w_out = 10 ** log_p_vap_w_out
    p_vap_w_in = 10 ** log_p_vap_w_in

    m_w_in = m_air_in * (percent_rh_in * p_vap_w_in) / (100 * P_amb) * (18.02 / 29)
    m_w_out = m_w_in + x_w * M_coat
    y_w_out = m_w_out * 29.0 / (m_air_in * 18.02)
    percent_rh_out = y_w_out * P_amb * 100 / p_vap_w_out
    m_w_out_max = (p_vap_w_out / P_amb * m_air_in * 18.02 / 29.0 - m_w_in) * 0.6 + m_w_in

    # there are two outputs (i) T air out (Celsius) and (ii) Relative Humidity Out (%)
    return np.array([T_air_out, percent_rh_out, m_w_out, m_w_out_max])

def output_objective(exp, interpretation="convex_hull"):
    response = coater_model(exp).T

    if interpretation == "convex_hull":
        """ Convex Hull Interpretation """
        hull = ConvexHull(response[:, (0, 1)])
        return hull.volume
    elif interpretation == "ellipsoid":
        """ Ellipsoid Approximation """
        cov_mat = np.cov(response[:, (0, 1)].T)
        sign, logdet = np.linalg.slogdet(cov_mat)
        return 4/3 * np.pi * np.sqrt(sign * np.exp(logdet))
    elif interpretation == "centroid_distance":
        """ Distance from Chosen Centroid """
        centroid = np.mean(response[:, (0, 1)], axis=0)
        dist = distance_matrix(response[:, (0, 1)], centroid[None, :])
        return np.sum(dist)
    elif interpretation == "orthogonal_bracketing":
        scaled_res = 0
        return

def maximize_input(x, bounds=np.array([[20, 85], [10, 80], [150, 450]])):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    return -input_objective(exp, bounds)

def maximize_output(x, interpretation="convex_hull"):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    return -output_objective(exp, interpretation)

def constraints(x):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    res = coater_model(exp).T
    c = np.array([
        200 - res[:, 0],            # max temperature 200 C
        res[:, 0] - 0,              # min temperature 0 C
        res[:, 3] - res[:, 2],      # m_w_out <= m_w_out_max
    ]).flatten()
    return c

def biobj(n_exp, method="slsqp", multistart_sites=10, initial_method=None, reproduce_feasible_space=True, interpretation="convex_hull", pareto_reso=11j, plot_custom_criteria=False, plot_qi_original_results=False, label_pareto_points=True):
    if reproduce_feasible_space:
        reso = 101j
        exp_max1, exp_max2, exp_max3 = np.mgrid[20:85:reso, 10:80:reso, 150:450:reso]
        exp_max1 = exp_max1.flatten()
        exp_max2 = exp_max2.flatten()
        exp_max3 = exp_max3.flatten()
        exp_max = np.array([exp_max1, exp_max2, exp_max3]).T
        response = coater_model(exp_max).T

        # filter out points which has m_w_out that exceeds m_w_out_max
        exp_max = exp_max[response[:, 2] <= response[:, 3]]
        response = response[response[:, 2] <= response[:, 3]]
        # filter out point which has T_air_out above 200
        exp_max = exp_max[response[:, 0] <= 200]
        response = response[response[:, 0] <= 200]
        # filter out point which has T_air_out below 0
        exp_max = exp_max[response[:, 0] >= 0]
        response = response[response[:, 0] >= 0]

        feasible_space_convex_hull = ConvexHull(response[:, (0, 1)])

    start_time = time()
    print("".center(100, "="))
    if initial_method is None:
        print(f"Running for {n_exp} number of experiments with {method} as optimizer and {multistart_sites} multi-start sites.")
    else:
        print(f"Running for {n_exp} number of experiments with {method} as optimizer and {multistart_sites} multi-start sites, but using {initial_method} as optimizer to compute the extreme Pareto points.")
    print("".center(100, "="))
    print(f"[{time() - start_time:.2f} s]: solving for maximal input bracketing design (bottom-right)")
    np.random.seed(12345)

    exp_inits = np.random.uniform(
        low=[20, 10, 150],
        high=[85, 80, 450],
        size=(n_exp * multistart_sites, 3),
    )
    exp_inits = exp_inits.reshape(multistart_sites, n_exp, 3)
    input_objectives = []
    output_objectives = []
    opt_results = []
    """ Computing the Maximal Input Bracketing Design """
    for exp_init in exp_inits:
        if initial_method is None:
            opt_input_result = minimize(
                    maximize_input,
                    x0=exp_init.flatten(),
                    bounds=np.asarray([
                                          (20, 85),
                                          (10, 80),
                                          (150, 450),
                                      ] * n_exp),
                    method=method,
                    options={"disp": False},
                    constraints={"type": "ineq", "fun": constraints},
            )
        else:
            opt_input_result = minimize(
                    maximize_input,
                    x0=exp_init.flatten(),
                    bounds=np.asarray([
                                          (20, 85),
                                          (10, 80),
                                          (150, 450),
                                      ] * n_exp),
                    method=initial_method,
                    options={"disp": False},
                    constraints={"type": "ineq", "fun": constraints},
            )
        opt_results.append(opt_input_result)

        opt_input_out_obj = output_objective(opt_input_result.x.reshape(n_exp, 3), interpretation=interpretation)
        output_objectives.append(opt_input_out_obj)
        opt_input_in_obj = -opt_input_result.fun
        input_objectives.append(opt_input_in_obj)
    final_solution_idx = np.argmax(input_objectives)
    final_opt_input_output_obj = output_objectives[final_solution_idx]
    final_opt_input_input_obj = input_objectives[final_solution_idx]
    final_opt_input_opt_exp = opt_results[final_solution_idx]
    final_opt_input_ellipsoid = output_objective(opt_input_result.x.reshape(n_exp, 3), interpretation="ellipsoid")
    final_opt_input_convex_hull = output_objective(opt_input_result.x.reshape(n_exp, 3), interpretation="convex_hull")
    print(f"[{time() - start_time:.2f} s]: completed, with {final_opt_input_input_obj:.2f} input criterion and {final_opt_input_output_obj:.2f} output criterion.")

    """ Plot the optimal input bracketing design """
    print(f"[{time() - start_time:.2f} s]: plotting results...")
    input_xlim = [10, 90]
    input_ylim = [10, 90]
    input_zlim = [140, 460]

    output_xlim = [0, 100]
    output_ylim = [0, 100]

    opt_input_exp = final_opt_input_opt_exp.x.reshape(n_exp, 3)
    fig1 = plt.figure(figsize=(14, 7))
    axes1a = fig1.add_subplot(121, projection="3d")
    axes1a.scatter(
        opt_input_exp[:, 0],
        opt_input_exp[:, 1],
        opt_input_exp[:, 2],
    )
    axes1a.set_xlabel("T air in (Celsius)")
    axes1a.set_ylabel("M coat (g/min)")
    axes1a.set_zlabel("Q air (ft3/min)")

    axes1a.set_xlim(input_xlim)
    axes1a.set_ylim(input_ylim)
    axes1a.set_zlim(input_zlim)

    opt_input_res = coater_model(opt_input_exp).T
    final_opt_input_centroid_dist = centroid_distances(opt_input_res[:, (0, 1)])
    axes1b = fig1.add_subplot(122)
    axes1b.scatter(
        opt_input_res[:, 0],
        opt_input_res[:, 1],
    )
    axes1b.set_xlabel("T air out (Celsius)")
    axes1b.set_ylabel("Relative humidity of air out (%)")
    axes1b.set_xlim(output_xlim)
    axes1b.set_ylim(output_ylim)

    axes1a.set_title(f"Input Criterion: {final_opt_input_input_obj:.2f}")
    axes1b.set_title(f"Output Criterion: {final_opt_input_output_obj:.2f}")

    convex_hull = ConvexHull(opt_input_res[:, (0, 1)])
    for simplex in convex_hull.simplices:
        axes1b.plot(opt_input_res[simplex, 0], opt_input_res[simplex, 1], c="tab:red", ls="dashed", linewidth=1, alpha=1, marker="None")

    if reproduce_feasible_space:
        for simplex in feasible_space_convex_hull.simplices:
            axes1b.plot(response[simplex, 0], response[simplex, 1], c="k", ls="solid", linewidth=1, alpha=1, marker="None")

    fig1.suptitle("Optimal D-input")
    fig1.tight_layout()

    """ Compute the optimal output exploration design """
    print(f"[{time() - start_time:.2f} s]: solving for maximal output exploration design (top-left)")
    input_objectives = []
    output_objectives = []
    opt_results = []
    for exp_init in exp_inits:
        if initial_method is None:
            opt_output_result = minimize(
                maximize_output,
                x0=exp_init.flatten(),
                bounds=np.asarray([
                                      (20, 85),
                                      (10, 80),
                                      (150, 450),
                                  ] * n_exp),
                method=method,
                constraints={"type": "ineq", "fun": constraints},
                args=(interpretation),
            )
        else:
            opt_output_result = minimize(
                maximize_output,
                x0=exp_init.flatten(),
                bounds=np.asarray([
                                      (20, 85),
                                      (10, 80),
                                      (150, 450),
                                  ] * n_exp),
                method=initial_method,
                constraints={"type": "ineq", "fun": constraints},
                args=(interpretation),
            )
        opt_results.append(opt_output_result)
        opt_output_out_obj = -opt_output_result.fun
        output_objectives.append(opt_output_out_obj)
        opt_output_in_obj = input_objective(opt_output_result.x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))
        input_objectives.append(opt_output_in_obj)
    final_opt_output_solution_idx = np.argmax(output_objectives)
    final_opt_output_output_obj = output_objectives[final_solution_idx]
    final_opt_output_input_obj = input_objectives[final_solution_idx]
    final_opt_output_opt_exp = opt_results[final_solution_idx]
    final_opt_output_convex_hull = output_objective(opt_output_result.x.reshape(n_exp, 3), interpretation="convex_hull")
    final_opt_output_ellipsoid = output_objective(opt_output_result.x.reshape(n_exp, 3), interpretation="ellipsoid")
    print(f"[{time() - start_time:.2f} s]: completed, with {final_opt_output_input_obj:.2f} input criterion and {final_opt_output_output_obj:.2f} output criterion.")

    """ Plot the optimal output exploration design """
    print(f"[{time() - start_time:.2f} s]: plotting results...")
    opt_output_exp = final_opt_output_opt_exp.x.reshape(n_exp, 3)
    fig2 = plt.figure(figsize=(14, 7))
    axes2a = fig2.add_subplot(121, projection="3d")
    axes2a.scatter(
        opt_output_exp[:, 0],
        opt_output_exp[:, 1],
        opt_output_exp[:, 2],
    )
    axes2a.set_xlabel("T air in (Celsius)")
    axes2a.set_ylabel("M coat (g/min)")
    axes2a.set_zlabel("Q air (ft3/min)")

    axes2a.set_xlim(input_xlim)
    axes2a.set_ylim(input_ylim)
    axes2a.set_zlim(input_zlim)

    opt_output_res = coater_model(opt_output_exp).T
    final_opt_output_centroid_dist = centroid_distances(opt_output_res[:, (0, 1)])
    axes2b = fig2.add_subplot(122)
    axes2b.scatter(
        opt_output_res[:, 0],
        opt_output_res[:, 1],
    )

    convex_hull = ConvexHull(opt_output_res[:, (0, 1)])
    for simplex in convex_hull.simplices:
        axes2b.plot(opt_output_res[simplex, 0], opt_output_res[simplex, 1], c="tab:red", ls="dashed", linewidth=1, alpha=1, marker="None")

    axes2b.set_xlabel("T air out (Celsius)")
    axes2b.set_ylabel("Relative humidity of air out (%)")
    axes2b.set_xlim(output_xlim)
    axes2b.set_ylim(output_ylim)

    axes2a.set_title(f"Input Criterion: {final_opt_output_input_obj:.2f}")
    axes2b.set_title(f"Output Criterion: {final_opt_output_output_obj:.2f}")

    """ Plot the Convex Hull of Feasible Space """
    if reproduce_feasible_space:
        for simplex in feasible_space_convex_hull.simplices:
            axes2b.plot(response[simplex, 0], response[simplex, 1], c="k", ls="solid", linewidth=1, alpha=1, marker="None")

    fig2.suptitle("Optimal Output Exploration")
    fig2.tight_layout()

    fig3 = plt.figure()
    fig4 = plt.figure()
    axes3 = fig3.add_subplot(111)
    axes4 = fig4.add_subplot(111)

    axes3.scatter(
        x=[final_opt_input_input_obj, final_opt_output_input_obj],
        y=[final_opt_input_output_obj, final_opt_output_output_obj],
    )
    if plot_custom_criteria:
        axes4.scatter(
            x=[final_opt_input_input_obj / final_opt_input_input_obj, final_opt_output_input_obj / final_opt_input_input_obj],
            y=[final_opt_input_centroid_dist / final_opt_output_centroid_dist, final_opt_output_centroid_dist / final_opt_output_centroid_dist],
            c="tab:purple",
            label="L2 Distance from Centroid of Chosen Points",
        )
    axes4.scatter(
        x=[final_opt_input_input_obj / final_opt_input_input_obj, final_opt_output_input_obj / final_opt_input_input_obj],
        y=[final_opt_input_convex_hull / final_opt_output_convex_hull, final_opt_output_convex_hull / final_opt_output_convex_hull],
        color="none",
        marker="d",
        edgecolor="tab:blue",
        s=200,
        label="Reproduced Convex Hull",
    )
    axes4.scatter(
        x=[final_opt_input_input_obj / final_opt_input_input_obj, final_opt_output_input_obj / final_opt_input_input_obj],
        y=[final_opt_input_ellipsoid / final_opt_output_ellipsoid, final_opt_output_ellipsoid / final_opt_output_ellipsoid],
        c="none",
        marker="^",
        edgecolor="tab:orange",
        s=200,
        label="Reproduced Ellipsoid",
    )
    axes4.text(
        x=0.1,
        y=0.4,
        s=f"Reproduction used the {interpretation} interpretation"
    )

    """ Generating Pareto Points in Between the Extremes """
    # epsilon-constraint method to populate the Pareto frontier
    input_values = np.mgrid[final_opt_output_input_obj:final_opt_input_input_obj:pareto_reso]
    input_values = input_values[1:-1]

    success_counter = 0
    fail_counter = 0
    crit_values = []
    for i, val in enumerate(input_values):
        print(f"[Pareto point {i+1}]".center(100, "="))
        print(f"[{time() - start_time:.2f} s]: intermediate input criterion of {val:.2f}")
        input_objectives = []
        output_objectives = []
        opt_results = []
        for exp_init in exp_inits:
            try:
                opt_output_result = minimize(
                    maximize_output,
                    x0=exp_init.flatten(),
                    bounds=np.asarray([
                                          (20, 85),
                                          (10, 80),
                                          (150, 450),
                                      ] * n_exp),
                    method=method,
                    constraints=[
                        {"type": "ineq", "fun": constraints},
                        {"type": "ineq", "fun": lambda x: (input_objective(x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))) - val},
                    ],
                    args=(interpretation),
                )
            except scipy.spatial.qhull.QhullError:
                pass
            opt_results.append(opt_output_result)
            opt_output_out_obj = -opt_output_result.fun
            output_objectives.append(opt_output_out_obj)
            opt_output_in_obj = input_objective(opt_output_result.x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))
            input_objectives.append(opt_output_in_obj)
        final_solution_idx = np.argmax(output_objectives)
        final_output_obj = output_objectives[final_solution_idx]
        final_input_obj = input_objectives[final_solution_idx]
        final_opt_exp_results = opt_results[final_solution_idx]
        final_opt_exp = final_opt_exp_results.x.reshape(n_exp, 3)
        pareto_convex_hull = output_objective(final_opt_exp, interpretation="convex_hull")
        pareto_ellipsoid = output_objective(final_opt_exp, interpretation="ellipsoid")
        fig_pareto = plt.figure(figsize=(14, 7))
        axes_paretoa = fig_pareto.add_subplot(121, projection="3d")
        axes_paretob = fig_pareto.add_subplot(122)
        axes_paretoa.scatter(
            final_opt_exp[:, 0],
            final_opt_exp[:, 1],
            final_opt_exp[:, 2],
        )
        axes_paretoa.set_title(f"Input Criterion: {final_input_obj:.2f}")
        axes_paretob.set_title(f"Output Criterion: {final_output_obj:.2f}")
        final_opt_exp_res = coater_model(final_opt_exp).T
        pareto_centroid_distances = centroid_distances(final_opt_exp_res[:, (0, 1)])
        axes_paretob.scatter(
            final_opt_exp_res[:, 0],
            final_opt_exp_res[:, 1],
        )
        convex_hull = ConvexHull(final_opt_exp_res[:, (0, 1)])
        for simplex in convex_hull.simplices:
            axes_paretob.plot(final_opt_exp_res[simplex, 0], final_opt_exp_res[simplex, 1], c="tab:red", ls="dashed", linewidth=1, alpha=1, marker="None")

        if reproduce_feasible_space:
            for simplex in feasible_space_convex_hull.simplices:
                axes_paretob.plot(response[simplex, 0], response[simplex, 1], c="k", ls="solid", linewidth=1, alpha=1, marker="None")

        axes_paretoa.set_xlabel("T air in (Celsius)")
        axes_paretoa.set_ylabel("M coat (g/min)")
        axes_paretoa.set_zlabel("Q air (ft3/min)")

        axes_paretoa.set_xlim(input_xlim)
        axes_paretoa.set_ylim(input_ylim)
        axes_paretoa.set_zlim(input_zlim)

        axes_paretob.set_xlabel("T air out (Celsius)")
        axes_paretob.set_ylabel("Relative humidity of air out (%)")
        axes_paretob.set_xlim(output_xlim)
        axes_paretob.set_ylim(output_ylim)

        fig_pareto.suptitle(f"Pareto point number {i+1}, input criterion: {val:.2f}")

        fig_pareto.tight_layout()

        if final_opt_exp_results.success:
            success_counter += 1
            crit_values.append([final_input_obj, final_output_obj])
            print(f"[{time() - start_time:.2f} s]: SUCCESS! Appended point with {final_input_obj:.2f} input criterion, and {final_output_obj:.2f} output criterion.")
            axes3.scatter(
                final_input_obj,
                final_output_obj,
                c="tab:red",
            )
            axes4.scatter(
                final_input_obj / final_opt_input_input_obj,
                pareto_convex_hull / final_opt_output_convex_hull,
                color="none",
                marker="d",
                edgecolor="tab:blue",
                s=200,
            )
            axes4.scatter(
                final_input_obj / final_opt_input_input_obj,
                pareto_ellipsoid / final_opt_output_ellipsoid,
                color="none",
                marker="^",
                edgecolor="tab:orange",
                s=200,
            )
            if label_pareto_points:
                axes3.annotate(
                    text=f"{i+1}",
                    xy=(final_input_obj, final_output_obj),
                    xytext=(final_input_obj, final_output_obj - 1000),
                    arrowprops={
                        "facecolor": "black",
                        "shrink": 0.05,
                        "width": 1,
                    },
                )
                axes4.annotate(
                    text=f"{i+1}",
                    xy=(final_input_obj / final_opt_input_input_obj, final_output_obj / final_opt_output_output_obj),
                    xytext=(final_input_obj / final_opt_input_input_obj, (final_output_obj - 1000) / final_opt_output_output_obj),
                    arrowprops={
                        "facecolor": "black",
                        "shrink": 0.05,
                        "width": 1,
                    },
                )
            if plot_custom_criteria:
                # plotting the L2 distance from chosen centroid
                axes4.scatter(
                    final_input_obj / final_opt_input_input_obj,
                    pareto_centroid_distances / final_opt_output_centroid_dist,
                    c="tab:purple",
                )
        else:
            fail_counter += 1
            print(f"[{time() - start_time:.2f} s]: FAIL! Pareto point number {i+1} failed to compute, with an intermediate input criterion value of {val:.2f}.")
            print(f"Optimizer status was: {final_opt_exp_results.status} with termination message: {final_opt_exp_results.message}")
            pass
    crit_values = np.array(crit_values)

    print(f"[Completed within {time() - start_time:.2f} seconds]".center(100, "="))
    print(f"A total of {success_counter + 2:d} Pareto points obtained, with {fail_counter:d} failed runs.")

    axes3.set_xlabel("Input Bracketing Criterion")
    axes3.set_ylabel("Output Exploration Criterion")

    axes3.set_xticks([0, 16, 32, 48, 64])
    axes3.set_title("Non-normalized Pareto Frontier")
    fig3.tight_layout()

    """ Qi's initial results """
    if plot_qi_original_results:
        qi_initial_ellipsoid = np.array([
            [1.0         ,		0.4600986928,			0.1017206992],
            [1.0         ,		0.4586615877,			0.1017206992],
            [1.0         ,		0.4600986928,			0.1017206992],
            [1.0         ,		0.4600986928,			0.1017206992],
            [1.0         ,		0.4215306034,			0.1017206992],
            [1.0         ,		0.4549216518,			0.1017206992],
            [0.9915485662,		0.7749805212,			0.6312666922],
            [0.9915485662,		0.7411652671,			0.6312666922],
            [0.9809844508,		0.7845727643,			0.6312666922],
            [0.9728695538,		0.8686866938,			0.8430850893],
            [0.9612974806,		0.9140680461,			0.8562489445],
            [0.9014123914,		0.9469136871,			0.9014249021],
            [0.8358007426,		0.9704874037,			0.9349328972],
            [0.7643977632,		0.9898883214,			0.9621581431],
            [0.5897566745,		1.0			,			0.9744244627],
            [0.6890077104,		1.0			,			0.9744244627],
            [0.6646040853,		1.0			,			0.9744244627],
            [0.0846351141,		1.0			,			0.9744244627],
            [1.0		 ,		0.4600986928,			0.1017206992],
            [0.9915485662,		0.7749805212,			0.6312666922],
            [0.9728695538,		0.9056618475,			0.8430850893],
            [0.9728695538,		0.9056618475,			0.8430850893],
            [0.9728695538,		0.9056618475,			0.8430850893],
            [0.9728695538,		0.9056618475,			0.8430850893],
            [0.9728695538,		0.9056618475,			0.8430850893],
            [0.95779596	 ,		0.9163968488,			0.8598390869],
            [0.7934855072,		0.9833088044,			0.9537811444],
            [0.7216752241,		1.0			,			0.9744244627],
            [0.7216752241,		1.0			,			0.9744244627],
        ])

        # plotting the ellipsoid approximation results
        axes4.scatter(
            x=qi_initial_ellipsoid[:, 0],
            y=qi_initial_ellipsoid[:, 1],
            marker="^",
            c="tab:orange",
            label="Qi's Original Results Ellipsoid",
        )

        # plotting the convex hull results
        axes4.scatter(
            x=qi_initial_ellipsoid[:, 0],
            y=qi_initial_ellipsoid[:, 2],
            marker="d",
            c="tab:blue",
            label="Qi's Original Results Convex Hull",
        )

    axes4.set_xlabel("Input Bracketing Criterion")
    axes4.set_ylabel("Output Exploration Criterion")

    axes4.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    axes4.set_title("Normalized Pareto Frontier")
    axes4.legend()
    fig4.tight_layout()

    plt.show()

    return axes4

""" A list of other possible criterion for the output space exploration """
def centroid_distances(points):
    points = points.reshape(int(points.size / 2), 2)
    centroid = np.mean(points, axis=0)
    dist = distance_matrix(points, centroid[None, :])
    return np.sum(dist)

if __name__ == '__main__':
    axes4 = biobj(
        n_exp=4,
        method="slsqp",
        initial_method="trust-constr",
        multistart_sites=200,
        interpretation="convex_hull",
        pareto_reso=11j,
        plot_custom_criteria=False,
        plot_qi_original_results=True,
        label_pareto_points=True,
    )
