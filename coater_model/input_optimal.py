from scipy.optimize import minimize
from scipy.spatial import ConvexHull
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

def output_objective(exp, interpretation="asd"):
    response = coater_model(exp).T

    if interpretation == "convex_hull":
        """ Convex Hull Interpretation """
        hull = ConvexHull(response[:, (0, 1)])
        return hull.volume
    else:
        """ Ellipsoid Approximation """
        cov_mat = np.cov(response[:, (0, 1)].T)
        sign, logdet = np.linalg.slogdet(cov_mat)

        return 4/3 * np.pi * np.sqrt(sign * np.exp(logdet))

def maximize_input(x, bounds=np.array([[20, 85], [10, 80], [150, 450]])):
    exp = x.reshape(n_exp, 3)
    return -input_objective(exp, bounds)

def maximize_output(x):
    exp = x.reshape(n_exp, 3)
    return -output_objective(exp)

def constraints(x):
    exp = x.reshape(n_exp, 3)
    res = coater_model(exp).T
    c = np.array([
        200 - res[:, 0],            # max temperature 200 C
        res[:, 0] - 0,              # min temperature 0 C
        res[:, 3] - res[:, 2],      # m_w_out <= m_w_out_max
    ]).flatten()
    return c

def biobj(n_exp):
    exp_init = np.random.uniform(
        low=[20, 10, 150],
        high=[85, 80, 450],
        size=(n_exp, 3),
    )
    opt_input_result = minimize(
            maximize_input,
            x0=exp_init,
            bounds=np.asarray([
                                  (20, 85),
                                  (10, 80),
                                  (150, 450),
                              ] * n_exp),
            method="slsqp",
            options={"disp": True},
            constraints={"type": "ineq", "fun": constraints},
    )
    opt_input_res = coater_model(opt_input_result.x)
    opt_input_hull = ConvexHull(opt_input_res[:, (0, 1)])
    opt_input_out_obj = opt_input_hull.volume
    opt_input_in_obj = opt_input_result.fun

    opt_output_result = minimize(
        maximize_output,
        x0=exp_init.flatten(),
        bounds=np.asarray([
                              (20, 85),
                              (10, 80),
                              (150, 450),
                          ] * n_exp),
        method="slsqp",
        constraints={"type": "ineq", "fun": constraints},
    )
    opt_output_res = coater_model(opt_output_result.x)
    opt_output_out_obj = opt_output_res.fun
    opt_output_in_obj = opt_output_res.x.reshape(n_exp, 3)
    opt_output_in_info_mat = opt_output_in_obj.T @ opt_output_in_obj
    opt_output_in_obj = np.linalg.det(opt_output_in_info_mat)

    return

if __name__ == '__main__':
    exp_1 = np.array([
        [20, 10, 450],
        [85, 10, 150],
        [85, 10, 450],
        [85, 80, 450],
    ])
    exp_2 = np.array([
        [20, 10, 150],
        [85, 10, 150],
        [85, 10, 450],
        [85, 80, 158],
    ])
    np.random.seed(1)

    n_exp = 4
    if n_exp == 8:
        exp_3 = np.array([
            [20, 10, 150],
            [85, 10, 150],
            [20, 80, 150],
            [20, 10, 450],

            [85, 80, 150],
            [85, 10, 450],
            [20, 80, 450],
            [85, 80, 450],
        ])
    else:
        exp_3 = np.random.uniform(
            low=[20, 10, 150],
            high=[85, 80, 450],
            size=(n_exp, 3),
        )

    """ Initial Guess for Optimization """
    exp_3[0, :] = [20.1, 79.9, 200]
    # exp_3[1, :] = [20.1, 64.25, 360]
    # exp_3[2, :] = [20.1, 79.9, 449.5]
    # exp_3[3, :] = [20, ]
    # exp_3[3, :] = [40, 50, 200]

    """ Reproduce the Coater Model Results """
    test = True
    if test:

        bounds_1 = np.array([
            [20, 85],           # T air in
            [10, 80],           # M coat
            [150, 450],         # Q air
        ])
        obj = input_objective(exp_1, bounds_1)
        print(obj)
        out = coater_model(exp_1)
        print(out)

        out2 = coater_model(exp_2)

        fig1 = plt.figure()
        axes1 = fig1.add_subplot(111)
        axes1.scatter(
            out[0, :],
            out[1, :],
            c="tab:blue",
            marker="x",
        )

        # convex hull of exp-1
        hull = ConvexHull(out.T[:, (0, 1)])
        for simplex1 in hull.simplices:
            axes1.plot(out.T[simplex1, 0], out.T[simplex1, 1], c="tab:blue", alpha=0.50, ls="--", marker="None")

        # axes1.plot(
        #     [out[0, 0], out[0, 1], out[0, 1], out[0, 2], out[0, 2], out[0, 3], out[0, 3], out[0, 0]],
        #     [out[1, 0], out[1, 1], out[1, 1], out[1, 2], out[1, 2], out[1, 3], out[1, 3], out[1, 0]],
        #     c="tab:blue",
        #     ls="--",
        # )

        axes1.scatter(
            out2[0, :],
            out2[1, :],
            c="tab:red",
            marker="x",
        )
        hull2 = ConvexHull(out2.T[:, (0, 1)])
        for simplex2 in hull2.simplices:
            axes1.plot(out2.T[simplex2, 0], out2.T[simplex2, 1], c="tab:red", alpha=0.50, ls="--", marker="None")

        # axes1.plot(
        #     [out2[0, 0], out2[0, 1], out2[0, 1], out2[0, 2], out2[0, 2], out2[0, 3], out2[0, 3], out2[0, 0]],
        #     [out2[1, 0], out2[1, 1], out2[1, 1], out2[1, 2], out2[1, 2], out2[1, 3], out2[1, 3], out2[1, 0]],
        #     c="tab:red",
        #     ls="--",
        # )

        axes1.set_xlim([10, 90])
        axes1.set_ylim([10, 100])

        response_space_vol = output_objective(exp_1)
        print(response_space_vol)

    """ Optimize Output Space Exploration """
    optimize_output = True
    manual = False
    if optimize_output:
        opt_time = time()
        opt_result = minimize(
            maximize_output,
            x0=exp_3.flatten(),
            bounds=np.asarray([
                       (20, 85),
                       (10, 80),
                       (150, 450),
                   ] * n_exp),
            method="slsqp",
            constraints={"type": "ineq", "fun": constraints},
        )
        opt_time = time() - opt_time
        print(f"Optimization took {opt_time:.2f} CPU seconds to complete.")
        opt_exp = np.array(opt_result.x)
        opt_exp = opt_exp.reshape(n_exp, 3)

        opt_response = coater_model(opt_exp).T
        opt_hull = ConvexHull(opt_response[:, (0, 1)])

        fig2 = plt.figure(figsize=(11, 5.5))
        axes2 = fig2.add_subplot(122)
        axes2.scatter(
            opt_response[:, 0],
            opt_response[:, 1],
            marker="h",
            c="None",
            edgecolors="tab:red",
            s=500,
            zorder=3,
        )
        for simplex in opt_hull.simplices:
            axes2.plot(opt_response[simplex, 0], opt_response[simplex, 1], c="k", ls="--")
        axes2.set_xlabel("Temperature Air Out (Celsius)")
        axes2.set_ylabel("Relative Humidity Out (%)")

        axes3 = fig2.add_subplot(121, projection="3d")
        axes3.scatter(
            opt_exp[:, 0],
            opt_exp[:, 1],
            opt_exp[:, 2],
            marker="h",
            s=500,
            c="tab:red",
            zorder=3,
        )
        axes3.grid(False)
        axes3.set_xlabel("Temperature Air In (Celsius)")
        axes3.set_ylabel("Mass of Coating (g/min)")
        axes3.set_zlabel("Air Inlet Flowrate (ft3/min)")

        fig2.tight_layout()

        print(f"The volume in the response space is {opt_hull.volume:.2f} units")

        if manual:
            opt_exp_manual = np.vstack((np.array([[20, 64.25, 360]]), opt_exp))
            opt_res_manual = np.vstack((coater_model(np.array([[20, 64.25, 360]])).T, opt_response))
            opt_hull_manual = ConvexHull(opt_res_manual[:, (0, 1)])

            fig2b = plt.figure(figsize=(11, 5.5))
            axes2b = fig2b.add_subplot(122)
            axes2b.scatter(
                opt_res_manual[:, 0],
                opt_res_manual[:, 1],
                marker="h",
                c="None",
                edgecolors="tab:red",
                s=500,
                zorder=3,
            )
            for simplex in opt_hull_manual.simplices:
                axes2b.plot(opt_res_manual[simplex, 0], opt_res_manual[simplex, 1], c="k", ls="--")

            axes3b = fig2b.add_subplot(121, projection="3d")
            axes3b.scatter(
                opt_exp_manual[:, 0],
                opt_exp_manual[:, 1],
                opt_exp_manual[:, 2],
                marker="h",
                s=500,
                c="tab:red",
                zorder=3,
            )
            axes3b.grid(False)
            fig2b.tight_layout()

            print(f"The volume in the response space with added manual point is {opt_hull_manual.volume:.2f} units")

    """ Reproduce the Feasible Space """
    feasible_output_space = True
    if feasible_output_space:
        reso = 21j
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

        cmap_mode = "all_basis"
        if cmap_mode == "x_basis":
            # coloring based on x-value
            cmap = (exp_max[:, 0] - 20) / (85 - 20)
        elif cmap_mode == "y_basis":
            # coloring based on y-value
            cmap = (exp_max[:, 1] - 10) / (80 - 10)
        elif cmap_mode == "z_basis":
            # coloring based on z-value
            cmap = (exp_max[:, 2] - 150) / (450 - 150)
        else:
            cmap = np.linspace(0, 1, exp_max.shape[0])

        axes3.scatter(
            exp_max[:, 0],
            exp_max[:, 1],
            exp_max[:, 2],
            c=cm.gist_rainbow(cmap)
        )
        axes2.scatter(
            response[:, 0],
            response[:, 1],
            s=1,
            alpha=1.0,
            c=cm.gist_rainbow(cmap)
        )
        axes3.view_init(elev=10, azim=65)
        # axes2.set_xlim([10, 90])
        axes2.set_ylim([0, 100])

        feasible_space_convex_hull = ConvexHull(response[:, (0, 1)])

        for simplex in feasible_space_convex_hull.simplices:
            axes2.plot(response[simplex, 0], response[simplex, 1], c="k", ls="solid", linewidth=7, alpha=0.1, marker="None")

        if manual:
            axes2b.scatter(
                response[:, 0],
                response[:, 1],
                s=1,
                alpha=0.1,
                c=cm.gist_rainbow(cmap)
            )
            # ind = np.where((response[:, 0] > 10.3) | (response[:, 0] < 10.8) | (response[:, 1] > 99.5) | (response[:, 1] < 100.5), response[:, 0], False)
            ind = np.where((response[:, 0] > 10.6805) & (response[:, 0] < 10.6815) & (response[:, 1] > 99.4600) & (response[:, 1] < 99.4625), response[:, 0], False)
            # ind = np.where(np.logical_and((response[:, 0] > 10.3), (response[:, 0] < 10.8)), response[:, 0], False)
            promising_init_cond = exp_max[ind != 0]

            print(promising_init_cond)

            fig5 = plt.figure()
            axes6 = fig5.add_subplot(111, projection="3d")
            axes6.scatter(
                promising_init_cond[:, 0],
                promising_init_cond[:, 1],
                promising_init_cond[:, 2],
            )
            axes6.set_xlim([20, 85])
            axes6.set_ylim([10, 80])
            axes6.set_zlim([150, 450])

    """ Optimize the D-Input Optimality """
    optimize_input = False
    if optimize_input:
        opt_result = minimize(
            maximize_input,
            x0=exp_3,
            bounds=np.asarray([
                                  (20, 85),
                                  (10, 80),
                                  (150, 450),
                              ] * n_exp),
            method="l-bfgs-b",
            options={"disp": True},
        )
        input_optimal_exp = opt_result.x
        input_optimal_exp = input_optimal_exp.reshape(n_exp, 3)
        input_optimal_res = coater_model(input_optimal_exp).T

        fig4 = plt.figure()
        axes4a = fig4.add_subplot(121, projection="3d")
        axes4b = fig4.add_subplot(122)
        axes4a.scatter(
            input_optimal_exp[:, 0],
            input_optimal_exp[:, 1],
            input_optimal_exp[:, 2],
        )
        # axes4b.set_xlim([10, 90])
        # axes4b.set_ylim([0, 100])
        axes4b.scatter(
            input_optimal_res[:, 0],
            input_optimal_res[:, 1],
        )

        print(input_objective(opt_result.x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]])))

    plt.show()
