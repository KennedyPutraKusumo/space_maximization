from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
    else:
        """ Ellipsoid Approximation """
        cov_mat = np.cov(response[:, (0, 1)].T)
        sign, logdet = np.linalg.slogdet(cov_mat)

        return 4/3 * np.pi * np.sqrt(sign * np.exp(logdet))

def maximize_input(x, bounds=np.array([[20, 85], [10, 80], [150, 450]])):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    return -input_objective(exp, bounds)

def maximize_output(x):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    return -output_objective(exp)

def constraints(x):
    exp = x.reshape(int(x.shape[0] / 3), 3)
    res = coater_model(exp).T
    c = np.array([
        200 - res[:, 0],            # max temperature 200 C
        res[:, 0] - 0,              # min temperature 0 C
        res[:, 3] - res[:, 2],      # m_w_out <= m_w_out_max
    ]).flatten()
    return c

def biobj(n_exp):
    np.random.seed(12345)
    exp_init = np.random.uniform(
        low=[20, 10, 150],
        high=[85, 80, 450],
        size=(n_exp, 3),
    )
    # exp_init[0, :] = [20.1, 79.9, 200]
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
    opt_input_out_obj = output_objective(opt_input_result.x.reshape(n_exp, 3))
    opt_input_in_obj = -opt_input_result.fun

    input_xlim = [10, 90]
    input_ylim = [10, 90]
    input_zlim = [140, 460]

    output_xlim = [0, 100]
    output_ylim = [0, 100]

    opt_input_exp = opt_input_result.x.reshape(n_exp, 3)
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
    axes1b = fig1.add_subplot(122)
    axes1b.scatter(
        opt_input_res[:, 0],
        opt_input_res[:, 1],
    )
    axes1b.set_xlabel("T air out (Celsius)")
    axes1b.set_ylabel("Relative humidity of air out (%)")
    axes1b.set_xlim(output_xlim)
    axes1b.set_ylim(output_ylim)

    fig1.suptitle("Optimal D-input")
    fig1.tight_layout()

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
    # opt_output_res = coater_model(opt_output_result.x.reshape(n_exp, 3))
    opt_output_out_obj = -opt_output_result.fun
    opt_output_in_obj = input_objective(opt_output_result.x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))

    opt_output_exp = opt_output_result.x.reshape(n_exp, 3)
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
    axes2b = fig2.add_subplot(122)
    axes2b.scatter(
        opt_output_res[:, 0],
        opt_output_res[:, 1],
    )
    axes2b.set_xlabel("T air out (Celsius)")
    axes2b.set_ylabel("Relative humidity of air out (%)")
    axes2b.set_xlim(output_xlim)
    axes2b.set_ylim(output_ylim)

    fig2.suptitle("Optimal Output Exploration")
    fig2.tight_layout()

    fig3 = plt.figure()
    axes3 = fig3.add_subplot(111)
    axes3.scatter(
        x=[opt_input_in_obj, opt_output_in_obj],
        y=[opt_input_out_obj, opt_output_out_obj],
    )

    # epsilon-constraint method to populate the Pareto frontier
    reso = 21j
    input_values = np.mgrid[opt_output_in_obj:opt_input_in_obj:reso]
    input_values = input_values[1:-1]

    crit_values = []
    for val in input_values:
        opt_output_result = minimize(
            maximize_output,
            x0=exp_init.flatten(),
            bounds=np.asarray([
                                  (20, 85),
                                  (10, 80),
                                  (150, 450),
                              ] * n_exp),
            method="trust-constr",
            constraints=[
                {"type": "ineq", "fun": constraints},
                {"type": "ineq", "fun": lambda x: (input_objective(x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))) - val},
            ],
            options={"disp": True},
        )
        if opt_output_result.success:
            in_obj = input_objective(opt_output_result.x.reshape(n_exp, 3), bounds=np.array([[20, 85], [10, 80], [150, 450]]))
            out_obj = opt_output_result.fun
            crit_values.append([in_obj, -out_obj])
        else:
            pass
    crit_values = np.array(crit_values)

    axes3.scatter(
        crit_values[:, 0],
        crit_values[:, 1],
    )

    plt.show()

    return
if __name__ == '__main__':
    biobj(4)
