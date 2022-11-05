from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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
    return np.array([T_air_out, percent_rh_out, m_w_out, m_w_out_max]).T

if __name__ == '__main__':
    reso = 5j
    exp_max1, exp_max2, exp_max3 = np.mgrid[20:85:reso, 10:80:reso, 150:450:reso]
    exp_max1 = exp_max1.flatten()
    exp_max2 = exp_max2.flatten()
    exp_max3 = exp_max3.flatten()
    exp_max = np.array([exp_max1, exp_max2, exp_max3]).T
    response = coater_model(exp_max)

    # filter out points which has m_w_out that exceeds m_w_out_max
    exp_max = exp_max[response[:, 2] <= response[:, 3]]
    response = response[response[:, 2] <= response[:, 3]]
    # filter out point which has T_air_out above 200
    exp_max = exp_max[response[:, 0] <= 200]
    response = response[response[:, 0] <= 200]
    # filter out point which has T_air_out below 0
    exp_max = exp_max[response[:, 0] >= 0]
    response = response[response[:, 0] >= 0]

    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    axes.scatter(
        exp_max[:, 0],
        exp_max[:, 1],
        exp_max[:, 2],
    )

    from prototype_implementation.EffortDesigner import EffortDesigner
    from prototype_implementation.BiobjectiveDesigner import BiObjectiveDesigner
    from prototype_implementation.criteria.MaximalSpread import MaximalSpread
    from prototype_implementation.criteria.MaximalCovering import MaximalCovering

    criteria = [MaximalSpread, MaximalCovering]
    for criterion1 in criteria:
        for criterion2 in criteria:
            designer_1 = EffortDesigner()
            designer_1.space_of_interest = "input"
            designer_1.package = "cvxpy"
            designer_1.optimizer = "GUROBI"
            designer_1.criterion = criterion1

            designer_2 = EffortDesigner()
            designer_2.space_of_interest = "output"
            designer_2.package = "cvxpy"
            designer_2.optimizer = "GUROBI"
            designer_2.criterion = criterion2

            designer_3 = BiObjectiveDesigner()
            designer_3.input_points = exp_max
            designer_3.output_points = response[:, [0, 1]]
            designer_3.designers = [designer_1, designer_2]

            designer_3.n_runs = 4
            designer_3.n_epsilon_points = 10
            designer_3.start_logging()
            designer_3.initialize(verbose=2)
            designer_3.design_experiments(plot=True, write=True)
            designer_3.plot_pareto_frontier()
            designer_3.stop_logging()
    plt.show()
