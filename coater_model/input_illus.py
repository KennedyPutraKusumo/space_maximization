from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
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
    return np.array([T_air_out, percent_rh_out, m_w_out, m_w_out_max])


if __name__ == '__main__':
    design_1 = np.array([
        [20, 10, 450],
        [85, 10, 150],
        [85, 10, 450],
        [85, 80, 450],
    ])
    design_2 = np.array([
        [20, 10, 150],
        [85, 10, 150],
        [85, 10, 450],
        [85, 80, 158],
    ])
    fig = plt.figure(figsize=(15, 7))
    axes = fig.add_subplot(121, projection="3d")
    axes.scatter(
        design_1[:, 0],
        design_1[:, 1],
        design_1[:, 2],
        s=100,
        color="none",
        edgecolor="tab:blue",
    )
    axes.scatter(
        design_2[:, 0],
        design_2[:, 1],
        design_2[:, 2],
        s=200,
        color="none",
        edgecolor="tab:orange",
        marker="H",
    )
    axes.set_xlabel("T air in (Celsius)")
    axes.set_ylabel("M Coat (g/min)")
    axes.set_zlabel("Q air (ft3/min)")
    axes.set_title("Input ($u$) Space")

    out = coater_model(design_1)
    out2 = coater_model(design_2)

    axes2 = fig.add_subplot(122)
    axes2.set_title("Output ($y$) Space")
    axes2.scatter(
        out[0, :],
        out[1, :],
        color="none",
        edgecolor="tab:blue",
        marker="o",
        s=100,
        label="Design 1",
    )
    axes2.scatter(
        out2[0, :],
        out2[1, :],
        color="none",
        edgecolor="tab:orange",
        marker="H",
        s=300,
        label="Design 2",
    )
    axes2.set_xlabel("T air out (Celsius)")
    axes2.set_ylabel("Relative Humidity Out (%)")
    hull = ConvexHull(out.T[:, (0, 1)])
    # centroid = np.mean(out.T[:, (0, 1)], axis=0)
    # axes2.scatter(
    #     centroid[0],
    #     centroid[1],
    #     marker="H",
    #     color="tab:red",
    #     edgecolor="tab:red",
    #     s=300,
    # )

    hull2 = ConvexHull(out2.T[:, (0, 1)])
    for simplex1 in hull.simplices:
        axes2.plot(out.T[simplex1, 0], out.T[simplex1, 1], c="tab:blue", ls="--", marker="None")
    axes2.text(
        x=55,
        y=20,
        s=f"Volume: {hull.volume:.2f}",
        c="tab:blue",
        fontsize="xx-large",
    )
    for simplex2 in hull2.simplices:
        axes2.plot(out2.T[simplex2, 0], out2.T[simplex2, 1], c="tab:orange", ls="--", marker="None")
    axes2.text(
        x=37,
        y=45,
        s=f"Volume: {hull2.volume:.2f}",
        c="tab:orange",
        fontsize="xx-large",
    )

    axes2.legend()
    fig.tight_layout()
    fig.savefig("input_illus.png", dpi=180)
    plt.show()
