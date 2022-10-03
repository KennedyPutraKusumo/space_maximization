from matplotlib import  pyplot as plt
import numpy as np


def simulate(t, ca0, T, theta):
    R = 8.314
    k1 = theta[0] * np.exp(-theta[1] / (R * T))
    k2 = theta[2] * np.exp(-theta[3] / (R * T))
    ca = ca0 * np.exp(-k1 * t)
    cb = k1 * ca0 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    cc = ca0 * (1 + 1/(k1 - k2) * (k2 * np.exp(-k1 * t) - k1 * np.exp(-k2 * t)))
    # cc = ca0 - ca - cb
    check = ca + cb + cc
    return np.array([
        ca,
        cb,
        cc,
        check,
    ])

def predict_exp(t, ca0, T, theta):
    eta = simulate(t, ca0, T, theta)
    return eta[0:3]

def run_exp(t, ca0, T, theta, yerr, seed):
    np.random.seed(seed)
    eta = simulate(t, ca0, T, theta)
    return eta[0:3] + np.random.multivariate_normal([0, 0, 0], cov=np.diag([yerr, yerr, yerr]), size=eta.shape[1]).T

def sensitivities(t, ca0, T, theta):
    R = 8.314
    f21 = -ca0*theta[0]*t*np.exp(-2*theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))) + ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-2*theta[1]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2 + ca0*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))
    f22 = ca0*theta[0]**2*t*np.exp(-2*theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))) - ca0*theta[0]**2*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-2*theta[1]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2) - ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))))
    f23 = ca0*theta[0]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))*np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T)))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))) - ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2
    f24 = -ca0*theta[0]*theta[2]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))*np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T)))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))) + ca0*theta[0]*theta[2]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2)

    f11 = -ca0*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))
    f12 = ca0*theta[0]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(R*T)
    f13 = np.zeros_like(t)
    f14 = np.zeros_like(t)

    f31 = ca0*theta[0]*t*np.exp(-2*theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))) - ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-2*theta[1]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2 + ca0*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))) - ca0*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))
    f32 = -ca0*theta[0]**2*t*np.exp(-2*theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))) + ca0*theta[0]**2*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-2*theta[1]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2) - ca0*theta[0]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T)))/(R*T) + ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))))
    f33 = -ca0*theta[0]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))*np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T)))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T))) + ca0*theta[0]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))/(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2
    f34 = ca0*theta[0]*theta[2]*t*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))*np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T)))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))) - ca0*theta[0]*theta[2]*(-np.exp(-theta[2]*t*np.exp(-theta[3]/(R*T))) + np.exp(-theta[0]*t*np.exp(-theta[1]/(R*T))))*np.exp(-theta[1]/(R*T))*np.exp(-theta[3]/(R*T))/(R*T*(-theta[0]*np.exp(-theta[1]/(R*T)) + theta[2]*np.exp(-theta[3]/(R*T)))**2)

    return np.array([
        [f11, f12, f13, f14],
        [f21, f22, f23, f24],
        [f31, f32, f33, f34],
    ])


def cqa_plot(t, ca0, T, theta):
    eta = simulate(t, ca0, T, theta)
    tau = np.max(t)
    ca = eta[0]
    cb = eta[1]
    cc = eta[2]
    profit = 1000 * (5 * cb - 1 * ca0) / (tau + 30)
    # purity = cb[-1] / (ca[-1] + cb[-1] + cc[-1])
    purity = cb / (ca + cb + cc)
    return np.array([
        profit,
        purity,
    ])

def cqas(t, ca0, T, theta):
    eta = simulate(t, ca0, T, theta)
    tau = np.max(t)
    ca = eta[0][-1]
    cb = eta[1][-1]
    cc = eta[2][-1]
    profit = 1000 * (5 * cb - 1 * ca0) / (tau + 30)
    # purity = cb[-1] / (ca[-1] + cb[-1] + cc[-1])
    purity = cb / (ca + cb + cc)
    return np.array([
        profit,
        purity,
    ])

def g(t, ca0, T, theta):
    cqa = cqas(t, ca0, T, theta)
    g = np.array([
        cqa[0] - 30,
        cqa[1] - 0.50,
    ])
    return g

if __name__ == '__main__':
    norm_sens = True
    marker_size = 1
    alpha = 0.8
    time_reso = 401
    tau = 24
    ca0 = 1
    t = np.linspace(0, tau, time_reso)
    mp = np.array([2, 2500, 1, 5000])
    T = 285
    eta = simulate(t, ca0, T, mp)
    s = sensitivities(t, ca0, T, mp)
    if norm_sens:
        s = s * mp[None, :, None]
    cqa = cqa_plot(t, ca0, T, mp)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(fr"$T={T}, \ \tau={np.max(t):.0f}, \ [A]_0={ca0}, \ k_1={mp[0]}, \ E_1={mp[1]}, \ k_2={mp[2]}, \ E_2={mp[3]}$")
    axes = fig.add_subplot(221)
    axes.scatter(
        t,
        eta[0],
        label="[A]",
        s=marker_size,
        alpha=0.5,
    )
    axes.scatter(
        t,
        eta[1],
        label="[B]",
        s=marker_size,
        alpha=0.5,
    )
    axes.scatter(
        t,
        eta[2],
        label="[C]",
        s=marker_size,
        alpha=0.5,
    )
    axes.scatter(
        t,
        eta[3],
        label="[A] + [B] + [C]",
        s=marker_size,
        alpha=0.5,
    )
    axes.legend()
    axes.set_xlabel("Time (hour)")
    axes.set_ylabel("Concentration (mol/L)")

    axes2 = fig.add_subplot(222)
    axes2.set_title(f"Sensitivity, normalized: {norm_sens}")
    axes2.scatter(
        t,
        s[0, 0],
        label=r"$\frac{d[A]}{d\theta_1}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[0, 1],
        label=r"$\frac{d[A]}{d\theta_2}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[0, 2],
        label=r"$\frac{d[A]}{d\theta_3}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[0, 3],
        label=r"$\frac{d[A]}{d\theta_4}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[1, 0],
        label=r"$\frac{d[B]}{d\theta_1}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[1, 1],
        label=r"$\frac{d[B]}{d\theta_2}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[1, 2],
        label=r"$\frac{d[B]}{d\theta_3}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[1, 3],
        label=r"$\frac{d[B]}{d\theta_4}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[2, 0],
        label=r"$\frac{d[C]}{d\theta_1}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[2, 1],
        label=r"$\frac{d[C]}{d\theta_2}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[2, 2],
        label=r"$\frac{d[C]}{d\theta_3}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.scatter(
        t,
        s[2, 3],
        label=r"$\frac{d[C]}{d\theta_4}$",
        s=marker_size,
        alpha=0.5,
    )
    axes2.legend()
    axes2.set_xlabel("Time (hour)")
    axes2.set_ylabel(r"Sensitivities of [B] w.r.t. $\mathbf{\theta}$ (mol/L)")

    axes3 = fig.add_subplot(223)
    axes3.scatter(
        t,
        cqa[0],
        label="Profit",
        s=marker_size,
        alpha=0.5,
    )
    axes3.set_xlabel("Time (hour)")
    axes3.set_ylabel("Profit ($/hour)")

    axes4 = fig.add_subplot(224)
    axes4.scatter(
        t,
        cqa[1],
        label="Purity",
        s=marker_size,
        alpha=0.5,
    )
    axes4.set_xlabel("Time (hour)")
    axes4.set_ylabel("Purity (mol/mol)")

    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=180)
    plt.show()
