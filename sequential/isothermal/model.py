from matplotlib import  pyplot as plt
import numpy as np


def simulate(t, theta, ca0):
    ca = ca0 * np.exp(-theta[0] * t)
    cb = theta[0] * ca0 / (theta[1] - theta[0]) * (np.exp(-theta[0] * t) - np.exp(-theta[1] * t))
    # cc = ca0 * [1 + 1/(theta[0] - theta[1]) * (theta[1] * np.exp(-theta[0] * t) - theta[0] * np.exp(-theta[1] * t))]
    cc = ca0 - ca - cb
    check = ca + cb + cc
    return np.array([
        ca,
        cb,
        cc,
        check,
    ])

def sensitivities(t, theta, ca0):
    f11 = -ca0 * t * np.exp(-t * theta[0])
    f12 = np.zeros_like(t)
    f21 = -ca0*t*theta[0]*np.exp(-t*theta[0])/(-theta[0] + theta[1]) + ca0*theta[0]*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])**2 + ca0*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])
    f22 = ca0*t*theta[0]*np.exp(-t*theta[1])/(-theta[0] + theta[1]) - ca0*theta[0]*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])**2
    f31 = ca0*t*theta[0]*np.exp(-t*theta[0])/(-theta[0] + theta[1]) + ca0*t*np.exp(-t*theta[0]) - ca0*theta[0]*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])**2 - ca0*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])
    f32 = -ca0*t*theta[0]*np.exp(-t*theta[1])/(-theta[0] + theta[1]) + ca0*theta[0]*(-np.exp(-t*theta[1]) + np.exp(-t*theta[0]))/(-theta[0] + theta[1])**2

    return np.array([
        [f11, f12],
        [f21, f22],
        [f31, f32],
    ])

def cqas(t, theta, ca0):
    eta = simulate(t, theta, ca0)
    tau = np.max(t)
    ca = eta[0]
    cb = eta[1]
    cc = eta[2]
    profit = (100 * cb - 20 * ca0) / (tau + 30)
    # purity = cb[-1] / (ca[-1] + cb[-1] + cc[-1])
    purity = cb / (ca + cb + cc)
    return np.array([
        profit,
        purity,
    ])

if __name__ == '__main__':
    time_reso = 21
    tau = 4
    ca0 = 1
    t = np.linspace(0, tau, time_reso)
    mp = [1, 0.5]
    eta = simulate(t, mp, ca0)
    s = sensitivities(t, mp, ca0)
    cqa = cqas(t, mp, ca0)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(fr"$\theta_1={mp[0]}, \ \theta_2={mp[1]}$")
    axes = fig.add_subplot(221)
    axes.scatter(
        t,
        eta[0],
        label="[A]",
    )
    axes.scatter(
        t,
        eta[1],
        label="[B]",
    )
    axes.scatter(
        t,
        eta[2],
        label="[C]",
    )
    axes.scatter(
        t,
        eta[3],
        label="[A] + [B] + [C]",
    )
    axes.legend()
    axes.set_xlabel("Time (hour)")
    axes.set_ylabel("Concentration (mol/L)")

    axes2 = fig.add_subplot(222)
    axes2.scatter(
        t,
        s[0, 0],
        label=r"$\frac{d[A]}{d\theta_1}$",
    )
    axes2.scatter(
        t,
        s[0, 1],
        label=r"$\frac{d[A]}{d\theta_2}$",
    )
    axes2.scatter(
        t,
        s[1, 0],
        label=r"$\frac{d[B]}{d\theta_1}$",
    )
    axes2.scatter(
        t,
        s[1, 1],
        label=r"$\frac{d[B]}{d\theta_2}$",
    )
    axes2.scatter(
        t,
        s[2, 0],
        label=r"$\frac{d[C]}{d\theta_1}$",
    )
    axes2.scatter(
        t,
        s[2, 1],
        label=r"$\frac{d[C]}{d\theta_2}$",
    )
    axes2.legend()
    axes2.set_xlabel("Time (hour)")
    axes2.set_ylabel(r"Sensitivities of [B] w.r.t. $\mathbf{\theta}$ (mol/L)")

    axes3 = fig.add_subplot(223)
    axes3.scatter(
        t,
        cqa[0],
        label="Profit",
    )
    axes3.set_xlabel("Time (hour)")
    axes3.set_ylabel("Profit ($/hour)")

    axes4 = fig.add_subplot(224)
    axes4.scatter(
        t,
        cqa[1],
        label="Purity",
    )
    axes4.set_xlabel("Time (hour)")
    axes4.set_ylabel("Purity (mol/mol)")

    fig.tight_layout()
    plt.show()
