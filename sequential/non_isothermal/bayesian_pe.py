from model import run_exp, predict_exp
from matplotlib import pyplot as plt
import emcee as mc
import numpy as np


def delta(p, input, data):
    predictions = predict_exp(input[0], input[1], input[2], p)
    diff = predictions - data
    return diff

def log_prior(p):
    if 0 < p[0] < 5 and 1000 < p[1] < 10000 and 0 < p[0] < 5 and 1000 < p[1] < 10000:
        return 0
    else:
        return -np.inf

def log_lkhd(p, input, data):
    rss = delta(p, input, data) ** 2
    if np.nansum(rss) == np.nan:
        print(p)
    return np.nansum(rss)

def log_probability(p, input, data):
    log_prob = log_prior(p) + log_lkhd(p, input, data)
    if np.isnan(log_prob).any():
        return -np.inf
    return log_prob

if __name__ == '__main__':
    time_reso = 41

    """ Model & Data Inputs """
    ca0 = 1
    T = 285
    tau = 24
    t = np.linspace(0, tau, time_reso)
    in_vars = [
        t,
        ca0,
        T,
    ]

    """ Bayesian PE Inputs """
    nwalkers = 32
    ndim = 4
    nsteps = int(1e4)

    """" Code """
    mp = np.array([2, 2500, 1, 5000])
    yerr = 0.05**2
    data = run_exp(
        t,
        ca0,
        T,
        mp,
        yerr,
        seed=1234,
    )

    prediction = predict_exp(
        t,
        ca0,
        T,
        mp,
    )

    diff = delta(mp, in_vars, data)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        t,
        data[0],
        c="C0",
    )
    axes.scatter(
        t,
        data[1],
        c="C1",
    )
    axes.scatter(
        t,
        data[2],
        c="C2",
    )
    axes.plot(
        t,
        prediction[0],
        c="C0",
    )
    axes.plot(
        t,
        prediction[1],
        c="C1",
    )
    axes.plot(
        t,
        prediction[2],
        c="C2",
    )
    for i, (delta_ind, prediction_ind) in enumerate(zip(diff, prediction)):
        for time, diff, pred in zip(t, delta_ind, prediction_ind):
            axes.plot(
                [time, time],
                [pred, pred - diff],
                c=f"C{i}",
            )

    rss = log_lkhd(mp, in_vars, data)
    print(rss)

    mp = [6.71233, 2543.00214, -151.12364, 5053.41089]
    log_prob = log_probability(mp, in_vars, data)
    print(log_prob)

    sampler = mc.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(in_vars, data)
    )
    initial_walker_pos = mp + np.random.randn(nwalkers, ndim)
    sampler.run_mcmc(initial_walker_pos, nsteps, progress=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$k_1$", r"$e_1$", r"$k_2$", r"$e_2$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("step number")
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

    import corner

    fig = corner.corner(
        flat_samples, labels=labels, truths=[mp[0], mp[1], mp[2], mp[3]]
    )

    plt.show()
