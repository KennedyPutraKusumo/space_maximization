from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from michael_addition.model_scipy import multvar_sim_cqa
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pickle


def output_coverage(exp, full_output=False, normalize_cqa=False):
    if exp.ndim == 1:
        exp = exp.reshape(int(exp.shape[0]/2), 2)

    cqa_list = multvar_sim_cqa(exp)
    cqa_means = np.mean(cqa_list, axis=0)[None, :]

    if normalize_cqa:
        cqa_list = cqa_list / cqa_means

    try:
        hull = ConvexHull(cqa_list)
    except ValueError:
        print(f"[WARNING]: Convex Hull computation error, returning 0 coverage volume.")
        return 0
    if normalize_cqa:
        cqa_list = cqa_list * cqa_means

    if full_output:
        return hull.volume, cqa_list, hull
    return hull.volume

def compute_optimal_output(n_exp, opt_init_x=None, optimizer="slsqp", random_seed=None, normalize_cqa=False, constraints=None):
    if opt_init_x is None:
        if random_seed is None:
            np.random.seed(123457)
        else:
            np.random.seed(random_seed)
        opt_init_x = np.random.uniform(
            low=[10, 400],
            high=[30, 1400],
            size=(n_exp, 2),
        )
    opt_bound = np.array([
        [10, 30],
        [400, 1400],
    ] * n_exp)
    if constraints is None:
        opt_result = minimize(
            lambda x: -output_coverage(x),
            x0=opt_init_x,
            bounds=opt_bound,
            method=optimizer,
            options={
                "disp": False,
            },
        )
    else:
        opt_result = minimize(
            lambda x: -output_coverage(x),
            x0=opt_init_x,
            bounds=opt_bound,
            method=optimizer,
            options={
                "disp": False,
            },
            constraints=constraints,
        )
    return opt_result

def multistart_opt_output(n_exp, multistart_sites, optimizer="slsqp", random_seed=None, normalize_cqa=False, constraints=None):
    if random_seed is None:
        np.random.seed(12345)
    else:
        np.random.seed(random_seed)

    initial_guesses = np.random.uniform(
        low=[10, 400],
        high=[30, 1400],
        size=(n_exp * multistart_sites, 2),
    )
    initial_guesses = initial_guesses.reshape((multistart_sites, n_exp, 2))
    opt_candidates = []
    obj_vals = []
    for i, init_x in enumerate(initial_guesses):
        if constraints is None:
            opt_output = compute_optimal_output(
                n_exp,
                opt_init_x=init_x,
                optimizer=optimizer,
                normalize_cqa=normalize_cqa,
            )
        else:
            opt_output = compute_optimal_output(
                n_exp,
                opt_init_x=init_x,
                optimizer=optimizer,
                normalize_cqa=normalize_cqa,
                constraints=constraints,
            )
        opt_candidates.append(opt_output.x.reshape((n_exp, 2)))
        obj_vals.append(opt_output.fun)
    final_exp = opt_candidates[np.argmin(obj_vals)]
    return final_exp


if __name__ == '__main__':
    """ Optional Routines """
    show_feasible_hull = True
    show_factorial_performance = True
    show_maximal_output = False

    """ Optimization Settings """
    random_seed = 12345             # benchmark: 12345
    optimizer = "slsqp"             # benchmark: "slsqp"
    n_exp = 4                       # benchmark: 4
    multistart_sites = 1000          # benchmark: [20, 99, 200, 500]

    """ Plotting Settings """
    markersize = 125
    markeralpha = 0.50

    if show_feasible_hull:
        with open("feasible_space.pkl", "rb") as file:
            fig = pickle.load(file)
            axes1, axes2 = fig.get_axes()
    else:
        fig = plt.figure(figsize=(13, 5))
        axes1 = fig.add_subplot(121)
        axes1.set_xlabel("Feed Ratio (AH/B)")
        axes1.set_ylabel("Residence Time (min)")
        axes2 = fig.add_subplot(122)
        axes2.set_xlabel("Conversion of Feed C (mol/mol)")
        axes2.set_ylabel("Concentration of AC- (mol/L)")
    fig.suptitle(
        f"Maximal Output Design (i) seed: {random_seed} (ii) optimizer: {optimizer} "
        f"(iii) n_exp: {n_exp} (iv) multistart sites: {multistart_sites}"
    )

    if show_factorial_performance:
        if show_feasible_hull:
            with open("feasible_space.pkl", "rb") as file:
                figb = pickle.load(file)
                axes1b, axes2b = figb.get_axes()
        else:
            # figb = plt.figure(figsize=(13, 5))
            figb = plt.figure(figsize=(8, 3))
            axes1b = figb.add_subplot(121)
            axes1b.set_xlabel("Feed Ratio (AH/B)")
            axes1b.set_ylabel("Residence Time (min)")
            axes2b = figb.add_subplot(122)
            axes2b.set_xlabel("Conversion of Feed C (mol/mol)")
            axes2b.set_ylabel("Concentration of AC- (mol/L)")
        figb.suptitle("Factorial Design")

        exp_1 = np.array([
            [10, 400],
            [10, 1400],
            [30, 400],
            [30, 1400],
        ])
        hullVol, CQAs, convHull = output_coverage(exp_1, full_output=True)
        print(f"Factorial design leads to {hullVol:.4E} output coverage in volume.")

        cmap = np.linspace(0, 1, exp_1.shape[0])
        axes1b.scatter(
            exp_1[:, 0],
            exp_1[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=markeralpha,
            s=markersize,
        )
        axes2b.scatter(
            CQAs[:, 0],
            CQAs[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=markeralpha,
            s=markersize,
        )
        for simplex in convHull.simplices:
            axes2b.plot(
                CQAs[simplex, 0],
                CQAs[simplex, 1],
                c="tab:red",
                ls=(1, (1, 1)),
            )
        axes2b.text(
            x=0.17,
            y=0.0015,
            s=f"Volume: {convHull.volume:.2E}",
            c="tab:red",
            fontsize="large",
        )
        figb.tight_layout()

    if show_maximal_output:
        opt_exp = multistart_opt_output(
            n_exp=n_exp,
            multistart_sites=multistart_sites,
            optimizer=optimizer,
            random_seed=random_seed,
        )
        hullVol, CQAs, convHull = output_coverage(opt_exp, full_output=True)
        print(f"The maximal output_design leads to {hullVol:.4E} output coverage in volume.")

        cmap = np.linspace(0, 1, opt_exp.shape[0])
        axes1.scatter(
            opt_exp[:, 0],
            opt_exp[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=markeralpha,
            s=markersize,
        )
        axes2.scatter(
            CQAs[:, 0],
            CQAs[:, 1],
            c=cm.gist_rainbow(cmap),
            alpha=markeralpha,
            s=markersize,
        )
        axes1.set_xlim([9, 31])
        axes1.set_ylim([350, 1450])
        for simplex in convHull.simplices:
            axes2.plot(
                CQAs[simplex, 0],
                CQAs[simplex, 1],
                c="tab:red",
                ls=(1, (1, 1)),
            )
        axes2.text(
            x=0.17,
            y=0.0015,
            s=f"Volume: {convHull.volume:.2E}",
            c="tab:red",
            fontsize="large",
        )

        fig.tight_layout()
        fig.savefig("ma_maximal_output.png", dpi=360)
    if show_factorial_performance:
        figb.savefig("ma_factorial.png", dpi=360)
    plt.show()
