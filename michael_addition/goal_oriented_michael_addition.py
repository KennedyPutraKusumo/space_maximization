from pydex.core.designer import Designer
from model_scipy import pydex_simulate
from matplotlib import pyplot as plt
import pickle
import numpy as np


estimable_only = True
goal_oriented = False
feasible_designs_only = True
plot_spaces = False
load_results = False
save_case = False
norm_sens_by_params = False
savefig = True
dpi = 180
fig_name = ""
if goal_oriented:
    fig_name += "vdi_opt"
else:
    fig_name = "d_opt"
if estimable_only:
    fig_name += "_estimable"
if feasible_designs_only:
    fig_name += "_feasible"
if norm_sens_by_params:
    fig_name += "_norm"
fig_name += ".png"

designer = Designer()
designer.simulate = lambda ti_controls, model_parameters: pydex_simulate(ti_controls, model_parameters, identifiable_only=estimable_only)
mp = np.array([49.7796, 8.9316, 1.3177, 0.3109, 3.8781])
if estimable_only:
    designer.model_parameters = mp[2:]
else:
    designer.model_parameters = mp

with open("ma_prob_ds_31jx31j_grid_100_scr.pkl", "rb") as file:
    ds_samples = pickle.load(file)
waterfall_samples = ds_samples.loc[(ds_samples["feas_probs"] <= 95) & (ds_samples["feas_probs"] >= 5)]
feasible_samples = ds_samples.loc[ds_samples["feas_probs"] < 5]

if plot_spaces:
    fig2 = plt.figure()
    axes2 = fig2.add_subplot(111)
    axes2.scatter(
        feasible_samples["d1"],
        feasible_samples["d2"],
        c="C0",
        alpha=0.5,
    )
    axes2.set_xlabel("Feed Ratio (AH/B)")
    axes2.set_ylabel(r"$\tau$ (min)")
    fig2.tight_layout()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        waterfall_samples["d1"],
        waterfall_samples["d2"],
        c=waterfall_samples["feas_probs"],
    )
    axes.set_xlabel("Feed Ratio (AH/B)")
    axes.set_ylabel("$\tau$ (min)")
    axes.set_xlim([9, 31])
    axes.set_ylim([350, 1450])
    fig.tight_layout()
    plt.show()

if feasible_designs_only:
    designer.ti_controls_candidates = np.array(
        [feasible_samples["d1"], feasible_samples["d2"]]).T
else:
    reso = 11
    designer.ti_controls_candidates = designer.enumerate_candidates(
        bounds=[
            [10, 30],
            [400, 1400],
        ],
        levels=[
            reso,
            reso,
        ],
    )
if goal_oriented:
    designer.ds_tic = waterfall_samples
    criterion = designer.vdi_criterion
    optimizer = "l-bfgs-b"
    pkg = "scipy"
    opts = {
        "disp": 5,
        "maxfun": int(30000),
    }
else:
    criterion = designer.d_opt_criterion
    solver_no = 1
    if solver_no == 1:
        optimizer = "MOSEK"
        pkg = "cvxpy"
        opts = {}
    elif solver_no == 2:
        optimizer = "l-bfgs-b"
        pkg = "scipy"
        opts = {
            "disp": True,
        }
    elif solver_no == 3:
        optimizer = "SLSQP"
        pkg = "scipy"
        opts = {
            "disp": True,
        }

# designer.error_cov = np.diag([1e-2, 1e-6, 1e-3, 1e-3, 1e-4, 1e-4, 1e-2]) / 1000

designer.initialize(verbose=2)
if load_results:
    designer.load_sensitivity(f"/goal_oriented_michael_addition_result/date_2022-5-4/run_1/run_1_sensitivity_{designer.n_c}_cand.pkl")
    designer.load_atomics(f"/goal_oriented_michael_addition_result/date_2022-5-4/run_1/run_1_atomics_{designer.n_c}_cand.pkl")
    save_atomics = False
    save_sens = False
elif save_case:
    save_atomics = True
    save_sens = True
else:
    save_atomics = False
    save_sens = False

designer.ti_controls_names = ["Feed Ratio (AH/B)", rf"$\tau$ (min)"]
designer._norm_sens_by_params = norm_sens_by_params
designer.design_experiment(
    criterion,
    optimizer=optimizer,
    package=pkg,
    opt_options=opts,
    save_atomics=save_atomics,
    save_sensitivities=save_sens,
)
designer.print_optimal_candidates()
fig = designer.plot_optimal_controls(non_opt_candidates=True, write=True)
fig.savefig(fig_name, dpi=dpi)
designer.show_plots()
