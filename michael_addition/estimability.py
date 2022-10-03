from pydex.core.designer import Designer
from model_scipy import pydex_simulate
import pickle
import numpy as np


feasible_designs_only = True

designer = Designer()
designer.simulate = pydex_simulate

with open("ma_prob_ds_31jx31j_grid_100_scr.pkl", "rb") as file:
    ds_samples = pickle.load(file)
waterfall_samples = ds_samples.loc[(ds_samples["feas_probs"] <= 95) & (ds_samples["feas_probs"] >= 5)]
feasible_samples = ds_samples.loc[ds_samples["feas_probs"] < 5]
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
designer.model_parameters = [49.7796, 8.9316, 1.3177, 0.3109, 3.8781]
designer.initialize(verbose=2)

designer.estimability_study()
