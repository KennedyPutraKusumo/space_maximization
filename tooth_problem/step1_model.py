from pydex.core.designer import Designer
import numpy as np


def sim(ti_controls, model_parameters, return_sensitivities=False):
    y = np.array([
            model_parameters[0] * ti_controls[0]**3 + model_parameters[1] * ti_controls[0] \
            + model_parameters[2] * ti_controls[1] ** 2 \
            + model_parameters[3] * ti_controls[1] ** 4 \
            + model_parameters[4] * ti_controls[0] * ti_controls[1] ** 2
    ])

    if not return_sensitivities:
        return y
    else:
        dydtheta = np.array([
            [ti_controls[0] ** 3],
            [ti_controls[0]],
            [ti_controls[1] ** 2],
            [ti_controls[1] ** 4],
            [ti_controls[0] * ti_controls[1] ** 2],
        ])
        return y, dydtheta


designer = Designer()
designer.use_finite_difference = False
designer.simulate = lambda tic, mp: sim(tic, mp, return_sensitivities=True)
theta = np.random.multivariate_normal([5.0, 0.6, 1.0, -2.2, -3.0], 0.50 * np.identity(5), 1000)
designer.model_parameters = theta
