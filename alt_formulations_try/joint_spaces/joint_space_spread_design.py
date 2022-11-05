if __name__ == '__main__':
    from prototype_implementation.EffortDesigner import EffortDesigner
    from prototype_implementation.criteria.MaximalSpread import MaximalSpread
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa
    from matplotlib import pyplot as plt
    from matplotlib import cm
    import numpy as np

    grid_reso = 11j
    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)
    joint = np.append(X, Y, axis=1)

    """ Obtain the Joint Space Design """
    designer_1 = EffortDesigner()
    designer_1.input_points = joint
    designer_1.space_of_interest = "input"
    designer_1.package = "cvxpy"
    designer_1.optimizer = "GUROBI"
    designer_1.criterion = MaximalSpread
    designer_1.n_runs = 4
    designer_1.initialize()

    designer_1.design_experiments()
    designer_1.get_optimal_candidates()
    designer_1.print_results()

    opt_df = designer_1.pd_df[designer_1.pd_df["Repetitions"] > 0]
    fig = plt.figure(figsize=(15, 8))
    axes1 = fig.add_subplot(121)
    c = np.linspace(0, 1, designer_1.npoints)
    axes1.scatter(
        designer_1.pd_df["Input 1"],
        designer_1.pd_df["Input 2"],
        c=cm.viridis(c),
        s=300,
    )
    axes1.scatter(
        opt_df["Input 1"],
        opt_df["Input 2"],
        facecolor="none",
        edgecolor="tab:red",
        marker="H",
        s=800,
    )
    axes2 = fig.add_subplot(122)
    axes2.scatter(
        designer_1.pd_df["Input 3"],
        designer_1.pd_df["Input 4"],
        c=cm.viridis(c),
        s=300,
    )
    axes2.scatter(
        opt_df["Input 3"],
        opt_df["Input 4"],
        facecolor="none",
        edgecolor="tab:red",
        marker="H",
        s=800,
    )
    fig.tight_layout()
    fig.savefig("joint_space_spread_design.png", dpi=160)

    """ Compute the Input Criterion """
    designer_2 = EffortDesigner()
    designer_2.input_points = X
    designer_2.output_points = Y
    designer_2.space_of_interest = "input"
    designer_2.package = "cvxpy"
    designer_2.optimizer = "GUROBI"
    designer_2.criterion = MaximalSpread
    designer_2.n_runs = 4
    designer_2.fixed_y = designer_1.pd_df["Repetitions"]
    designer_2.initialize()
    designer_2.design_experiments()
    designer_2.get_optimal_candidates()
    designer_2.print_results()

    """ Compute the Output Criterion """
    designer_3 = EffortDesigner()
    designer_3.input_points = X
    designer_3.output_points = Y
    designer_3.space_of_interest = "output"
    designer_3.package = "cvxpy"
    designer_3.optimizer = "GUROBI"
    designer_3.criterion = MaximalSpread
    designer_3.n_runs = 4
    designer_3.fixed_y = designer_1.pd_df["Repetitions"]
    designer_3.initialize()
    designer_3.design_experiments()
    designer_3.get_optimal_candidates()
    designer_3.print_results()
    designer_3.plot_results(
        title="Michael Addition",
        in_labels=[
            "Feed Ratio (AH/B)",
            "Residence Time (min)",
        ],
        out_labels=[
            "Conversion of Feed C (mol/mol)",
            "Concentration of AC- (mol/L)",
        ],
    )

    """ Add the Joint Space to Pareto Frontier """
    objs = np.array([
        # from biobj_results/date_2022_10_07/biobj_input_spread_output_spread_11x11_grid_4_runs_10_paretos
        [2.000, 0.714],
        [1.200, 1.026],
        [1.800, 0.798],
        [1.612, 0.834],
        [1.600, 0.894],
        [1.414, 0.922],
        [1.414, 0.943],
        [1.400, 1.006],
    ])
    joint_obj = np.array(
        [designer_2.objective_value, designer_3.objective_value],
    )
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        objs[:, 0],
        objs[:, 1],
        c=cm.viridis(np.linspace(0, 1, objs.shape[0]))
    )
    axes.scatter(
        joint_obj[0],
        joint_obj[1],
        c="tab:red",
        marker="x",
    )
    axes.set_xlabel("Maximal Input Spread (higher is better)")
    axes.set_ylabel("Maximal Output Spread (higher is better)")
    fig.tight_layout()
    fig.savefig("pareto_with_joint_spread_design.png", dpi=180)
    designer_3.show_plots()
