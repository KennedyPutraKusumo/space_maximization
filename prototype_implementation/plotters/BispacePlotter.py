from matplotlib import pyplot as plt
from matplotlib import cm


class BispacePlotter:
    def __init__(self, X, Y, y, in_labels=None, out_labels=None, title=None, marker_labels=None):
        self.X = X
        self.Y = Y
        self.y = y
        self.npoints, self.ndim = self.X.shape
        # optional plotting attribute
        self.in_labels = in_labels
        self.out_labels = out_labels
        self.in_xlim = [np.min(X, axis=0)[0], np.max(X, axis=0)[0]]
        self.in_ylim = [np.min(X, axis=0)[1], np.max(X, axis=0)[1]]
        self.out_xlim = [np.min(Y, axis=0)[0], np.max(Y, axis=0)[0]]
        self.out_ylim = [np.min(Y, axis=0)[1], np.max(Y, axis=0)[1]]
        # default options
        self.figsize = (15, 8)
        self.tight_layout = True
        self.cmap = cm.viridis
        self.colour_scale = np.linspace(0, 1, X.shape[0])
        self.markersize = 300
        self.markersize_selected = 800
        self.title = title
        self.marker_labels = marker_labels
        self.fontsize = "small"
        self.fontweight = "bold"
        self.linewidth = 2

    def plot(self):
        if self.ndim == 2:
            fig = self.plot2d()
            return fig
        else:
            print(
                f"[WARNING]: plotting for {self.ndim} number of dimensional points not"
                f"implemented yet, skipping command."
            )
            pass

    def plot2d(self):
        fig = plt.figure(figsize=self.figsize)
        if self.title:
            fig.suptitle(self.title)
        axes1 = fig.add_subplot(121)
        axes1.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=self.cmap(self.colour_scale),
            s=self.markersize,
        )
        axes2 = fig.add_subplot(122)
        axes2.scatter(
            self.Y[:, 0],
            self.Y[:, 1],
            c=self.cmap(self.colour_scale),
            s=self.markersize,
        )
        in_scale = [self.in_xlim[1] - self.in_xlim[0], self.in_ylim[1] - self.in_ylim[0]]
        out_scale = [self.out_xlim[1] - self.out_xlim[0], self.out_ylim[1] - self.out_ylim[0]]
        axes1.set_xlim(left=self.in_xlim[0] - 0.05 * in_scale[0], right=self.in_xlim[1] + 0.05 * in_scale[0])
        axes1.set_ylim(bottom=self.in_ylim[0] - 0.05 * in_scale[1], top=self.in_ylim[1] + 0.05 * in_scale[1])
        axes2.set_xlim(left=self.out_xlim[0] - 0.05 * out_scale[0], right=self.out_xlim[1] + 0.05 * out_scale[0])
        axes2.set_ylim(bottom=self.out_ylim[0] - 0.05 * out_scale[1], top=self.out_ylim[1] + 0.05 * out_scale[1])
        if self.in_labels:
            axes1.set_xlabel(self.in_labels[0])
            axes1.set_ylabel(self.in_labels[1])
        if self.out_labels:
            axes2.set_xlabel(self.out_labels[0])
            axes2.set_ylabel(self.out_labels[1])
        if self.tight_layout:
            fig.tight_layout()
        if self.marker_labels is not None:
            for i, ml in enumerate(self.marker_labels):
                axes1.text(
                    s=ml,
                    x=self.X[i, 0],
                    y=self.X[i, 1],
                    verticalalignment="center_baseline",
                    horizontalalignment="center",
                    c="white",
                    fontsize=self.fontsize,
                    fontweight=self.fontweight,
                )
                axes2.text(
                    s=ml,
                    x=self.Y[i, 0],
                    y=self.Y[i, 1],
                    verticalalignment="center_baseline",
                    horizontalalignment="center",
                    c="white",
                    fontsize=self.fontsize,
                    fontweight=self.fontweight,
                )
        axes1.scatter(
            self.X[:, 0],
            self.X[:, 1],
            marker="H",
            edgecolor="tab:red",
            facecolor="none",
            s=self.markersize_selected * self.y,
            lw=self.linewidth,
        )
        axes2.scatter(
            self.Y[:, 0],
            self.Y[:, 1],
            marker="H",
            edgecolor="tab:red",
            facecolor="none",
            s=self.markersize_selected * self.y,
            lw=self.linewidth,
        )
        return fig

    @staticmethod
    def show_plots():
        plt.show()

if __name__ == '__main__':
    import numpy as np
    from mip_formulations.ma_maximal_spread import multvar_sim_cqa

    grid_reso = 11j
    n_centroids = 4
    rnd_seed = 123
    in_label = [
        "Feed Ratio (AH/B)",
        "Residence Time (min)",
    ]
    out_label = [
        "Conversion of Feed C (mol/mol)",
        "Concentration of AC- (mol/L)",
    ]

    x1, x2 = np.mgrid[10:30:grid_reso, 400:1400:grid_reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    Y = multvar_sim_cqa(X)
    y = np.zeros(X.shape[0])
    np.random.seed(123)
    for _ in range(4):
        y[np.random.randint(0, X.shape[0], n_centroids)] = 1

    plotter1 = BispacePlotter(X, Y, y)
    fig = plotter1.plot()
    plotter1.show_plots()
