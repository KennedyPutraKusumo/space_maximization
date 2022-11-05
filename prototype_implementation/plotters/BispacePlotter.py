from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class BispacePlotter:
    def __init__(self, X, Y, y, in_labels=None, out_labels=None, title=None, marker_labels=None):
        self.X = X
        self.Y = Y
        self.y = y
        self.npoints, self.indim = self.X.shape
        self.npoints, self.outdim = self.Y.shape
        # optional plotting attribute
        self.in_labels = in_labels
        self.out_labels = out_labels
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
        fig = plt.figure(figsize=self.figsize)
        if self.title:
            fig.suptitle(self.title)
        if self.indim == 2:
            axes1 = fig.add_subplot(121)
            self.plot2d(axes1, self.X, self.in_labels)
        elif self.indim == 3:
            axes1 = fig.add_subplot(121, projection="3d")
            self.plot3d(axes1, self.X, self.in_labels)
        else:
            print(
                f"[WARNING]: plotting for {self.indim} number of dimensional points not "
                f"implemented yet, skipping command."
            )
            pass
        if self.outdim == 2:
            axes2 = fig.add_subplot(122)
            self.plot2d(axes2, self.Y, self.out_labels)
        elif self.outdim == 3:
            axes2 = fig.add_subplot(122, projection="3d")
            self.plot3d(axes2, self.Y, self.out_labels)
        else:
            print(
                f"[WARNING]: plotting for {self.indim} number of dimensional points not "
                f"implemented yet, skipping command."
            )
            pass
        if self.tight_layout:
            fig.tight_layout()
        return fig

    def plot2d(self, axes, points, labels=None):
        axes.scatter(
            points[:, 0],
            points[:, 1],
            c=self.cmap(self.colour_scale),
            s=self.markersize,
        )
        if labels is not None:
            axes.set_xlabel(labels[0])
            axes.set_ylabel(labels[1])

        if self.marker_labels is not None:
            for i, ml in enumerate(self.marker_labels):
                axes.text(
                    s=ml,
                    x=points[i, 0],
                    y=points[i, 1],
                    verticalalignment="center_baseline",
                    horizontalalignment="center",
                    c="white",
                    fontsize=self.fontsize,
                    fontweight=self.fontweight,
                )
        axes.scatter(
            points[:, 0],
            points[:, 1],
            marker="H",
            edgecolor="tab:red",
            facecolor="none",
            s=self.markersize_selected * self.y,
            lw=self.linewidth,
        )
        return axes

    def plot3d(self, axes, points, labels):
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=self.cmap(self.colour_scale),
            s=self.markersize,
        )
        if labels is not None:
            axes.set_xlabel(labels[0])
            axes.set_ylabel(labels[1])
            axes.set_zlabel(labels[2])

        if self.marker_labels is not None:
            for i, ml in enumerate(self.marker_labels):
                axes.text(
                    s=ml,
                    x=points[i, 0],
                    y=points[i, 1],
                    z=points[i, 2],
                    verticalalignment="center_baseline",
                    horizontalalignment="center",
                    c="white",
                    fontsize=self.fontsize,
                    fontweight=self.fontweight,
                    zdir=None,
                )
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            marker="H",
            edgecolor="tab:red",
            facecolor="none",
            s=self.markersize_selected * self.y,
            lw=self.linewidth,
        )
        return axes

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
