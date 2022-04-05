from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    fig = plt.figure()
    axes = fig.add_subplot(111)

    epss = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for e in epss:
        X = np.array([
            [-1, -1],
            [-1 + e, 1],
            [1, -1 + e],
            [1, 1 - e],
        ])
        axes.scatter(
            X[:, 0],
            X[:, 1],
        )

    plt.show()
