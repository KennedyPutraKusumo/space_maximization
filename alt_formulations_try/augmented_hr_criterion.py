from matplotlib import pyplot as plt
import numpy as np

""" A list of other possible criterion for the output space exploration """

def augmented_hr_criterion(output_points):
    anchor_idx = np.random.randint(
        low=0,
        high=output_points.shape[0],
        size=1,
    )[0]
    anchor = output_points[anchor_idx]

    euclidean_distances = []
    for point in output_points:
        euclidean_distances.append(np.sqrt((point - anchor)[:, None].T @ (point - anchor)[:, None]))

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(
        output_points[:, 0],
        output_points[:, 1],
    )

    axes.scatter(
        output_points[anchor_idx, 0],
        output_points[anchor_idx, 1],
        c="none",
        edgecolor="tab:red",
        marker="h",
        facecolors=None,
        s=200,
    )

    return np.array(euclidean_distances).squeeze()

if __name__ == '__main__':
    res_points_1 = np.array([
        [20, 20],
        [85, 17],
        [11.5, 70],
        [50, 87],
    ])
    res_points_2 = np.array([
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    distances = augmented_hr_criterion(res_points_1)
    print(distances)
    print(distances.sum())
    plt.show()
