import numpy as np


class BiObjectiveWeaklyDominatedPointFilter:
    def __init__(self):
        self._points = None
        self._sense = None
        self.retentate = None
        self.permeate = None
        self.repeated_pairs_idx = None

    @property
    def sense(self):
        return self._sense

    @sense.setter
    def sense(self, sense):
        self._sense = sense

    @sense.deleter
    def sense(self):
        self._sense = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points

    @points.deleter
    def points(self):
        self._points = None

    def identify_weakly_dominated_points(self):
        assert self._points is not None, "Please define the points to identify the weakly dominated points first"
        assert self._sense is not None, "Please define the sense of optimality first"
        self.permeate = []
        self.retentate = []
        self.permeate, self.retentate = self.filter(self.points)
        return self.permeate, self.retentate

    def filter(self, points):
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points):
                if not np.all(point2 == point1):
                    x1, y1 = point1
                    x2, y2 = point2
                    if x1 == x2:
                        if y1 == y2:
                            self.repeated_pairs_idx.append([i, j])
                        elif self.sense[1] == "maximize":
                            if y2 > y1:
                                self.permeate.append(point2)
                                self.retentate.append(point1)
                            elif y2 < y1:
                                self.permeate.append(point1)
                                self.retentate.append(point2)
                        elif self.sense[1] == "minimize":
                            if y2 > y1:
                                self.permeate.append(point1)
                                self.retentate.append(point2)
                            elif y2 < y1:
                                self.permeate.append(point2)
                                self.retentate.append(point1)
                    elif y1 == y2:
                        if x1 == x2:
                            self.repeated_pairs_idx.append([i, j])
                        elif self.sense[0] == "maximize":
                            if x2 > x1:
                                self.permeate.append(point2)
                                self.retentate.append(point1)
                            elif x2 < x1:
                                self.permeate.append(point1)
                                self.retentate.append(point2)
                        elif self.sense[0] == "minimize":
                            if x2 > x1:
                                self.permeate.append(point1)
                                self.retentate.append(point2)
                            elif x2 < x1:
                                self.permeate.append(point2)
                                self.retentate.append(point1)
                    else:
                        self.permeate.append(point1)
                        self.permeate.append(point2)

        self.permeate = np.array(self.permeate)
        self.retentate = np.array(self.retentate)
        to_del_idx = []
        for i, perm in enumerate(self.permeate):
            for j, rent in enumerate(self.retentate):
                if np.all(perm == rent):
                    to_del_idx.append(i)
        self.permeate = np.delete(self.permeate, to_del_idx, axis=0)
        return np.unique(self.permeate, axis=0), np.unique(self.retentate, axis=0)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure()
    axes = fig.add_subplot(111)
    if False:
        set_name = "set_1"
        obj_vals = np.array([
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [3.5, 6.0],
            [4.0, 6.0],
        ])
        if True:
            sense = np.array(["maximize", "minimize"])
        else:
            sense = np.array(["minimize", "maximize"])
    else:
        set_name = "set_2"
        obj_vals = np.array([
            [1.0, 4.0],
            [2.0, 4.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 2.0],
            [5.0, 1.0],
            [6.0, 1.0],
        ])
        if False:
            sense = np.array(["maximize", "maximize"])
        else:
            sense = np.array(["minimize", "minimize"])
    axes.scatter(
        obj_vals[:, 0],
        obj_vals[:, 1],
    )

    filter_1 = BiObjectiveWeaklyDominatedPointFilter()
    filter_1.points = obj_vals
    filter_1.sense = sense
    filter_1.identify_weakly_dominated_points()
    axes.scatter(
        filter_1.permeate[:, 0],
        filter_1.permeate[:, 1],
        edgecolor="tab:red",
        facecolor="none",
        marker="H",
        s=100,
    )
    axes.scatter(
        filter_1.retentate[:, 0],
        filter_1.retentate[:, 1],
        edgecolor="tab:green",
        facecolor="none",
        marker="s",
        s=100,
    )
    print(filter_1.retentate)
    print(filter_1.permeate)
    axes.set_xlabel(f"Obj 1 ({sense[0]})")
    axes.set_ylabel(f"Obj 2 ({sense[1]})")
    fig.tight_layout()
    fig.savefig(f"{set_name}_{sense[0]}_{sense[1]}.png", dpi=160)
    plt.show()
