from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


comp_stat = pd.read_csv("computational_stats.csv", index_col="Criterion")
comp_stat.plot(
    x="Grid Size",
    y="Computational Time (s)",
)
plt.show()
