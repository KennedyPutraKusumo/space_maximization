from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    with open("feasible_space.pkl", "rb") as file:
        fig = pickle.load(file)
    fig.tight_layout()
    axes1, axes2 = fig.get_axes()

    plt.show()
