import matplotlib.pyplot as plt


def plot_dataset(inputs, labels):
    plt.figure(dpi=200)
    plt.scatter(
        inputs[:, 0],
        inputs[:, 1],
        alpha=0.9,
        linewidths=0,
        s=10,
        c=['red' if y else 'green' for y in labels],
    )