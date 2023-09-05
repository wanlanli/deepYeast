import matplotlib.pyplot as plt


def plot_cells(image, ct=None, ot=None, tip=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image, 'gray')
    if ct is not None:
        ax.scatter(ct[:, 1], ct[:, 0], c='y')
    if ot is not None:
        for i in range(0, ot.shape[0]):
            ax.plot(ot[i][1], ot[i][0], label=i)
        ax.legend()
    if tip is not None:
        for i in range(0, tip.shape[0]):
            ax.scatter(tip[i, :, 1], tip[i, :, 0], label=i, marker='*')
