import numpy as np


def plot_grid(x, cmap=None, border_color=None, axes=None, linewidth=1):
    """Plot a grid labeling including grid boundaries.

    Parameters
    ==========
    x : array-like, 2d
        Input array to plot.

    cmap : matplotlib colormap
        Colormap to be used for array.

    border_color : matplotlib color
        Color to be used to plot borders between grid-cells.

    axes : matplotlib axes object
        We will plot into this.

    linewidth : int
        Linewidth for the grid-cell borders.

    Returns
    =======
    axes : matplotlib axes object
        The axes object that contains the plot.
    """
    import matplotlib.pyplot as plt
    if axes is not None:
        axes.matshow(x, cmap=cmap)
    else:
        axes = plt.matshow(x, cmap=cmap).get_axes()
    axes.set_xticks(np.arange(1, x.shape[1]) - .5)
    axes.set_yticks(np.arange(1, x.shape[0]) - .5)
    if border_color is None:
        border_color = 'black'
    axes.grid(linestyle="-", linewidth=linewidth, color=border_color)
    axes.set_xticklabels(())
    axes.set_yticklabels(())
    return axes
