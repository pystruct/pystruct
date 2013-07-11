import matplotlib.pyplot as plt
import numpy as np
from simple_inference import simple_grid, general_graph


def example_binary():
    # generate trivial data
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

    # create unaries
    unaries = x_noisy
    # as we convert to int, we need to multipy to get sensible values
    unaries = np.dstack([-unaries, unaries])
    # create potts pairwise
    pairwise = np.eye(2)

    # do simple cut
    result = np.argmax(simple_grid(unaries, pairwise)[0], axis=-1)

    # use the gerneral graph algorithm
    # first, we construct the grid graph
    inds = np.arange(x.size).reshape(x.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])

    # we flatten the unaries
    pairwise_per_edge = np.repeat(pairwise[np.newaxis, :, :], edges.shape[0],
                                  axis=0)
    result_graph = np.argmax(general_graph(unaries.reshape(-1, 2), edges,
                                           pairwise_per_edge)[0], axis=-1)

    # plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="cut_simple")
    plt.imshow(result, interpolation='nearest')
    plt.subplot(236, title="cut_from_graph")
    plt.imshow(result_graph.reshape(x.shape), interpolation='nearest')

    plt.show()


def example_multinomial():
    # generate dataset with three stripes
    np.random.seed(4)
    x = np.zeros((10, 12, 3))
    x[:, :4, 0] = 1
    x[:, 4:8, 1] = 1
    x[:, 8:, 2] = 1
    unaries = x + 1.5 * np.random.normal(size=x.shape)
    x = np.argmax(x, axis=2)
    unaries = unaries
    x_thresh = np.argmax(unaries, axis=2)

    # potts potential
    pairwise_potts = 2 * np.eye(3)
    result = np.argmax(simple_grid(unaries, pairwise_potts)[0], axis=-1)
    # potential that penalizes 0-1 and 1-2 less than 0-2
    pairwise_1d = 2 * np.eye(3) + 2
    pairwise_1d[-1, 0] = 0
    pairwise_1d[0, -1] = 0
    print(pairwise_1d)
    result_1d = np.argmax(simple_grid(unaries, pairwise_1d)[0], axis=-1)
    plt.subplot(141, title="original")
    plt.imshow(x, interpolation="nearest")
    plt.subplot(142, title="thresholded unaries")
    plt.imshow(x_thresh, interpolation="nearest")
    plt.subplot(143, title="potts potentials")
    plt.imshow(result, interpolation="nearest")
    plt.subplot(144, title="1d topology potentials")
    plt.imshow(result_1d, interpolation="nearest")
    plt.show()


#example_binary()
example_multinomial()
