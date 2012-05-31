import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from examples_binary_grid_crf import make_dataset_big_checker

from latent_crf import LatentFixedGraphCRF
from structured_perceptron import LatentStructuredPerceptron

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    #X, Y = make_dataset_checker_multinomial()
    #X, Y = make_dataset_big_checker_extended()
    X, Y = make_dataset_big_checker()
    #X, Y = make_dataset_blocks_multinomial(n_samples=100)
    size_y = Y[0].size
    shape_y = Y[0].shape
    inds = np.arange(size_y).reshape(shape_y)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    downleft = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
    downright = np.c_[inds[:-1, 1:].ravel(), inds[1:, :-1].ravel()]
    edges = np.vstack([horz, vert, downleft, downright]).astype(np.int32)
    graph = sparse.coo_matrix((np.ones(edges.shape[0]),
        (edges[:, 0], edges[:, 1])), shape=(size_y, size_y)).tocsr()
    graph = graph + graph.T

    crf = LatentFixedGraphCRF(n_labels=2, n_states_per_label=3, graph=graph)
    clf = LatentStructuredPerceptron(problem=crf, max_iter=50)
    X_flat = [x.reshape(-1, 2) for x in X]
    Y_flat = [y.ravel() for y in Y]
    clf.fit(X_flat, Y_flat)
    #clf.fit(X, Y)
    Y_pred = clf.predict(X_flat)
    #Y_pred = clf.predict(X)

    i = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        plt.subplot(131)
        plt.imshow(y, interpolation='nearest')
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(np.argmin(x, axis=2), interpolation='nearest')
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(y_pred.reshape(x.shape[0], x.shape[1]),
                interpolation='nearest')
        plt.colorbar()
        plt.savefig("data_%03d.png" % i)
        plt.close()
        i += 1
        if i > 20:
            break

if __name__ == "__main__":
    main()
