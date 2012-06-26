import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from pyqpbo import alpha_expansion_graph

#from crf import MultinomialFixedGraphCRFNoBias
from crf import MultinomialFixedGraphCRFNoBias
#from crf import MultinomialGridCRF
#from structured_perceptron import StructuredPerceptron
from structured_svm import StructuredSVM
from toy_datasets import make_dataset_big_checker


from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    X, Y = make_dataset_big_checker(n_samples=1)
    size_y = Y[0].size
    shape_y = Y[0].shape
    n_labels = len(np.unique(Y))
    inds = np.arange(size_y).reshape(shape_y)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    downleft = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
    downright = np.c_[inds[:-1, 1:].ravel(), inds[1:, :-1].ravel()]
    edges = np.vstack([horz, vert, downleft, downright]).astype(np.int32)
    graph = sparse.coo_matrix((np.ones(edges.shape[0]),
        (edges[:, 0], edges[:, 1])), shape=(size_y, size_y)).tocsr()
    graph = graph + graph.T

    crf = MultinomialFixedGraphCRFNoBias(n_states=n_labels, graph=graph)
    #crf = MultinomialGridCRF(n_labels=4)
    #clf = StructuredPerceptron(problem=crf, max_iter=50)
    clf = StructuredSVM(problem=crf, max_iter=20, C=1000000, verbose=20,
            check_constraints=True,
            positive_constraint=np.arange(crf.size_psi - 1))
    #clf = SubgradientStructuredSVM(problem=crf, max_iter=100, C=10000)
    X_flat = [x.reshape(-1, n_labels).copy("C") for x in X]
    Y_flat = [y.ravel() for y in Y]
    clf.fit(X_flat, Y_flat)
    #clf.fit(X, Y)
    Y_pred = clf.predict(X_flat)
    #Y_pred = clf.predict(X)

    i = 0
    loss = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        y_pred = y_pred.reshape(x.shape[:2])
        #loss += np.sum(y != y_pred)
        loss += np.sum(np.logical_xor(y, y_pred))
        if i > 4:
            continue
        fig, plots = plt.subplots(1, 4)
        plots[0].imshow(y, interpolation='nearest')
        plots[0].set_title("gt")
        pw_z = np.zeros((n_labels, n_labels), dtype=np.int32)
        un = np.ascontiguousarray(
                -1000 * x.reshape(-1, n_labels)).astype(np.int32)
        unaries = alpha_expansion_graph(edges, un, pw_z)
        plots[1].imshow(unaries.reshape(y.shape), interpolation='nearest')
        plots[1].set_title("unaries only")
        plots[2].imshow(y_pred, interpolation='nearest')
        plots[2].set_title("prediction")
        loss_augmented = clf.problem.loss_augmented_inference(
                x.reshape(-1, n_labels), y.ravel(), clf.w)
        loss_augmented = loss_augmented.reshape(y.shape)
        plots[3].imshow(loss_augmented, interpolation='nearest')
        plots[3].set_title("loss augmented")
        fig.savefig("data_%03d.png" % i)
        plt.close(fig)
        i += 1
    print("loss: %f" % loss)

if __name__ == "__main__":
    main()
