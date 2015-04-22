import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_greater

from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score

from pystruct.models import GraphCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM
from pystruct.datasets import generate_blocks
from pystruct.utils import make_grid_edges
from pystruct.inference import get_installed


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    # generate graphs explicitly for each example
    for inference_method in get_installed(["lp", "qpbo", "ad3", 'ogm']):
        X, Y = generate_blocks(n_samples=3)
        crf = GraphCRF(inference_method=inference_method)
        clf = NSlackSSVM(model=crf, max_iter=20, C=100, check_constraints=True,
                         break_on_bad=False, n_jobs=1)
        x1, x2, x3 = X
        y1, y2, y3 = Y
        n_states = len(np.unique(Y))
        # delete some rows to make it more fun
        x1, y1 = x1[:, :-1], y1[:, :-1]
        x2, y2 = x2[:-1], y2[:-1]
        # generate graphs
        X_ = [x1, x2, x3]
        G = [make_grid_edges(x) for x in X_]

        # reshape / flatten x and y
        X_ = [x.reshape(-1, n_states) for x in X_]
        Y = [y.ravel() for y in [y1, y2, y3]]

        X = list(zip(X_, G))

        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        for y, y_pred in zip(Y, Y_pred):
            assert_array_equal(y, y_pred)


def test_standard_svm_blobs_2d():
    # no edges, reduce to crammer-singer svm
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]
    X_train_graphs = [(x[np.newaxis, :], np.empty((0, 2), dtype=np.int))
                      for x in X_train]

    X_test_graphs = [(x[np.newaxis, :], np.empty((0, 2), dtype=np.int))
                     for x in X_test]

    pbl = GraphCRF(n_features=3, n_states=3, inference_method='unary')
    svm = OneSlackSSVM(pbl, check_constraints=True, C=1000)

    svm.fit(X_train_graphs, Y_train[:, np.newaxis])
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test_graphs)))


def test_standard_svm_blobs_2d_class_weight():
    # no edges, reduce to crammer-singer svm
    X, Y = make_blobs(n_samples=210, centers=3, random_state=1, cluster_std=3,
                      shuffle=False)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X, Y = X[:170], Y[:170]

    X_graphs = [(x[np.newaxis, :], np.empty((0, 2), dtype=np.int)) for x in X]

    pbl = GraphCRF(n_features=3, n_states=3, inference_method='unary')
    svm = OneSlackSSVM(pbl, check_constraints=False, C=1000)

    svm.fit(X_graphs, Y[:, np.newaxis])

    weights = 1. / np.bincount(Y)
    weights *= len(weights) / np.sum(weights)

    pbl_class_weight = GraphCRF(n_features=3, n_states=3, class_weight=weights,
                                inference_method='unary')
    svm_class_weight = OneSlackSSVM(pbl_class_weight, C=10,
                                    check_constraints=False,
                                    break_on_bad=False)
    svm_class_weight.fit(X_graphs, Y[:, np.newaxis])

    assert_greater(f1_score(Y, np.hstack(svm_class_weight.predict(X_graphs))),
                   f1_score(Y, np.hstack(svm.predict(X_graphs))))
