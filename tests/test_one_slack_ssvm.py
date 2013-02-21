import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_less

from sklearn.datasets import load_digits

from pystruct.problems import GridCRF, GraphCRF, BinarySVMProblem
from pystruct.learners import OneSlackSSVM
import pystruct.toy_datasets as toy
from pystruct.utils import make_grid_edges


def test_multinomial_blocks_one_slack():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3,
                                           seed=0)
    n_labels = len(np.unique(Y))
    for inference_method in ['lp']:
        crf = GridCRF(n_states=n_labels, inference_method=inference_method)
        clf = OneSlackSSVM(problem=crf, max_iter=50, C=100, verbose=100,
                           check_constraints=True, break_on_bad=True)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)


def test_constraint_removal():
    digits = load_digits()
    X, y = digits.data, digits.target
    y = 2 * (y % 2) - 1  # even vs odd as +1 vs -1
    X = X / 16.
    pbl = BinarySVMProblem(n_features=X.shape[1])
    clf_no_removal = OneSlackSSVM(problem=pbl, max_iter=500, verbose=1, C=10,
                                  inactive_window=0, tol=0.01)
    clf_no_removal.fit(X, y)
    clf = OneSlackSSVM(problem=pbl, max_iter=500, verbose=1, C=10, tol=0.01,
                       inactive_threshold=1e-8)
    clf.fit(X, y)

    # results are mostly equal
    # if we decrease tol, they will get more similar
    assert_less(np.mean(clf.predict(X) != clf_no_removal.predict(X)), 0.015)

    # without removal, have as many constraints as iterations
    assert_equal(len(clf_no_removal.objective_curve_),
                 len(clf_no_removal.constraints_))

    # with removal, there are less constraints than iterations
    assert_less(len(clf.constraints_),
                len(clf.objective_curve_))


def test_binary_blocks_one_slack_graph():
    #testing cutting plane ssvm on easy binary dataset
    # generate graphs explicitly for each example
    for inference_method in ["dai", "lp"]:
        print("testing %s" % inference_method)
        X, Y = toy.generate_blocks(n_samples=3)
        crf = GraphCRF(inference_method=inference_method)
        clf = OneSlackSSVM(problem=crf, max_iter=100, C=100, verbose=100,
                           check_constraints=True, break_on_bad=True,
                           n_jobs=1)
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

        X = zip(X_, G)

        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        for y, y_pred in zip(Y, Y_pred):
            assert_array_equal(y, y_pred)


def test_one_slack_constraint_caching():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3,
                                           seed=0)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method='lp')
    clf = OneSlackSSVM(problem=crf, max_iter=50, C=100, verbose=100,
                       check_constraints=True, break_on_bad=True,
                       inference_cache=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    assert_equal(len(clf.inference_cache_), len(X))
    # there should be 9 constraints, which are less than the 16 iterations
    # that are done
    assert_equal(len(clf.inference_cache_[0]), 9)
    # check that we didn't change the behavior of how we construct the cache
    constraints_per_sample = [len(cache) for cache in clf.inference_cache_]
    assert_equal(np.max(constraints_per_sample), 10)
    assert_equal(np.min(constraints_per_sample), 8)
