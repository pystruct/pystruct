import numpy as np
from numpy.testing import assert_array_equal
from tempfile import mkstemp
from nose.tools import assert_true, assert_equal, assert_less, assert_greater

from sklearn.datasets import load_digits, load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GridCRF, GraphCRF, BinaryClf
from pystruct.learners import OneSlackSSVM
from pystruct.datasets import (generate_blocks_multinomial, generate_blocks,
                               generate_checker)
from pystruct.utils import make_grid_edges, SaveLogger
from pystruct.inference import get_installed

# we always try to get the fastest installed inference method
inference_method = get_installed(["qpbo", "ad3", "lp"])[0]


def test_multinomial_blocks_one_slack():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.5, seed=0)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = OneSlackSSVM(model=crf, max_iter=150, C=1,
                       check_constraints=True, break_on_bad=True, tol=.1,
                       inference_cache=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_svm_as_crf_pickling():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')
    logger = SaveLogger(file_name)
    svm = OneSlackSSVM(pbl, check_constraints=True, C=1, n_jobs=1,
                       logger=logger)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))


def test_constraint_removal():
    digits = load_digits()
    X, y = digits.data, digits.target
    y = 2 * (y % 2) - 1  # even vs odd as +1 vs -1
    X = X / 16.
    pbl = BinaryClf(n_features=X.shape[1])
    clf_no_removal = OneSlackSSVM(model=pbl, max_iter=500, C=1,
                                  inactive_window=0, tol=0.01)
    clf_no_removal.fit(X, y)
    clf = OneSlackSSVM(model=pbl, max_iter=500, C=1, tol=0.01,
                       inactive_threshold=1e-8)
    clf.fit(X, y)
    # check that we learned something
    assert_greater(clf.score(X, y), .92)

    # results are mostly equal
    # if we decrease tol, they will get more similar
    assert_less(np.mean(clf.predict(X) != clf_no_removal.predict(X)), 0.02)

    # without removal, have as many constraints as iterations
    assert_equal(len(clf_no_removal.objective_curve_),
                 len(clf_no_removal.constraints_))

    # with removal, there are less constraints than iterations
    assert_less(len(clf.constraints_),
                len(clf.objective_curve_))


def test_binary_blocks_one_slack_graph():
    #testing cutting plane ssvm on easy binary dataset
    # generate graphs explicitly for each example
    X, Y = generate_blocks(n_samples=3)
    crf = GraphCRF(inference_method=inference_method)
    clf = OneSlackSSVM(model=crf, max_iter=100, C=1,
                       check_constraints=True, break_on_bad=True,
                       n_jobs=1, tol=.1)
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


def test_one_slack_constraint_caching():
    # testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.5, seed=0,
                                       size_x=9)
    n_labels = len(np.unique(Y))
    exact_inference = get_installed([('ad3', {'branch_and_bound': True}), "lp"])[0]
    crf = GridCRF(n_states=n_labels, inference_method=exact_inference)
    clf = OneSlackSSVM(model=crf, max_iter=150, C=1,
                       check_constraints=True, break_on_bad=True,
                       inference_cache=50, inactive_window=0)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    assert_equal(len(clf.inference_cache_), len(X))
    # there should be 13 constraints, which are less than the 94 iterations
    # that are done
    # check that we didn't change the behavior of how we construct the cache
    constraints_per_sample = [len(cache) for cache in clf.inference_cache_]
    if exact_inference == "lp":
        assert_equal(len(clf.inference_cache_[0]), 18)
        assert_equal(np.max(constraints_per_sample), 18)
        assert_equal(np.min(constraints_per_sample), 18)
    else:
        assert_equal(len(clf.inference_cache_[0]), 13)
        assert_equal(np.max(constraints_per_sample), 20)
        assert_equal(np.min(constraints_per_sample), 11)


def test_one_slack_attractive_potentials():
    # test that submodular SSVM can learn the block dataset
    X, Y = generate_blocks(n_samples=10)
    crf = GridCRF(inference_method=inference_method)
    submodular_clf = OneSlackSSVM(model=crf, max_iter=200, C=1,
                                  check_constraints=True,
                                  negativity_constraint=[5],
                                  inference_cache=50)
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    assert_array_equal(Y, Y_pred)
    assert_true(submodular_clf.w[5] < 0)


def test_one_slack_repellent_potentials():
    # test non-submodular problem with and without submodularity constraint
    # dataset is checkerboard
    X, Y = generate_checker()
    crf = GridCRF(inference_method=inference_method)
    clf = OneSlackSSVM(model=crf, max_iter=10, C=.01,
                       check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    # standard crf can predict perfectly
    assert_array_equal(Y, Y_pred)

    submodular_clf = OneSlackSSVM(model=crf, max_iter=10, C=.01,
                                  check_constraints=True,
                                  negativity_constraint=[4, 5, 6])
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    assert_less(submodular_clf.score(X, Y), .99)
    # submodular crf can not do better than unaries
    for i, x in enumerate(X):
        y_pred_unaries = crf.inference(x, np.array([1, 0, 0, 1, 0, 0, 0]))
        assert_array_equal(y_pred_unaries, Y_pred[i])


def test_switch_to_ad3():
    # test if switching between qpbo and ad3 works

    if not get_installed(['qpbo']) or not get_installed(['ad3']):
        return
    X, Y = generate_blocks_multinomial(n_samples=5, noise=1.5, seed=0)
    crf = GridCRF(n_states=3, inference_method='qpbo')

    ssvm = OneSlackSSVM(crf, inference_cache=50, max_iter=10000)

    ssvm_with_switch = OneSlackSSVM(crf, inference_cache=50, max_iter=10000,
                                    switch_to=('ad3'))
    ssvm.fit(X, Y)
    ssvm_with_switch.fit(X, Y)
    assert_equal(ssvm_with_switch.model.inference_method, 'ad3')
    # we check that the dual is higher with ad3 inference
    # as it might use the relaxation, that is pretty much guraranteed
    assert_greater(ssvm_with_switch.objective_curve_[-1],
                   ssvm.objective_curve_[-1])
