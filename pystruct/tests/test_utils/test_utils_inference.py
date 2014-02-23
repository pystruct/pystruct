import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_almost_equal

from pystruct.utils import compress_sym, expand_sym
from pystruct.datasets import generate_blocks_multinomial
from pystruct.models import GridCRF
from pystruct.learners import (NSlackSSVM, OneSlackSSVM, SubgradientSSVM,
                               FrankWolfeSSVM)
from pystruct.inference import get_installed
from pystruct.utils import objective_primal

# we always try to get the fastest installed inference method
inference_method = get_installed(["qpbo", "ad3", "lp"])[0]


def test_symmetric_tools_symmetric():
    rnd = np.random.RandomState(0)
    # generate random symmetric matrix
    for size in [4, 6, 11]:
        x = rnd.normal(size=(size, size))
        x = x + x.T

        compressed = compress_sym(x, make_symmetric=False)
        assert_equal(compressed.shape, (size * (size + 1) / 2, ))

        uncompressed = expand_sym(compressed)
        assert_array_equal(x, uncompressed)


def test_symmetric_tools_upper():
    rnd = np.random.RandomState(0)
    # generate random matrix with only upper triangle.
    # expected result is full symmetric matrix
    for size in [4, 6, 11]:
        x = rnd.normal(size=(size, size))
        x = x + x.T
        x_ = x.copy()
        x[np.tri(size, k=-1, dtype=np.bool)] = 0

        compressed = compress_sym(x, make_symmetric=True)
        assert_equal(compressed.shape, (size * (size + 1) / 2, ))

        uncompressed = expand_sym(compressed)
        assert_array_equal(x_, uncompressed)


def test_ssvm_objectives():
    # test that the algorithms provide consistent objective curves.
    # this is not that strong a test now but at least makes sure that
    # the objective function is called.
    X, Y = generate_blocks_multinomial(n_samples=10, noise=1.5, seed=0)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    # once for n-slack
    clf = NSlackSSVM(model=crf, max_iter=5, C=1, tol=.1)
    clf.fit(X, Y)
    primal_objective = objective_primal(clf.model, clf.w, X, Y, clf.C)
    assert_almost_equal(clf.primal_objective_curve_[-1], primal_objective)

    # once for one-slack
    clf = OneSlackSSVM(model=crf, max_iter=5, C=1, tol=.1)
    clf.fit(X, Y)
    primal_objective = objective_primal(clf.model, clf.w, X, Y, clf.C,
                                        variant='one_slack')
    assert_almost_equal(clf.primal_objective_curve_[-1], primal_objective)

    # now subgradient. Should also work in batch-mode.
    clf = SubgradientSSVM(model=crf, max_iter=5, C=1, batch_size=-1)
    clf.fit(X, Y)
    primal_objective = objective_primal(clf.model, clf.w, X, Y, clf.C)
    assert_almost_equal(clf.objective_curve_[-1], primal_objective)

    # frank wolfe
    clf = FrankWolfeSSVM(model=crf, max_iter=5, C=1, batch_mode=True)
    clf.fit(X, Y)
    primal_objective = objective_primal(clf.model, clf.w, X, Y, clf.C)
    assert_almost_equal(clf.primal_objective_curve_[-1], primal_objective)
    # block-coordinate Frank-Wolfe
    clf = FrankWolfeSSVM(model=crf, max_iter=5, C=1, batch_mode=False)
    clf.fit(X, Y)
    primal_objective = objective_primal(clf.model, clf.w, X, Y, clf.C)
    assert_almost_equal(clf.primal_objective_curve_[-1], primal_objective)
