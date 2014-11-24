import numpy as np
from itertools import product

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_greater, assert_true
from pystruct.models import GridCRF, BinaryClf
from pystruct.learners import StructuredPerceptron
from pystruct.datasets import generate_blocks, generate_blocks_multinomial


def test_binary_blocks():
    X, Y = generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=40)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks():
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    crf = GridCRF(n_states=X.shape[-1])
    clf = StructuredPerceptron(model=crf, max_iter=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_xor():
    """Test perceptron behaviour against hand-computed values for XOR"""
    X = np.array([[a, b, 1] for a in (-1, 1) for b in (-1, 1)], dtype=np.float)
    Y = np.array([-1, 1, 1, -1])
    # Should cycle weight vectors (1, 1, -1), (0, 2, 0), (1, 1, 1), (0, 0, 0)
    # but this depends on how ties are settled.  Maybe the test can be
    # made robust to this
    # Batch version should cycle (0, 0, -2), (0, 0, 0)

    expected_predictions = [
        np.array([1, 1, 1, 1]),      # online, no average, w = (0, 0, 0, 0)
        np.array([-1, 1, -1, 1]),    # online, average, w ~= (0.5, 1, 0)
        np.array([1, 1, 1, 1]),      # batch, no average, w = (0, 0, 0)
        np.array([-1, -1, -1, -1])   # batch, average, w ~= (0, 0, -2)
    ]
    pcp = StructuredPerceptron(model=BinaryClf(n_features=3), max_iter=2)
    for pred, (batch, average) in zip(expected_predictions,
                                      product((False, True), (False, True))):
        pcp.set_params(batch=batch, average=average)
        pcp.fit(X, Y)
        # We don't compare w explicitly but its prediction.  As the perceptron
        # is invariant to the scaling of w, this will allow the optimization of
        # the underlying implementation
        assert_array_equal(pcp.predict(X), pred)


def test_partial_averaging():
    """Use XOR weight cycling to test partial averaging"""
    X = np.array([[a, b, 1] for a in (-1, 1) for b in (-1, 1)], dtype=np.float)
    Y = np.array([-1, 1, 1, -1])
    pcp = StructuredPerceptron(model=BinaryClf(n_features=3), max_iter=5,
                               decay_exponent=1, decay_t0=1)
    weight = {}
    for average in (0, 1, 4, -1):
        pcp.set_params(average=average)
        pcp.fit(X, Y)
        weight[average] = pcp.w
    assert_array_equal(weight[4], weight[-1])
    assert_array_almost_equal(weight[0], [1.5, 3, 0])
    assert_array_almost_equal(weight[1], [1.75, 3.5, 0])
    assert_array_almost_equal(weight[4], [2.5, 5, 0])


def test_averaging_early_stopping():
    """Test averaging over final epoch when early stopping"""
    # we use logical OR, an easy problem solved after the second epoch
    X = np.array([[a, b, 1] for a in (-1, 1) for b in (-1, 1)], dtype=np.float)
    Y = np.array([-1, 1, 1, 1])
    pcp = StructuredPerceptron(model=BinaryClf(n_features=3), max_iter=3,
                               average=-1)
    pcp.fit(X, Y)
    # The exact weight is used without the influence of the early iterations
    assert_array_equal(pcp.w, [1, 1, 1])

    # If we were expecting 3 iterations, we would end up with a zero vector
    pcp.set_params(average=2)
    pcp.fit(X, Y)
    assert_array_equal(pcp.w, [0, 0, 0])


def test_overflow_averaged():
    X = np.array([[np.finfo('d').max]])
    Y = np.array([-1])
    pcp = StructuredPerceptron(model=BinaryClf(n_features=1),
                               max_iter=2, average=True)
    pcp.fit(X, Y)
    assert_true(np.isfinite(pcp.w[0]))


def test_averaged():
    # Under a lot of noise, averaging helps.  This fails with less noise.
    X, Y = generate_blocks_multinomial(n_samples=15, noise=3, seed=0)
    X_train, Y_train = X[:10], Y[:10]
    X_test, Y_test = X[10:], Y[10:]
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=3)
    clf.fit(X_train, Y_train)
    no_avg_test = clf.score(X_test, Y_test)
    clf.set_params(average=True)
    clf.fit(X_train, Y_train)
    avg_test = clf.score(X_test, Y_test)
    assert_greater(avg_test, no_avg_test)
