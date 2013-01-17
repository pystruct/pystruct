import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true

from sklearn.datasets import make_blobs

from pystruct.problems import BinarySVMProblem
from pystruct.learners import StructuredSVM, SubgradientStructuredSVM

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_problem_1d():
    # 10 1d datapoints between -1 and 1
    X = np.random.uniform(size=10)
    # linearly separable labels
    Y = 1 - 2 * (X < .5)
    pbl = BinarySVMProblem(n_features=1)
    w = [1, -.5]
    Y_pred = np.hstack([pbl.inference(x, w) for x in X])
    assert_array_equal(Y, Y_pred)

    # check that sign of psi and inference agree
    for x, y in zip(X, Y):
        assert_true(np.dot(w, pbl.psi(x, y)) > np.dot(w, pbl.psi(x, -y)))

    # check that sign of psi and the sign of y correspond
    for x, y in zip(X, Y):
        assert_true(np.dot(w, pbl.psi(x, y)) == -np.dot(w, pbl.psi(x, -y)))


def test_simple_1d_dataset_cutting_plane():
    # 10 1d datapoints between 0 and 1
    X = np.random.uniform(size=30)
    # linearly separable labels
    Y = 1 - 2 * (X < .5)

    pbl = BinarySVMProblem(n_features=1)
    svm = StructuredSVM(pbl, verbose=3, check_constraints=True, C=1000)
    svm.fit(X, Y)
    assert_array_equal(Y, np.hstack(svm.predict(X)))


def test_blobs_2d_cutting_plane():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = BinarySVMProblem(n_features=2)
    svm = StructuredSVM(pbl, verbose=3, check_constraints=True, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_subgradient():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = BinarySVMProblem(n_features=2)
    svm = SubgradientStructuredSVM(pbl, verbose=3,
                                   C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))
