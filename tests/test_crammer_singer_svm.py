
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import make_blobs

from pystruct.problems import CrammerSingerSVMProblem
from pystruct.learners import StructuredSVM, SubgradientStructuredSVM

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_simple_1d_dataset_cutting_plane():
    # 10 1d datapoints between 0 and 1
    X = np.random.uniform(size=(30, 1))
    Y = (X.ravel() > 0.5).astype(np.int)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    pbl = CrammerSingerSVMProblem(n_features=2)
    svm = StructuredSVM(pbl, verbose=10, check_constraints=True, C=10000)
    svm.fit(X, Y)
    assert_array_equal(Y, np.hstack(svm.predict(X)))


def test_blobs_2d_cutting_plane():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3)
    svm = StructuredSVM(pbl, verbose=10, check_constraints=True, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_subgradient():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3)
    svm = SubgradientStructuredSVM(pbl, verbose=10,
                                   C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))
