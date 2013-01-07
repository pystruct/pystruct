import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true

from pystruct.problems import BinarySVMProblem
from pystruct.learners import StructuredSVM

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

    #tracer()


def test_simple_1d_dataset_cutting_plane():
    # 10 1d datapoints between 0 and 1
    X = np.random.uniform(size=30)
    # linearly separable labels
    Y = 1 - 2 * (X < .5)

    pbl = BinarySVMProblem(n_features=1)
    svm = StructuredSVM(pbl, verbose=10, check_constraints=True, C=1000)
    svm.fit(X, Y)
    tracer()
    assert_array_equal(Y, np.hstack(svm.predict(X)))
