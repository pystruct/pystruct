from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_less

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GridCRF, GraphCRF
from pystruct.datasets import generate_blocks_multinomial
from pystruct.learners import FrankWolfeSSVM
from pystruct.utils import SaveLogger


def test_multinomial_blocks_frankwolfe():
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.5, seed=0)
    crf = GridCRF(inference_method='qpbo')
    clf = FrankWolfeSSVM(model=crf, C=1, max_iter=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks_frankwolfe_batch():
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    crf = GridCRF(inference_method='qpbo')
    clf = FrankWolfeSSVM(model=crf, C=1, max_iter=500, batch_mode=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_svm_as_crf_pickling_bcfw():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')
    logger = SaveLogger(file_name)
    svm = FrankWolfeSSVM(pbl, C=10, logger=logger, max_iter=50)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))


def test_svm_as_crf_pickling_batch():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')
    logger = SaveLogger(file_name)
    svm = FrankWolfeSSVM(pbl, C=10, logger=logger, max_iter=50, batch_mode=False)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))
