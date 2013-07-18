from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_less

from sklearn.datasets import load_iris

from pystruct.models import GridCRF, GraphCRF
from pystruct.learners import SubgradientSSVM
import pystruct.toy_datasets as toy
from pystruct.utils import SaveLogger, train_test_split


def test_multinomial_blocks_subgradient():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3,
                                           seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels)
    clf = SubgradientSSVM(model=crf, max_iter=50, C=10, momentum=.98,
                          learning_rate=0.001)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_checker_subgradient():
    X, Y = toy.generate_checker_multinomial(n_samples=10, noise=0.0)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels)
    clf = SubgradientSSVM(model=crf, max_iter=50, C=10,
                          momentum=.98, learning_rate=0.01)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_parallel():
    # fixme: travis doesn't like parallelism?
    pass
    #testing subgradient ssvm on easy binary dataset
    #X, Y = toy.generate_blocks(n_samples=10)
    #crf = GridCRF()
    #clf = SubgradientSSVM(model=crf, max_iter=100, C=1,
                          #momentum=.0, learning_rate=0.1, n_jobs=-1)
    #clf.fit(X, Y)
    #Y_pred = clf.predict(X)
    #assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_online():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=200, C=10, momentum=.0,
                          learning_rate=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_subgradient_svm_as_crf_pickling():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')
    logger = SaveLogger(file_name)
    svm = SubgradientSSVM(pbl, C=10, n_jobs=1, logger=logger,
                          max_iter=50, momentum=0, learning_rate=0.01)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))
