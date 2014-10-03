from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_less

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GridCRF, GraphCRF
from pystruct.learners import SubgradientSSVM
from pystruct.inference import get_installed
from pystruct.datasets import (generate_blocks_multinomial,
                               generate_checker_multinomial, generate_blocks)
from pystruct.utils import SaveLogger


inference_method = get_installed(["qpbo", "ad3", "lp"])[0]


def test_multinomial_blocks_subgradient():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.6, seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_checker_subgradient():
    X, Y = generate_checker_multinomial(n_samples=10, noise=0.4)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_parallel():
    # fixme: travis doesn't like parallelism?
    pass
    #testing subgradient ssvm on easy binary dataset
    #X, Y = generate_blocks(n_samples=10)
    #crf = GridCRF()
    #clf = SubgradientSSVM(model=crf, max_iter=100, C=1,
                          #momentum=.0, learning_rate=0.1, n_jobs=-1)
    #clf.fit(X, Y)
    #Y_pred = clf.predict(X)
    #assert_array_equal(Y, Y_pred)


def test_binary_blocks():
    #testing subgradient ssvm on easy binary dataset
    X, Y = generate_blocks(n_samples=5)
    crf = GridCRF(inference_method=inference_method)
    clf = SubgradientSSVM(model=crf)
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
    svm = SubgradientSSVM(pbl, logger=logger, max_iter=100)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))


def test_multinomial_blocks_subgradient_batch():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.6, seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=100, batch_size=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    
    clf2 = SubgradientSSVM(model=crf, max_iter=100, batch_size=len(X))
    clf2.fit(X, Y)
    Y_pred2 = clf2.predict(X)
    assert_array_equal(Y, Y_pred2)
    
