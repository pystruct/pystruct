from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils.testing import assert_less

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GridCRF, GraphCRF
from pystruct.learners import SubgradientSSVM
import pystruct.toy_datasets as toy
from pystruct.utils import SaveLogger


def test_binary_blocks_subgradient_parallel():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=200, C=100, verbose=10,
                          momentum=.0, learning_rate=0.1, n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_online():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=200, C=100, verbose=10,
                          momentum=.0, learning_rate=0.1)
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

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='lp')
    logger = SaveLogger(file_name, verbose=1)
    svm = SubgradientSSVM(pbl, verbose=0, C=100, n_jobs=1, logger=logger,
                          max_iter=50, momentum=0, learning_rate=0.01)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))
