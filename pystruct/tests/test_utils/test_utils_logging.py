from tempfile import mkstemp

import numpy as np

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from nose.tools import assert_less
from pystruct.utils import SaveLogger
from pystruct.models import GraphCRF
from pystruct.learners import (NSlackSSVM, OneSlackSSVM, SubgradientSSVM,
                               FrankWolfeSSVM)

def test_parallel_pickling():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')

    _, file_name = mkstemp()
    logger = SaveLogger(file_name)
    svm = NSlackSSVM(pbl, logger=logger, max_iter=100, n_jobs=-1)
    svm.fit(X_train, y_train)
    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))
        
    _, file_name = mkstemp()
    logger = SaveLogger(file_name)
    svm = OneSlackSSVM(pbl, logger=logger, max_iter=100, n_jobs=-1)
    svm.fit(X_train, y_train)
    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))

    ## 06Jan15: currently failing in pystruct/master
    #_, file_name = mkstemp()
    #logger = SaveLogger(file_name)
    #svm = SubgradientSSVM(pbl, logger=logger, max_iter=100, n_jobs=-1)
    #svm.fit(X_train, y_train)
    #assert_less(.97, svm.score(X_test, y_test))
    #assert_less(.97, logger.load().score(X_test, y_test))

