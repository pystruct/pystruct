import numpy as np
from tempfile import mkstemp

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GraphCRF
from pystruct.learners import NSlackSSVM
from pystruct.utils import SaveLogger
from pystruct.inference import get_installed

from nose.tools import assert_less, assert_almost_equal

# we always try to get the fastest installed inference method
inference_method = get_installed(["qpbo", "ad3", "max-product", "lp"])[0]


def test_logging():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method=inference_method)
    logger = SaveLogger(file_name)
    svm = NSlackSSVM(pbl, C=100, n_jobs=1, logger=logger)
    svm.fit(X_train, y_train)

    score_current = svm.score(X_test, y_test)
    score_auto_saved = logger.load().score(X_test, y_test)

    alt_file_name = file_name + "alt"
    logger.save(svm, alt_file_name)
    logger.file_name = alt_file_name
    logger.load()
    score_manual_saved = logger.load().score(X_test, y_test)

    assert_less(.97, score_current)
    assert_less(.97, score_auto_saved)
    assert_less(.97, score_manual_saved)
    assert_almost_equal(score_auto_saved, score_manual_saved)
