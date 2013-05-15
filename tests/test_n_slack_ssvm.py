import numpy as np
from tempfile import mkstemp

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GraphCRF
from pystruct.learners import StructuredSVM
from pystruct.utils import SaveLogger

from sklearn.utils.testing import assert_less


def test_n_slack_svm_as_crf_pickling():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='lp')
    logger = SaveLogger(file_name, verbose=1)
    svm = StructuredSVM(pbl, verbose=0, C=100, n_jobs=1, logger=logger)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))
