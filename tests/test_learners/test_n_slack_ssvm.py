import numpy as np
from tempfile import mkstemp

from sklearn.datasets import load_iris

from pystruct.models import GraphCRF
from pystruct.learners import NSlackSSVM
from pystruct.utils import SaveLogger, train_test_split
import pystruct.toy_datasets as toy
from pystruct.models import GridCRF, DirectionalGridCRF
from pystruct.inference import get_installed

from nose.tools import assert_less
from numpy.testing import assert_array_equal


def test_n_slack_svm_as_crf_pickling():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='lp')
    logger = SaveLogger(file_name)
    svm = NSlackSSVM(pbl, C=100, n_jobs=1, logger=logger)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))


def test_multinomial_blocks_cutting_plane():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy.generate_blocks_multinomial(n_samples=40, noise=0.5, seed=0)
    n_labels = len(np.unique(Y))
    for inference_method in get_installed(['ad3']):
        crf = GridCRF(n_states=n_labels, inference_method=inference_method)
        clf = NSlackSSVM(model=crf, max_iter=100, C=100, verbose=3,
                         check_constraints=False, batch_size=1)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)


def test_multinomial_blocks_directional():
    # testing cutting plane ssvm with directional CRF on easy multinomial
    # dataset
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    n_labels = len(np.unique(Y))
    for inference_method in get_installed(['ad3']):
        crf = DirectionalGridCRF(n_states=n_labels,
                                 inference_method=inference_method)
        clf = NSlackSSVM(model=crf, max_iter=100, C=100, verbose=3,
                         check_constraints=False, batch_size=1)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)


def test_multinomial_checker_cutting_plane():
    X, Y = toy.generate_checker_multinomial(n_samples=10, noise=.1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels)
    clf = NSlackSSVM(model=crf, max_iter=20, C=100000, check_constraints=True,
                     verbose=3)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
