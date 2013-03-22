
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_greater

from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score

from pystruct.problems import CrammerSingerSVMProblem
from pystruct.learners import (OneSlackSSVM, StructuredSVM,
                               SubgradientStructuredSVM)


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
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)
    svm = StructuredSVM(pbl, verbose=10, check_constraints=True, C=1000,
                        batch_size=1)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_one_slack():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)
    svm = OneSlackSSVM(pbl, verbose=10, check_constraints=True, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_subgradient():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)
    svm = SubgradientStructuredSVM(pbl, verbose=10, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_equal_class_weights():
    # test that equal class weight is the same as no class weight
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)
    svm = OneSlackSSVM(pbl, verbose=10, C=10)

    svm.fit(X_train, Y_train)
    predict_no_class_weight = svm.predict(X_test)

    pbl_class_weight = CrammerSingerSVMProblem(n_features=3, n_classes=3,
                                               class_weight=np.ones(3))
    svm_class_weight = OneSlackSSVM(pbl_class_weight, verbose=10, C=10)
    svm_class_weight.fit(X_train, Y_train)
    predict_class_weight = svm_class_weight.predict(X_test)

    assert_array_equal(predict_no_class_weight, predict_class_weight)
    assert_array_almost_equal(svm.w, svm_class_weight.w)


def test_class_weights():
    # test that equal class weight is the same as no class weight
    X, Y = make_blobs(n_samples=210, centers=3, random_state=1, cluster_std=3,
                      shuffle=False)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X, Y = X[:170], Y[:170]

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)
    svm = OneSlackSSVM(pbl, verbose=10, C=10)

    svm.fit(X, Y)

    weights = 1. / np.bincount(Y)
    weights /= np.sum(weights)
    pbl_class_weight = CrammerSingerSVMProblem(n_features=3, n_classes=3,
                                               class_weight=weights)
    svm_class_weight = OneSlackSSVM(pbl_class_weight, verbose=10, C=10)
    svm_class_weight.fit(X, Y)

    assert_greater(f1_score(Y, svm_class_weight.predict(X)),
                   f1_score(Y, svm.predict(X)))
