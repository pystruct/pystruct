import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true

from sklearn.datasets import make_blobs

from pystruct.models import BinaryClf
from pystruct.learners import (NSlackSSVM, SubgradientSSVM,
                               OneSlackSSVM)


def test_model_1d():
    # 10 1d datapoints between -1 and 1
    np.random.seed(0)
    X = np.random.uniform(size=(10, 1))
    # linearly separable labels
    Y = 1 - 2 * (X.ravel() < .5)
    pbl = BinaryClf(n_features=2)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    w = [1, -.5]
    Y_pred = np.hstack([pbl.inference(x, w) for x in X])
    assert_array_equal(Y, Y_pred)

    # check that sign of joint_feature and inference agree
    for x, y in zip(X, Y):
        assert_true(np.dot(w, pbl.joint_feature(x, y)) > np.dot(w, pbl.joint_feature(x, -y)))

    # check that sign of joint_feature and the sign of y correspond
    for x, y in zip(X, Y):
        assert_true(np.dot(w, pbl.joint_feature(x, y)) == -np.dot(w, pbl.joint_feature(x, -y)))


def test_simple_1d_dataset_cutting_plane():
    # 10 1d datapoints between 0 and 1
    X = np.random.uniform(size=(30, 1))
    # linearly separable labels
    Y = 1 - 2 * (X.ravel() < .5)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    pbl = BinaryClf(n_features=2)
    svm = NSlackSSVM(pbl, check_constraints=True, C=1000)
    svm.fit(X, Y)
    assert_array_equal(Y, np.hstack(svm.predict(X)))


def test_blobs_2d_cutting_plane():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = BinaryClf(n_features=3)
    svm = NSlackSSVM(pbl, check_constraints=True, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_subgradient():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = BinaryClf(n_features=3)
    svm = SubgradientSSVM(pbl, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_2d_one_slack():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, X_test, Y_train, Y_test = X[:40], X[40:], Y[:40], Y[40:]

    pbl = BinaryClf(n_features=3)
    svm = OneSlackSSVM(pbl, C=1000)

    svm.fit(X_train, Y_train)
    assert_array_equal(Y_test, np.hstack(svm.predict(X_test)))


def test_blobs_batch():
    # make two gaussian blobs
    X, Y = make_blobs(n_samples=80, centers=2, random_state=1)
    Y = 2 * Y - 1

    pbl = BinaryClf(n_features=2)

    # test joint_feature
    joint_feature_mean = pbl.batch_joint_feature(X, Y)
    joint_feature_mean2 = np.sum([pbl.joint_feature(x, y) for x, y in zip(X, Y)], axis=0)
    assert_array_equal(joint_feature_mean, joint_feature_mean2)

    # test inference
    w = np.random.uniform(-1, 1, size=pbl.size_joint_feature)
    Y_hat = pbl.batch_inference(X, w)
    for i, (x, y_hat) in enumerate(zip(X, Y_hat)):
        assert_array_equal(Y_hat[i], pbl.inference(x, w))

    # test inference
    Y_hat = pbl.batch_loss_augmented_inference(X, Y, w)
    for i, (x, y, y_hat) in enumerate(zip(X, Y, Y_hat)):
        assert_array_equal(Y_hat[i], pbl.loss_augmented_inference(x, y, w))


def test_break_ties():
    pbl = BinaryClf(n_features=2)
    X = np.array([[-1., -1.], [-1., 1.], [1., 1.]])
    w = np.array([1., 1.])
    assert_array_equal(pbl.batch_inference(X, w), np.array([-1, 1, 1]))
