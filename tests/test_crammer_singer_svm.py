
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_greater, assert_equal

from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score

from pystruct.problems import CrammerSingerSVMProblem
from pystruct.learners import (OneSlackSSVM, StructuredSVM,
                               SubgradientStructuredSVM)

def test_crammer_singer_problem():
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3)

    # test inference energy
    rng = np.random.RandomState(0)
    w = rng.uniform(size=pbl.size_psi)
    x = X[0]
    y, energy = pbl.inference(x, w, return_energy=True)
    assert_equal(energy, np.dot(w, pbl.psi(x, y)))

    # test inference_result:
    energies = [np.dot(w, pbl.psi(x, y_hat)) for y_hat in xrange(3)]
    assert_equal(np.argmax(energies), y)

    # test loss_augmented inference energy
    y, energy = pbl.loss_augmented_inference(x, Y[0], w, return_energy=True)
    assert_equal(energy, np.dot(w, pbl.psi(x, y)) + pbl.loss(Y[0], y))

    # test batch versions
    Y_batch = pbl.batch_inference(X, w)
    Y_ = [pbl.inference(x, w) for x in X]
    assert_array_equal(Y_batch, Y_)

    Y_batch = pbl.batch_loss_augmented_inference(X, Y, w)
    Y_ = [pbl.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]
    assert_array_equal(Y_batch, Y_)

    loss_batch = pbl.batch_loss(Y, Y_)
    loss = [pbl.loss(y, y_) for y, y_ in zip(Y, Y_)]
    assert_array_equal(loss_batch, loss)

def test_crammer_singer_problem_class_weight():
    X, Y = make_blobs(n_samples=80, centers=3, random_state=42)
    # we have to add a constant 1 feature by hand :-/
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    pbl = CrammerSingerSVMProblem(n_features=3, n_classes=3,
                                  class_weight=[1, 2, 1])

    rng = np.random.RandomState(0)
    w = rng.uniform(size=pbl.size_psi)
    # test inference energy
    x = X[0]
    y, energy = pbl.inference(x, w, return_energy=True)
    assert_equal(energy, np.dot(w, pbl.psi(x, y)))

    # test inference_result:
    energies = [np.dot(w, pbl.psi(x, y_hat)) for y_hat in xrange(3)]
    assert_equal(np.argmax(energies), y)

    # test loss_augmented inference energy
    y, energy = pbl.loss_augmented_inference(x, Y[0], w, return_energy=True)
    assert_equal(energy, np.dot(w, pbl.psi(x, y)) + pbl.loss(Y[0], y))

    # test batch versions
    Y_batch = pbl.batch_inference(X, w)
    Y_ = [pbl.inference(x, w) for x in X]
    assert_array_equal(Y_batch, Y_)

    Y_batch = pbl.batch_loss_augmented_inference(X, Y, w)
    Y_ = [pbl.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]
    assert_array_equal(Y_batch, Y_)

    loss_batch = pbl.batch_loss(Y, Y_)
    loss = [pbl.loss(y, y_) for y, y_ in zip(Y, Y_)]
    assert_array_equal(loss_batch, loss)


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
    weights *= len(weights) / np.sum(weights)
    pbl_class_weight = CrammerSingerSVMProblem(n_features=3, n_classes=3,
                                               class_weight=weights)
    svm_class_weight = OneSlackSSVM(pbl_class_weight, verbose=10, C=10)
    svm_class_weight.fit(X, Y)

    assert_greater(f1_score(Y, svm_class_weight.predict(X)),
                   f1_score(Y, svm.predict(X)))
