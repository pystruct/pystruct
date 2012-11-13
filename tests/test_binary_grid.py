import numpy as np
from numpy.testing import assert_array_equal
from crf import GridCRF
from structured_svm import StructuredSVM, SubgradientStructuredSVM
import toy_datasets


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    X, Y = toy_datasets.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredSVM(problem=crf, max_iter=20, C=100, verbose=0,
            check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy_datasets.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientStructuredSVM(problem=crf, max_iter=200, C=100,
            verbose=0, momentum=.0, learningrate=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_checker_subgradient():
    #testing subgradient ssvm on non-submodular binary dataset
    X, Y = toy_datasets.generate_checker(n_samples=10)
    crf = GridCRF()
    clf = SubgradientStructuredSVM(problem=crf, max_iter=100, C=100,
            verbose=0, momentum=.9, learningrate=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_ssvm_repellent_potentials():
    # test non-submodular learning with and without positivity constraint
    # dataset is checkerboard
    X, Y = toy_datasets.generate_checker()
    crf = GridCRF()
    clf = StructuredSVM(problem=crf, max_iter=200, C=100,
            verbose=0, check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    # standard crf can predict perfectly
    assert_array_equal(Y, Y_pred)

    submodular_clf = StructuredSVM(problem=crf, max_iter=200, C=100,
            verbose=0, check_constraints=True, positive_constraint=[2, 3, 4])
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    # submodular crf can not do better than unaries
    for i, x  in enumerate(X):
        y_pred_unaries = crf.inference(x, np.array([1, 1, 0, 0, 0]))
        assert_array_equal(y_pred_unaries, Y_pred[i])


def test_binary_ssvm_attractive_potentials():
    # test that submodular SSVM can learn the block dataset
    X, Y = toy_datasets.generate_blocks(n_samples=10)
    crf = GridCRF()
    submodular_clf = StructuredSVM(problem=crf, max_iter=200, C=100,
            verbose=0, check_constraints=True, positive_constraint=[1])
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    assert_array_equal(Y, Y_pred)
