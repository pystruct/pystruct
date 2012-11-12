import numpy as np
from numpy.testing import assert_array_equal
from crf import MultinomialGridCRF
from structured_svm import StructuredSVM, SubgradientStructuredSVM
import toy_datasets


def test_multinomial_blocks_cutting_plane():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy_datasets.generate_blocks_multinomial(n_samples=10, noise=0.3,
            seed=0)
    n_labels = len(np.unique(Y))
    crf = MultinomialGridCRF(n_states=n_labels)
    clf = StructuredSVM(problem=crf, max_iter=10, C=100, verbose=0,
            check_constraints=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks_subgradient():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = toy_datasets.generate_blocks_multinomial(n_samples=10, noise=0.3,
            seed=1)
    n_labels = len(np.unique(Y))
    crf = MultinomialGridCRF(n_states=n_labels)
    clf = SubgradientStructuredSVM(problem=crf, max_iter=50, C=10,
            verbose=0, momentum=.98, learningrate=0.001, plot=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_checker_cutting_plane():
    X, Y = toy_datasets.generate_checker_multinomial(n_samples=10, noise=0.0)
    n_labels = len(np.unique(Y))
    crf = MultinomialGridCRF(n_states=n_labels)
    clf = StructuredSVM(problem=crf, max_iter=20, C=100000, verbose=20,
            check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_checker_subgradient():
    X, Y = toy_datasets.generate_checker_multinomial(n_samples=10, noise=0.0)
    n_labels = len(np.unique(Y))
    crf = MultinomialGridCRF(n_states=n_labels)
    clf = SubgradientStructuredSVM(problem=crf, max_iter=50, C=10,
            verbose=10, momentum=.98, learningrate=0.01, plot=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
