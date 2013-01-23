from numpy.testing import assert_array_equal
from pystruct.problems import GridCRF
from pystruct.learners import SubgradientStructuredSVM
import pystruct.toy_datasets as toy


def test_binary_blocks_subgradient_parallel():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientStructuredSVM(problem=crf, max_iter=200, C=100,
                                   verbose=10, momentum=.0, learning_rate=0.1,
                                   n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_oline():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientStructuredSVM(problem=crf, max_iter=200, C=100,
                                   verbose=10, momentum=.0, learning_rate=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
