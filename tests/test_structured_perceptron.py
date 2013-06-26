from numpy.testing import assert_array_equal

from pystruct.models import GridCRF
from pystruct.learners import StructuredPerceptron
import pystruct.toy_datasets as toy


def test_binary_blocks_perceptron_online():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=20, verbose=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_perceptron_parallel():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=200, verbose=10, batch=True,
                               n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
