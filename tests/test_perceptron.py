#import numpy as np
from numpy.testing import assert_array_equal
from pystruct.models import GridCRF
from pystruct.learners import StructuredPerceptron
import pystruct.toy_datasets as toy


def test_binary_blocks():
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=40)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks():
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    crf = GridCRF(n_states=X.shape[-1])
    clf = StructuredPerceptron(model=crf, max_iter=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
