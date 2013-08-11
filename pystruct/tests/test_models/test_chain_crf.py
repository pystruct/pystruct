import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from nose.tools import assert_raises

from pystruct.models import ChainCRF


def test_initialize():
    rnd = np.random.RandomState(0)
    x = rnd.normal(size=(13, 5))
    y = rnd.randint(3, size=13)
    crf = ChainCRF(n_states=3, n_features=5)
    # no-op
    crf.initialize([x], [y])

    #test initialization works
    crf = ChainCRF()
    crf.initialize([x], [y])
    assert_equal(crf.n_states, 3)
    assert_equal(crf.n_features, 5)

    crf = ChainCRF(n_states=2)
    assert_raises(ValueError, crf.initialize, X=[x], Y=[y])
    pass


def test_directed_chain():
    # check that a directed model actually works differntly in the two
    # directions.  chain of length three, three states 0, 1, 2 which want to be
    # in this order, evidence only in the middle
    x = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    w = np.array([1, 0, 0,  # unary
                  0, 1, 0,
                  0, 0, 1,
                  0, 1, 0,  # pairwise
                  0, 0, 1,
                  0, 0, 0])
    crf = ChainCRF(n_states=3, n_features=3)
    y = crf.inference(x, w)
    assert_array_equal([0, 1, 2], y)
