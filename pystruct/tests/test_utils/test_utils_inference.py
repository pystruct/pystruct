import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal

from pystruct.utils import compress_sym, expand_sym


def test_symmetric_tools_symmetric():
    rnd = np.random.RandomState(0)
    # generate random symmetric matrix
    for size in [4, 6, 11]:
        x = rnd.normal(size=(size, size))
        x = x + x.T

        compressed = compress_sym(x, make_symmetric=False)
        assert_equal(compressed.shape, (size * (size + 1) / 2, ))

        uncompressed = expand_sym(compressed)
        assert_array_equal(x, uncompressed)


def test_symmetric_tools_upper():
    rnd = np.random.RandomState(0)
    # generate random matrix with only upper triangle.
    # expected result is full symmetric matrix
    for size in [4, 6, 11]:
        x = rnd.normal(size=(size, size))
        x = x + x.T
        x_ = x.copy()
        x[np.tri(size, k=-1, dtype=np.bool)] = 0

        compressed = compress_sym(x, make_symmetric=True)
        assert_equal(compressed.shape, (size * (size + 1) / 2, ))

        uncompressed = expand_sym(compressed)
        assert_array_equal(x_, uncompressed)
