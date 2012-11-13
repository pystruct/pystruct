import numpy as np
from numpy.testing import assert_array_equal

import toy_datasets as toy
from latent_crf import LatentGridCRF
#from pyqpbo import binary_grid, alpha_expansion_grid

#import itertools

from IPython.core.debugger import Tracer
tracer = Tracer()
# why have binary and multinomial different numbers of parameters?


def test_blocks_crf_unaries():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1,  0,  1,  0,
                  0,
                  0,  0,
                  0,  0,  0,
                  0,  0,  0, 0])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    y_hat = crf.inference(x, w)
    assert_array_equal(y_hat / 2, np.argmax(x, axis=-1))


def test_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    # original:
    # w = np.array([1, 1, 0, -2, 0])
    w = np.array([1,  1,  1,  1,
                  0,
                  0,   0,
                  -4, -4,  0,
                  -4, -4,  0, 0])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat / 2)
