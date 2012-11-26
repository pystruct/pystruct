import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal

import pystruct.toy_datasets as toy
from pystruct.crf import exhaustive_loss_augmented_inference
from pystruct.latent_crf import LatentGridCRF
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
    h_hat = crf.inference(x, w)
    assert_array_equal(h_hat / 2, np.argmax(x, axis=-1))


def test_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1,  1,  1,  1,
                  0,
                  0,   0,
                  -4, -4,  0,
                  -4, -4,  0, 0])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    h_hat = crf.inference(x, w)
    assert_array_equal(y, h_hat / 2)

    h = crf.latent(x, y, w)
    assert_equal(crf.loss(h, h_hat), 0)


def test_latent_consistency():
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    for i in xrange(10):
        w = np.random.normal(size=14)
        y = np.random.randint(2, size=(4, 4))
        x = np.random.normal(size=(4, 4, 2))
        h = crf.latent(x, y, w)
        assert_array_equal(h / 2, y)


def test_loss_augmented_inference_exhaustive():
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2,
                        inference_method='dai')
    for i in xrange(10):
        w = np.random.normal(size=14)
        y = np.random.randint(2, size=(2, 2))
        x = np.random.normal(size=(2, 2, 2))
        h_hat = crf.loss_augmented_inference(x, y * 2, w)
        h = exhaustive_loss_augmented_inference(crf, x, y * 2, w)
        assert_array_equal(h, h_hat)
