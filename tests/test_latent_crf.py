import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal

import pystruct.toy_datasets as toy
from pystruct.crf import exhaustive_loss_augmented_inference
from pystruct.latent_crf import LatentGridCRF
from pystruct.latent_crf import kmeans_init
from pystruct.inference_methods import _make_grid_edges

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_k_means_initialization():
    X, Y = toy.generate_big_checker(n_samples=10)
    edges = _make_grid_edges(X[0], return_lists=True)

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=1)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=3)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H / 3)

    # for dataset with more than two states
    X, Y = toy.generate_blocks_multinomial(n_samples=10)
    edges = _make_grid_edges(X[0], return_lists=True)

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=1)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=2)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H / 2)


def test_k_means_initialization_crf():
    X, Y = toy.generate_big_checker(n_samples=10)
    crf = LatentGridCRF(n_labels=2, n_states_per_label=1,
                        inference_method='lp')
    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


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
