import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal

import pystruct.toy_datasets as toy
from pystruct.utils import exhaustive_loss_augmented_inference, make_grid_edges
from pystruct.problems import LatentGridCRF, LatentDirectionalGridCRF
from pystruct.problems.latent_grid_crf import kmeans_init

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_k_means_initialization():
    n_samples = 10
    X, Y = toy.generate_big_checker(n_samples=n_samples)
    edges = [make_grid_edges(x, return_lists=True) for x in X]
    n_labels = len(np.unique(Y))
    X = X.reshape(n_samples, -1, n_labels)

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=1, n_labels=n_labels)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=3, n_labels=n_labels)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H / 3)

    # for dataset with more than two states
    X, Y = toy.generate_blocks_multinomial(n_samples=10)
    n_labels = len(np.unique(Y))
    edges = [make_grid_edges(x, return_lists=True) for x in X]

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=1, n_labels=n_labels)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=2, n_labels=n_labels)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H / 2)


def test_k_means_initialization_crf():
    X, Y = toy.generate_big_checker(n_samples=10)
    crf = LatentGridCRF(n_labels=2, n_states_per_label=1,
                        inference_method='lp')
    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


def test_k_means_initialization_directional_crf():
    X, Y = toy.generate_big_checker(n_samples=10)
    crf = LatentDirectionalGridCRF(n_labels=2, n_states_per_label=1,
                                   inference_method='lp')
    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


def test_blocks_crf_unaries():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    pairwise_weights = np.array([0,
                                 0,  0,
                                 0,  0,  0,
                                 0,  0,  0, 0])
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    h_hat = crf.inference(x, w)
    assert_array_equal(h_hat / 2, np.argmax(x, axis=-1))


def test_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    pairwise_weights = np.array([0,
                                 0,   0,
                                -4, -4,  0,
                                -4, -4,  0, 0])
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    h_hat = crf.inference(x, w)
    assert_array_equal(y, h_hat / 2)

    h = crf.latent(x, y, w)
    assert_equal(crf.loss(h, h_hat), 0)


def test_blocks_crf_directional():
    # test latent directional CRF on blocks
    # test that all results are the same as equivalent LatentGridCRF
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    pairwise_weights = np.array([0,
                                 0,   0,
                                -4, -4,  0,
                                -4, -4,  0, 0])
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    pw_directional = np.array([0,   0, -4, -4,
                               0,   0, -4, -4,
                               -4, -4,  0,  0,
                               -4, -4,  0,  0,
                               0,   0, -4, -4,
                               0,   0, -4, -4,
                               -4, -4,  0,  0,
                               -4, -4,  0,  0])
    w_directional = np.hstack([unary_weights.ravel(), pw_directional])
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    directional_crf = LatentDirectionalGridCRF(n_labels=2,
                                               n_states_per_label=2)
    h_hat = crf.inference(x, w)
    h_hat_d = directional_crf.inference(x, w_directional)
    assert_array_equal(h_hat, h_hat_d)

    h = crf.latent(x, y, w)
    h_d = directional_crf.latent(x, y, w_directional)
    assert_array_equal(h, h_d)

    h_hat = crf.loss_augmented_inference(x, y, w)
    h_hat_d = directional_crf.loss_augmented_inference(x, y, w_directional)
    assert_array_equal(h_hat, h_hat_d)

    psi = crf.psi(x, h_hat)
    psi_d = directional_crf.psi(x, h_hat)
    assert_array_equal(np.dot(psi, w), np.dot(psi_d, w_directional))


def test_latent_consistency_zero_pw():
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    for i in xrange(10):
        w = np.zeros(18)
        w[:8] = np.random.normal(size=8)
        y = np.random.randint(2, size=(5, 5))
        x = np.random.normal(size=(5, 5, 2))
        h = crf.latent(x, y, w)
        assert_array_equal(h / 2, y)


def test_latent_consistency():
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2)
    for i in xrange(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(4, 4))
        x = np.random.normal(size=(4, 4, 2))
        h = crf.latent(x, y, w)
        assert_array_equal(h / 2, y)


def test_loss_augmented_inference_exhaustive():
    crf = LatentGridCRF(n_labels=2, n_states_per_label=2,
                        inference_method='dai')
    for i in xrange(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(2, 2))
        x = np.random.normal(size=(2, 2, 2))
        h_hat = crf.loss_augmented_inference(x, y * 2, w)
        h = exhaustive_loss_augmented_inference(crf, x, y * 2, w)
        assert_array_equal(h, h_hat)
