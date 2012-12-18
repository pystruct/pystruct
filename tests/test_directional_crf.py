import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal

import pystruct.toy_datasets as toy
from pystruct.lp_new import lp_general_graph
from pystruct.inference_methods import _make_grid_edges
from pystruct.crf import DirectionalGridCRF


def test_inference():
    # Test inference with different weights in different directions

    X, Y = toy.generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]
    edges = _make_grid_edges(x, neighborhood=4)

    edge_list = _make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)

    pw_horz = -1 * np.eye(n_states)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # generate edge weights
    edge_weights_horizontal = np.repeat(pw_horz[np.newaxis, :, :],
                                        edge_list[0].shape[0], axis=0)
    edge_weights_vertical = np.repeat(pw_vert[np.newaxis, :, :],
                                      edge_list[1].shape[0], axis=0)
    edge_weights = np.vstack([edge_weights_horizontal, edge_weights_vertical])

    # do inference
    res = lp_general_graph(-x.reshape(-1, n_states), edges, edge_weights,
                           exact=False)

    # same inference through CRF inferface
    crf = DirectionalGridCRF(n_states=3, inference_method='lp')
    w = np.hstack([np.ones(3), -pw_horz.ravel(), -pw_vert.ravel()])
    y_pred = crf.inference(x, w, relaxed=True)
    assert_array_almost_equal(res[0], y_pred[0].reshape(-1, n_states))
    assert_array_almost_equal(res[1], y_pred[1])
    assert_array_equal(y, np.argmax(y_pred[0], axis=-1))


def test_psi_discrete():
    X, Y = toy.generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    crf = DirectionalGridCRF(n_states=3, inference_method='lp')
    psi_y = crf.psi(x, y)
    assert_equal(psi_y.shape, (crf.size_psi,))
    # first unary, then horizontal, then vertical
    unary_psi = crf.get_unary_weights(psi_y)
    pw_psi_horz, pw_psi_vert = crf.get_pairwise_weights(psi_y)
    xx, yy = np.indices(y.shape)
    assert_array_almost_equal(unary_psi,
                              np.bincount(y.ravel(), x[xx, yy, y].ravel()))
    assert_array_equal(pw_psi_vert, np.diag([9 * 4, 9 * 4, 9 * 4]))
    vert_psi = np.diag([10 * 3, 10 * 3, 10 * 3])
    vert_psi[1, 0] = 10
    vert_psi[2, 1] = 10
    assert_array_equal(pw_psi_horz, vert_psi)


def test_psi_continuous():
    # first make perfect prediction, including pairwise part
    X, Y = toy.generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]

    pw_horz = -1 * np.eye(n_states)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # create crf, assemble weight, make prediction
    crf = DirectionalGridCRF(n_states=3, inference_method='lp')
    w = np.hstack([np.ones(3), -pw_horz.ravel(), -pw_vert.ravel()])
    y_pred = crf.inference(x, w, relaxed=True)

    # compute psi for prediction
    psi_y = crf.psi(x, y_pred)
    assert_equal(psi_y.shape, (crf.size_psi,))
    # first unary, then horizontal, then vertical
    unary_psi = crf.get_unary_weights(psi_y)
    pw_psi_horz, pw_psi_vert = crf.get_pairwise_weights(psi_y)

    # test unary
    xx, yy = np.indices(y.shape)
    assert_array_almost_equal(unary_psi,
                              np.bincount(y.ravel(), x[xx, yy, y].ravel()))
