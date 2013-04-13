import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
##from nose.tools import assert_equal, assert_almost_equal, assert_raises
from nose.tools import assert_almost_equal, assert_equal

from pystruct.problems import EdgeFeatureGraphCRF
from pystruct.inference.linear_programming import lp_general_graph
from pystruct.utils import make_grid_edges
import pystruct.toy_datasets as toy


def test_inference():
    # Test inference with different weights in different directions

    X, Y = toy.generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]
    edges = make_grid_edges(x, neighborhood=4)

    edge_list = make_grid_edges(x, 4, return_lists=True)
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
    res = lp_general_graph(-x.reshape(-1, n_states), edges, edge_weights)

    edge_features = np.zeros((edges.shape[0], 2))
    edge_features[:len(edge_list[0]), 0] = 1
    edge_features[len(edge_list[0]):, 1] = 1

    x = (x.reshape(-1, n_states), edges, edge_features)
    y = y.ravel()

    for inference_method in ["lp", "ad3"]:
        # same inference through CRF inferface
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2)
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=True)
        assert_array_almost_equal(res[0], y_pred[0].reshape(-1, n_states))
        assert_array_almost_equal(res[1], y_pred[1])
        assert_array_equal(y, np.argmax(y_pred[0], axis=-1))

    for inference_method in ["lp", "ad3", "qpbo"]:
        # again, this time discrete predictions only
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2)
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=False)
        assert_array_equal(y, y_pred)
