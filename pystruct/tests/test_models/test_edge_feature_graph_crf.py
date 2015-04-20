import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)
from nose.tools import assert_raises

from pystruct.models import EdgeFeatureGraphCRF
from pystruct.inference.linear_programming import lp_general_graph
from pystruct.inference import compute_energy, get_installed
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.datasets import generate_blocks_multinomial


def test_initialization():
    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]

    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)

    edge_features = edge_list_to_features(edge_list)
    x = (x.reshape(-1, n_states), edges, edge_features)
    y = y.ravel()
    crf = EdgeFeatureGraphCRF()
    crf.initialize([x], [y])
    assert_equal(crf.n_edge_features, 2)
    assert_equal(crf.n_features, 3)
    assert_equal(crf.n_states, 3)

    crf = EdgeFeatureGraphCRF(n_states=3,
                              n_features=3,
                              n_edge_features=2)
    # no-op
    crf.initialize([x], [y])

    crf = EdgeFeatureGraphCRF(n_states=4,
                              n_edge_features=2)
    # incompatible
    assert_raises(ValueError, crf.initialize, X=[x], Y=[y])


def test_inference():
    # Test inference with different weights in different directions

    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]

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

    edge_features = edge_list_to_features(edge_list)
    x = (x.reshape(-1, n_states), edges, edge_features)
    y = y.ravel()

    for inference_method in get_installed(["lp", "ad3"]):
        # same inference through CRF inferface
        crf = EdgeFeatureGraphCRF(inference_method=inference_method)
        crf.initialize([x], [y])
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=True)
        if isinstance(y_pred, tuple):
            # ad3 produces an integer result if it found the exact solution
            assert_array_almost_equal(res[1], y_pred[1])
            assert_array_almost_equal(res[0], y_pred[0].reshape(-1, n_states))
            assert_array_equal(y, np.argmax(y_pred[0], axis=-1))

    for inference_method in get_installed(["lp", "ad3", "qpbo"]):
        # again, this time discrete predictions only
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2)
        crf.initialize([x], [y])
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=False)
        assert_array_equal(y, y_pred)


def test_joint_feature_discrete():
    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)
    edge_features = edge_list_to_features(edge_list)
    x = (x.reshape(-1, 3), edges, edge_features)
    y_flat = y.ravel()
    for inference_method in get_installed(["lp", "ad3", "qpbo"]):
        crf = EdgeFeatureGraphCRF(inference_method=inference_method)
        crf.initialize([x], [y_flat])
        joint_feature_y = crf.joint_feature(x, y_flat)
        assert_equal(joint_feature_y.shape, (crf.size_joint_feature,))
        # first horizontal, then vertical
        # we trust the unaries ;)
        pw_joint_feature_horz, pw_joint_feature_vert = joint_feature_y[crf.n_states *
                                         crf.n_features:].reshape(
                                             2, crf.n_states, crf.n_states)
        xx, yy = np.indices(y.shape)
        assert_array_equal(pw_joint_feature_vert, np.diag([9 * 4, 9 * 4, 9 * 4]))
        vert_joint_feature = np.diag([10 * 3, 10 * 3, 10 * 3])
        vert_joint_feature[0, 1] = 10
        vert_joint_feature[1, 2] = 10
        assert_array_equal(pw_joint_feature_horz, vert_joint_feature)


def test_joint_feature_continuous():
    # FIXME
    # first make perfect prediction, including pairwise part
    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]
    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)
    edge_features = edge_list_to_features(edge_list)
    x = (x.reshape(-1, 3), edges, edge_features)
    y = y.ravel()

    pw_horz = -1 * np.eye(n_states)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # create crf, assemble weight, make prediction
    for inference_method in get_installed(["lp", "ad3"]):
        crf = EdgeFeatureGraphCRF(inference_method=inference_method)
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        crf.initialize([x], [y])
        y_pred = crf.inference(x, w, relaxed=True)

        # compute joint_feature for prediction
        joint_feature_y = crf.joint_feature(x, y_pred)
        assert_equal(joint_feature_y.shape, (crf.size_joint_feature,))
        # FIXME
        # first horizontal, then vertical
        # we trust the unaries ;)
        #pw_joint_feature_horz, pw_joint_feature_vert = joint_feature_y[crf.n_states *
                                 #crf.n_features:].reshape(2,
                                                          #crf.n_states,
                                                          #crf.n_states)


def test_energy_continuous():
    # make sure that energy as computed by ssvm is the same as by lp
    np.random.seed(0)
    for inference_method in get_installed(["lp", "ad3"]):
        found_fractional = False
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2, n_features=3)
        while not found_fractional:
            x = np.random.normal(size=(7, 8, 3))
            edge_list = make_grid_edges(x, 4, return_lists=True)
            edges = np.vstack(edge_list)
            edge_features = edge_list_to_features(edge_list)
            x = (x.reshape(-1, 3), edges, edge_features)

            unary_params = np.random.normal(size=(3, 3))
            pw1 = np.random.normal(size=(3, 3))
            pw2 = np.random.normal(size=(3, 3))
            w = np.hstack([unary_params.ravel(), pw1.ravel(), pw2.ravel()])
            res, energy = crf.inference(x, w, relaxed=True, return_energy=True)
            found_fractional = np.any(np.max(res[0], axis=-1) != 1)

            joint_feature = crf.joint_feature(x, res)
            energy_svm = np.dot(joint_feature, w)

            assert_almost_equal(energy, -energy_svm)


def test_energy_discrete():
    for inference_method in get_installed(["qpbo", "ad3"]):
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2, n_features=3)
        for i in range(10):
            x = np.random.normal(size=(7, 8, 3))
            edge_list = make_grid_edges(x, 4, return_lists=True)
            edges = np.vstack(edge_list)
            edge_features = edge_list_to_features(edge_list)
            x = (x.reshape(-1, 3), edges, edge_features)

            unary_params = np.random.normal(size=(3, 3))
            pw1 = np.random.normal(size=(3, 3))
            pw2 = np.random.normal(size=(3, 3))
            w = np.hstack([unary_params.ravel(), pw1.ravel(), pw2.ravel()])
            y_hat = crf.inference(x, w, relaxed=False)
            energy = compute_energy(crf._get_unary_potentials(x, w),
                                    crf._get_pairwise_potentials(x, w), edges,
                                    y_hat)

            joint_feature = crf.joint_feature(x, y_hat)
            energy_svm = np.dot(joint_feature, w)

            assert_almost_equal(energy, energy_svm)
