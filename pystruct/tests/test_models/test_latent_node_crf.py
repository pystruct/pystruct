import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from nose.tools import assert_raises

from pystruct.datasets import generate_blocks_multinomial
from pystruct.inference.linear_programming import lp_general_graph
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.models import LatentNodeCRF, EdgeFeatureLatentNodeCRF
from pystruct.inference import get_installed


def test_initialize():
    # 17 nodes, three features, 5 visible states, 2 hidden states
    rnd = np.random.RandomState(0)
    feats = rnd.normal(size=(17, 3))
    edges = np.zeros((0, 2), dtype=np.int)  # no edges
    x = (feats, edges, 4)   # 4 latent variables
    y = rnd.randint(5, size=17)
    crf = LatentNodeCRF(n_labels=5, n_features=3)
    # no-op
    crf.initialize([x], [y])
    assert_equal(crf.n_states, 5 + 2)

    #test initialization works
    crf = LatentNodeCRF()
    crf.initialize([x], [y])
    assert_equal(crf.n_labels, 5)
    assert_equal(crf.n_states, 5 + 2)
    assert_equal(crf.n_features, 3)

    crf = LatentNodeCRF(n_labels=3)
    assert_raises(ValueError, crf.initialize, X=[x], Y=[y])


def test_inference_trivial():
    # size 6 chain graph
    # first three and last three have a latent variable
    features = np.array([-1, 1, -1, 1, -1, 1])
    unary_parameters = np.array([-1, 1])
    pairwise_parameters = np.array([+0,
                                    +0, 0,
                                    +3, 0, 0,
                                    +0, 3, 0, 0])
    w = np.hstack([unary_parameters, pairwise_parameters])
    crf = LatentNodeCRF(n_labels=2, n_features=1, n_hidden_states=2)
    # edges for latent states. Latent states named 6, 7
    node_indices = np.arange(features.size)
    other_edges = []
    for n in node_indices[:3]:
        other_edges.append([n, 6])
    for n in node_indices[3:]:
        other_edges.append([n, 7])

    all_edges = np.vstack(other_edges)
    x = (features.reshape(-1, 1), all_edges, 2)

    # test inference
    h, energy_lp = crf.inference(x, w, return_energy=True)
    assert_array_equal(h, [0, 0, 0, 1, 1, 1, 2, 3])

    y = crf.label_from_latent(h)
    assert_array_equal(y, [0, 0, 0, 1, 1, 1])

    y_unaries = np.argmax(crf._get_unary_potentials(x, w), axis=1)[:6]
    assert_array_equal(y_unaries, features > 0)

    # test joint_feature
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))
    assert_almost_equal(energy_joint_feature, -energy_lp)

    # test loss
    h_unaries = crf.latent(x, y_unaries, w)
    assert_equal(crf.loss(h, h_unaries), 2)

    # continuous inference and joint_feature:
    h_continuous, energy_lp = crf.inference(x, w, return_energy=True,
                                            relaxed=True)
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))
    assert_almost_equal(energy_joint_feature, -energy_lp)

    # test continuous loss
    assert_equal(crf.loss(h, h_continuous), 0)

    #test loss-augmented inference energy
    h_hat, energy_lp = crf.loss_augmented_inference(x, h, w,
                                                    return_energy=True)
    assert_almost_equal(-energy_lp, np.dot(w, crf.joint_feature(x, h_hat)) +
                        crf.loss(h_hat, y))


def test_inference_chain():
    # same with pairwise edges:
    features = np.array([-1, 1, -1, 1, -1, 1])
    unary_parameters = np.array([-1, 1])
    pairwise_parameters = np.array([+1,
                                    +0, 1,
                                    +3, 0, 0,
                                    +0, 3, 0, 0])
    w = np.hstack([unary_parameters, pairwise_parameters])
    crf = LatentNodeCRF(n_labels=2, n_features=1, n_hidden_states=2)
    edges = np.vstack([np.arange(5), np.arange(1, 6)]).T

    # edges for latent states. Latent states named 6, 7
    node_indices = np.arange(features.size)
    other_edges = []
    for n in node_indices[:3]:
        other_edges.append([n, 6])
    for n in node_indices[3:]:
        other_edges.append([n, 7])
    all_edges = np.vstack([edges, other_edges])

    x = (features.reshape(-1, 1), all_edges, 2)
    h, energy_lp = crf.inference(x, w, return_energy=True)
    y = np.argmax(crf._get_unary_potentials(x, w), axis=1)[:6]
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))

    assert_almost_equal(energy_joint_feature, -energy_lp)
    assert_array_equal(y, features > 0)
    assert_array_equal(h, [0, 0, 0, 1, 1, 1, 2, 3])

    # continuous inference and joint_feature:
    h, energy_lp = crf.inference(x, w, return_energy=True, relaxed=True)
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))
    assert_almost_equal(energy_joint_feature, -energy_lp)


def test_inference_trivial_features():
    # size 6 chain graph
    # first three and last three have a latent variable
    # last two features are for latent variables
    features = np.array([-1, 1, -1, 1, -1, 1, 0, 0], dtype=np.float)
    unary_parameters = np.array([-1, 1, 0, 0])
    pairwise_parameters = np.array([+0,
                                    +0, 0,
                                    +3, 0, 0,
                                    +0, 3, 0, 0])
    w = np.hstack([unary_parameters, pairwise_parameters])
    crf = LatentNodeCRF(n_labels=2, n_features=1, n_hidden_states=2,
                        latent_node_features=True)
    # edges for latent states. Latent states named 6, 7
    node_indices = np.arange(6)
    other_edges = []
    for n in node_indices[:3]:
        other_edges.append([n, 6])
    for n in node_indices[3:]:
        other_edges.append([n, 7])

    all_edges = np.vstack(other_edges)
    x = (features.reshape(-1, 1), all_edges, 2)

    # test inference
    h, energy_lp = crf.inference(x, w, return_energy=True)
    assert_array_equal(h, [0, 0, 0, 1, 1, 1, 2, 3])

    y = crf.label_from_latent(h)
    assert_array_equal(y, [0, 0, 0, 1, 1, 1])

    y_unaries = np.argmax(crf._get_unary_potentials(x, w), axis=1)[:6]
    assert_array_equal(y_unaries, features[:6] > 0)

    # test joint_feature
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))
    assert_almost_equal(energy_joint_feature, -energy_lp)

    # test loss
    h_unaries = crf.latent(x, y_unaries, w)
    assert_equal(crf.loss(h, h_unaries), 2)

    # continuous inference and joint_feature:
    h_continuous, energy_lp = crf.inference(x, w, return_energy=True,
                                            relaxed=True)
    energy_joint_feature = np.dot(w, crf.joint_feature(x, h))
    assert_almost_equal(energy_joint_feature, -energy_lp)

    # test continuous loss
    assert_equal(crf.loss(h, h_continuous), 0)

    #test loss-augmented inference energy
    h_hat, energy_lp = crf.loss_augmented_inference(x, h, w,
                                                    return_energy=True)
    assert_almost_equal(-energy_lp, np.dot(w, crf.joint_feature(x, h_hat)) +
                        crf.loss(h_hat, y))


def test_edge_feature_latent_node_crf_no_latent():
    # no latent nodes

    # Test inference with different weights in different directions

    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1, size_x=8)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]

    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)

    pw_horz = -1 * np.eye(n_states + 5)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states + 5)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # generate edge weights
    edge_weights_horizontal = np.repeat(pw_horz[np.newaxis, :, :],
                                        edge_list[0].shape[0], axis=0)
    edge_weights_vertical = np.repeat(pw_vert[np.newaxis, :, :],
                                      edge_list[1].shape[0], axis=0)
    edge_weights = np.vstack([edge_weights_horizontal, edge_weights_vertical])

    # do inference
    # pad x for hidden states...
    x_padded = -100 * np.ones((x.shape[0], x.shape[1], x.shape[2] + 5))
    x_padded[:, :, :x.shape[2]] = x
    res = lp_general_graph(-x_padded.reshape(-1, n_states + 5), edges,
                           edge_weights)

    edge_features = edge_list_to_features(edge_list)
    x = (x.reshape(-1, n_states), edges, edge_features, 0)
    y = y.ravel()

    for inference_method in get_installed(["lp"]):
        # same inference through CRF inferface
        crf = EdgeFeatureLatentNodeCRF(n_labels=3,
                                       inference_method=inference_method,
                                       n_edge_features=2, n_hidden_states=5)
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=True)
        assert_array_almost_equal(res[0], y_pred[0].reshape(-1, n_states + 5),
                                  4)
        assert_array_almost_equal(res[1], y_pred[1], 4)
        assert_array_equal(y, np.argmax(y_pred[0], axis=-1))

    for inference_method in get_installed(["qpbo", "ad3", "lp"])[:2]:
        # again, this time discrete predictions only
        crf = EdgeFeatureLatentNodeCRF(n_labels=3,
                                       inference_method=inference_method,
                                       n_edge_features=2, n_hidden_states=5)
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        y_pred = crf.inference(x, w, relaxed=False)
        assert_array_equal(y, y_pred)
