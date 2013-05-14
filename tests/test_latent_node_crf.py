import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_almost_equal

#import pystruct.toy_datasets as toy
#from pystruct.utils import (exhaustive_loss_augmented_inference,
                            #make_grid_edges, find_constraint)
from pystruct.models import LatentNodeCRF
#from pystruct.models.latent_grid_crf import kmeans_init


def test_inference_trivial():
    # size 6 chain graph
    # first three and last three have a latent variable
    features = np.array([-1,  1, -1, 1, -1,  1])
    unary_parameters = np.array([-1, 1])
    pairwise_parameters = np.array([+0,
                                    +0,  0,
                                    +3,  0, 0,
                                    +0,  3, 0, 0])
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

    y_unaries = np.argmax(crf.get_unary_potentials(x, w), axis=1)[:6]
    assert_array_equal(y_unaries, features > 0)

    # test psi
    energy_psi = np.dot(w, crf.psi(x, h))
    assert_almost_equal(energy_psi, -energy_lp)

    # test loss
    h_unaries = crf.latent(x, y_unaries, w)
    assert_equal(crf.loss(h, h_unaries), 2)

    # continuous inference and psi:
    h_continuous, energy_lp = crf.inference(x, w, return_energy=True,
                                            relaxed=True)
    energy_psi = np.dot(w, crf.psi(x, h))
    assert_almost_equal(energy_psi, -energy_lp)

    # test continuous loss
    assert_equal(crf.loss(h, h_continuous), 0)

    #test loss-augmented inference energy
    h_hat, energy_lp = crf.loss_augmented_inference(x, h, w,
                                                    return_energy=True)
    assert_equal(-energy_lp, np.dot(w, crf.psi(x, h_hat)) + crf.loss(h_hat, y))
    #print(h_hat)
    #print(h)
    #print(crf.loss(h_hat, h))


def test_inference_chain():
    # same with pairwise edges:
    features = np.array([-1,  1, -1, 1, -1,  1])
    unary_parameters = np.array([-1, 1])
    pairwise_parameters = np.array([+1,
                                    +0,  1,
                                    +3,  0, 0,
                                    +0,  3, 0, 0])
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
    y = np.argmax(crf.get_unary_potentials(x, w), axis=1)[:6]
    energy_psi = np.dot(w, crf.psi(x, h))

    assert_almost_equal(energy_psi, -energy_lp)
    assert_array_equal(y, features > 0)
    assert_array_equal(h, [0, 0, 0, 1, 1, 1, 2, 3])

    # continuous inference and psi:
    h, energy_lp = crf.inference(x, w, return_energy=True, relaxed=True)
    energy_psi = np.dot(w, crf.psi(x, h))
    assert_almost_equal(energy_psi, -energy_lp)
