import itertools

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_true
from pystruct.models import GraphCRF, LatentNodeCRF, EdgeFeatureLatentNodeCRF
from pystruct.learners import (NSlackSSVM, LatentSSVM,
                               SubgradientLatentSSVM, OneSlackSSVM,
                               SubgradientSSVM)
from pystruct.datasets import generate_blocks, make_simple_2x2
from pystruct.utils import make_grid_edges


def make_edges_2x2():
    edges = []
    node_indices = np.arange(4 * 4).reshape(4, 4)
    for i, (x, y) in enumerate(itertools.product([0, 2], repeat=2)):
        for j in range(x, x + 2):
            for k in range(y, y + 2):
                edges.append([i + 4 * 4, node_indices[j, k]])
    return edges


def test_binary_blocks_cutting_plane_latent_node():
    #testing cutting plane ssvm on easy binary dataset
    # we use the LatentNodeCRF without latent nodes and check that it does the
    # same as GraphCRF
    X, Y = generate_blocks(n_samples=3)
    crf = GraphCRF()
    clf = NSlackSSVM(model=crf, max_iter=20, C=100, check_constraints=True,
                     break_on_bad=False, n_jobs=1)
    x1, x2, x3 = X
    y1, y2, y3 = Y
    n_states = len(np.unique(Y))
    # delete some rows to make it more fun
    x1, y1 = x1[:, :-1], y1[:, :-1]
    x2, y2 = x2[:-1], y2[:-1]
    # generate graphs
    X_ = [x1, x2, x3]
    G = [make_grid_edges(x) for x in X_]

    # reshape / flatten x and y
    X_ = [x.reshape(-1, n_states) for x in X_]
    Y = [y.ravel() for y in [y1, y2, y3]]

    X = list(zip(X_, G))

    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    for y, y_pred in zip(Y, Y_pred):
        assert_array_equal(y, y_pred)

    latent_crf = LatentNodeCRF(n_labels=2, n_hidden_states=0)
    latent_svm = LatentSSVM(NSlackSSVM(model=latent_crf, max_iter=20, C=100,
                                       check_constraints=True,
                                       break_on_bad=False, n_jobs=1),
                            latent_iter=3)
    X_latent = list(zip(X_, G, np.zeros(len(X_))))
    latent_svm.fit(X_latent, Y, H_init=Y)
    Y_pred = latent_svm.predict(X_latent)
    for y, y_pred in zip(Y, Y_pred):
        assert_array_equal(y, y_pred)

    assert_array_almost_equal(latent_svm.w, clf.w)


def test_latent_node_boxes_standard_latent():
    # learn the "easy" 2x2 boxes dataset.
    # a 2x2 box is placed randomly in a 4x4 grid
    # we add a latent variable for each 2x2 patch
    # that should make the model fairly simple

    X, Y = make_simple_2x2(seed=1, n_samples=40)
    latent_crf = LatentNodeCRF(n_labels=2, n_hidden_states=2, n_features=1)
    one_slack = OneSlackSSVM(latent_crf)
    n_slack = NSlackSSVM(latent_crf)
    subgradient = SubgradientSSVM(latent_crf, max_iter=100)
    for base_svm in [one_slack, n_slack, subgradient]:
        base_svm.C = 10
        latent_svm = LatentSSVM(base_svm,
                                latent_iter=10)

        G = [make_grid_edges(x) for x in X]

        # make edges for hidden states:
        edges = make_edges_2x2()

        G = [np.vstack([make_grid_edges(x), edges]) for x in X]

        # reshape / flatten x and y
        X_flat = [x.reshape(-1, 1) for x in X]
        Y_flat = [y.ravel() for y in Y]

        X_ = list(zip(X_flat, G, [2 * 2 for x in X_flat]))
        latent_svm.fit(X_[:20], Y_flat[:20])

        assert_array_equal(latent_svm.predict(X_[:20]), Y_flat[:20])
        assert_equal(latent_svm.score(X_[:20], Y_flat[:20]), 1)

        # test that score is not always 1
        assert_true(.98 < latent_svm.score(X_[20:], Y_flat[20:]) < 1)


def test_latent_node_boxes_latent_subgradient():
    # same as above, now with elementary subgradients

    X, Y = make_simple_2x2(seed=1)
    latent_crf = LatentNodeCRF(n_labels=2, n_hidden_states=2, n_features=1)
    latent_svm = SubgradientLatentSSVM(model=latent_crf, max_iter=50, C=10)

    G = [make_grid_edges(x) for x in X]

    edges = make_edges_2x2()
    G = [np.vstack([make_grid_edges(x), edges]) for x in X]

    # reshape / flatten x and y
    X_flat = [x.reshape(-1, 1) for x in X]
    Y_flat = [y.ravel() for y in Y]

    X_ = list(zip(X_flat, G, [4 * 4 for x in X_flat]))
    latent_svm.fit(X_, Y_flat)

    assert_equal(latent_svm.score(X_, Y_flat), 1)


def test_latent_node_boxes_standard_latent_features():
    # learn the "easy" 2x2 boxes dataset.
    # we make it even easier now by adding features that encode the correct
    # latent state. This basically tests that the features are actually used

    X, Y = make_simple_2x2(seed=1, n_samples=20, n_flips=6)
    latent_crf = LatentNodeCRF(n_labels=2, n_hidden_states=2, n_features=1,
                               latent_node_features=True)
    one_slack = OneSlackSSVM(latent_crf)
    n_slack = NSlackSSVM(latent_crf)
    subgradient = SubgradientSSVM(latent_crf, max_iter=100)
    for base_svm in [one_slack, n_slack, subgradient]:
        base_svm.C = 10
        latent_svm = LatentSSVM(base_svm,
                                latent_iter=10)

        G = [make_grid_edges(x) for x in X]

        # make edges for hidden states:
        edges = make_edges_2x2()

        G = [np.vstack([make_grid_edges(x), edges]) for x in X]

        # reshape / flatten x and y
        X_flat = [x.reshape(-1, 1) for x in X]
        # augment X with the features for hidden units
        X_flat = [np.vstack([x, y[::2, ::2].reshape(-1, 1)])
                  for x, y in zip(X_flat, Y)]
        Y_flat = [y.ravel() for y in Y]

        X_ = list(zip(X_flat, G, [2 * 2 for x in X_flat]))
        latent_svm.fit(X_[:10], Y_flat[:10])

        assert_array_equal(latent_svm.predict(X_[:10]), Y_flat[:10])
        assert_equal(latent_svm.score(X_[:10], Y_flat[:10]), 1)

        # we actually become prefect ^^
        assert_true(.98 < latent_svm.score(X_[10:], Y_flat[10:]) <= 1)


def test_latent_node_boxes_edge_features():
    # learn the "easy" 2x2 boxes dataset.
    # smoketest using a single constant edge feature

    X, Y = make_simple_2x2(seed=1, n_samples=40)
    latent_crf = EdgeFeatureLatentNodeCRF(n_labels=2, n_hidden_states=2, n_features=1)
    base_svm = OneSlackSSVM(latent_crf)
    base_svm.C = 10
    latent_svm = LatentSSVM(base_svm,
                            latent_iter=10)

    G = [make_grid_edges(x) for x in X]

    # make edges for hidden states:
    edges = make_edges_2x2()

    G = [np.vstack([make_grid_edges(x), edges]) for x in X]

    # reshape / flatten x and y
    X_flat = [x.reshape(-1, 1) for x in X]
    Y_flat = [y.ravel() for y in Y]

    #X_ = zip(X_flat, G, [2 * 2 for x in X_flat])
    # add edge features
    X_ = [(x, g, np.ones((len(g), 1)), 4) for x, g in zip(X_flat, G)]
    latent_svm.fit(X_[:20], Y_flat[:20])

    assert_array_equal(latent_svm.predict(X_[:20]), Y_flat[:20])
    assert_equal(latent_svm.score(X_[:20], Y_flat[:20]), 1)

    # test that score is not always 1
    assert_true(.98 < latent_svm.score(X_[20:], Y_flat[20:]) < 1)
