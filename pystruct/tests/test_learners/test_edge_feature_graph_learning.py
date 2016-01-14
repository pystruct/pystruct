import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import NSlackSSVM
from pystruct.datasets import generate_blocks_multinomial
from pystruct.utils import make_grid_edges


def edge_list_to_features(edge_list):
    edges = np.vstack(edge_list)
    edge_features = np.zeros((edges.shape[0], 2))
    edge_features[:len(edge_list[0]), 0] = 1
    edge_features[len(edge_list[0]):, 1] = 1
    return edge_features


def test_multinomial_blocks_directional_simple():
    # testing cutting plane ssvm with directional CRF on easy multinomial
    # dataset
    X_, Y_ = generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    G = [make_grid_edges(x, return_lists=True) for x in X_]
    edge_features = [edge_list_to_features(edge_list) for edge_list in G]
    edges = [np.vstack(g) for g in G]
    X = list(zip([x.reshape(-1, 3) for x in X_], edges, edge_features))
    Y = [y.ravel() for y in Y_]

    crf = EdgeFeatureGraphCRF(n_states=3, n_edge_features=2)
    clf = NSlackSSVM(model=crf, max_iter=10, C=1, check_constraints=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks_directional_anti_symmetric():
    # testing cutting plane ssvm with directional CRF on easy multinomial
    # dataset
    X_, Y_ = generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    G = [make_grid_edges(x, return_lists=True) for x in X_]
    edge_features = [edge_list_to_features(edge_list) for edge_list in G]
    edges = [np.vstack(g) for g in G]
    X = list(zip([x.reshape(-1, 3) for x in X_], edges, edge_features))
    Y = [y.ravel() for y in Y_]

    crf = EdgeFeatureGraphCRF(symmetric_edge_features=[0],
                              antisymmetric_edge_features=[1])
    clf = NSlackSSVM(model=crf, C=100)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    pairwise_params = clf.w[-9 * 2:].reshape(2, 3, 3)
    sym = pairwise_params[0]
    antisym = pairwise_params[1]
    assert_array_almost_equal(sym, sym.T)
    assert_array_almost_equal(antisym, -antisym.T)
