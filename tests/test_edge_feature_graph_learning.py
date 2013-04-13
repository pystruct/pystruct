import numpy as np
from numpy.testing import assert_array_equal

from pystruct.problems import EdgeFeatureGraphCRF
from pystruct.learners import StructuredSVM
import pystruct.toy_datasets as toy
from pystruct.utils import make_grid_edges


def edge_list_to_features(edge_list):
    edges = np.vstack(edge_list)
    edge_features = np.zeros((edges.shape[0], 2))
    edge_features[:len(edge_list[0]), 0] = 1
    edge_features[len(edge_list[0]):, 1] = 1
    return edge_features


def test_multinomial_blocks_directional():
    # testing cutting plane ssvm with directional CRF on easy multinomial
    # dataset
    X_, Y_ = toy.generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    G = [make_grid_edges(x, return_lists=True) for x in X_]
    edge_features = [edge_list_to_features(edge_list) for edge_list in G]
    edges = [np.vstack(g) for g in G]
    X = zip([x.reshape(-1, 3) for x in X_], edges, edge_features)
    Y = [y.ravel() for y in Y_]

    for inference_method in ['lp', 'ad3']:
        crf = EdgeFeatureGraphCRF(n_states=3,
                                  inference_method=inference_method,
                                  n_edge_features=2)
        clf = StructuredSVM(problem=crf, max_iter=10, C=100, verbose=0,
                            check_constraints=False, n_jobs=-1)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)
