import numpy as np
from nose.tools import assert_true, assert_false
from numpy.testing import assert_array_equal
from scipy import sparse

from pystruct.inference.maxprod import is_tree, inference_max_product
from pystruct.inference import inference_ad3


def test_is_tree():
    # generate chain
    chain = np.c_[np.arange(1, 10), np.arange(9)]
    assert_true(is_tree(chain, len(chain) + 1))
    assert_true(is_tree(chain))
    # generate circle
    circle = np.vstack([chain, [9, 0]])
    assert_false(is_tree(circle))
    assert_false(is_tree(circle, len(chain) + 1))

    # union of two disjoint chains
    two_chains = np.vstack([chain, chain + 10])
    assert_true(is_tree(two_chains, 20))

    # union of chain and circle
    disco_graph = np.vstack([chain, circle + 10])
    assert_false(is_tree(disco_graph))

    # generate random fully connected graph
    graph = np.random.uniform(size=(10, 10))
    edges = np.c_[graph.nonzero()]
    assert_false(is_tree(edges))

    tree = sparse.csgraph.minimum_spanning_tree(sparse.csr_matrix(graph))
    tree_edges = np.c_[tree.nonzero()]
    assert_true(is_tree(tree_edges, 10))
    assert_true(is_tree(tree_edges))


def test_tree_max_product():
    chain = np.c_[np.arange(1, 10), np.arange(9)]
    unary_potentials = np.random.normal(size=(10, 3))
    pairwise_potentials = np.random.normal(size=(3, 3))
    result_ad3 = inference_ad3(unary_potentials, pairwise_potentials, chain,
                               branch_and_bound=True)
    result_mp = inference_max_product(unary_potentials, pairwise_potentials,
                                      chain)
    assert_array_equal(result_ad3, result_mp)
