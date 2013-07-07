import numpy as np
from nose.tools import assert_true, assert_false
from scipy import sparse

from sklearn.utils.mst import minimum_spanning_tree

from pystruct.inference.maxprod import is_tree


def test_is_tree():
    # generate chain
    chain = np.c_[np.arange(1, 10), np.arange(9)]
    assert_true(is_tree(len(chain) + 1, chain))
    # generate circle
    circle = np.vstack([chain, [9, 0]])
    assert_false(is_tree(len(chain) + 1, circle))

    # union of two disjoint chains
    two_chains = np.vstack([chain, chain + 10])
    assert_true(is_tree(20, two_chains))

    # union of chain and circle
    disco_graph = np.vstack([chain, circle + 10])
    assert_false(is_tree(20, disco_graph))

    # generate random fully connected graph
    graph = np.random.uniform(size=(10, 10))
    edges = np.c_[graph.nonzero()]
    assert_false(is_tree(10, edges))

    tree = minimum_spanning_tree(sparse.csr_matrix(graph))
    tree_edges = np.c_[tree.nonzero()]
    assert_true(is_tree(10, tree_edges))
