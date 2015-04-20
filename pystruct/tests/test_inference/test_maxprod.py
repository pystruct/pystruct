import numpy as np
from nose.tools import assert_true, assert_false
from nose import SkipTest

from numpy.testing import assert_array_equal
from scipy import sparse

from pystruct.inference.maxprod import (is_forest, inference_max_product,
                                        iterative_max_product, is_chain)
from pystruct.inference import inference_ad3
from pystruct.datasets import generate_blocks, generate_blocks_multinomial
from pystruct.models import GridCRF


def test_is_chain():
    chain = np.c_[np.arange(9), np.arange(1, 10)]
    assert_true(is_chain(chain, len(chain) + 1))
    assert_false(is_chain(chain, len(chain)))
    # generate circle
    circle = np.vstack([chain, [9, 0]])
    assert_false(is_chain(circle, len(circle)))
    # reversed order not accepted
    assert_false(is_chain(chain[::-1], len(chain) + 1))


def test_is_forest():
    # generate chain
    chain = np.c_[np.arange(1, 10), np.arange(9)]
    assert_true(is_forest(chain, len(chain) + 1))
    assert_true(is_forest(chain))
    # generate circle
    circle = np.vstack([chain, [9, 0]])
    assert_false(is_forest(circle))
    assert_false(is_forest(circle, len(chain) + 1))

    # union of two disjoint chains
    two_chains = np.vstack([chain, chain + 10])
    assert_true(is_forest(two_chains, 20))

    # union of chain and circle
    disco_graph = np.vstack([chain, circle + 10])
    assert_false(is_forest(disco_graph))

    # generate random fully connected graph
    graph = np.random.uniform(size=(10, 10))
    edges = np.c_[graph.nonzero()]
    assert_false(is_forest(edges))

    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        tree = minimum_spanning_tree(sparse.csr_matrix(graph))
        tree_edges = np.c_[tree.nonzero()]
        assert_true(is_forest(tree_edges, 10))
        assert_true(is_forest(tree_edges))
    except ImportError:
        pass


def test_tree_max_product_chain():
    rnd = np.random.RandomState(0)
    forward = np.c_[np.arange(9), np.arange(1, 10)]
    backward = np.c_[np.arange(1, 10), np.arange(9)]
    for i in range(10):
        unary_potentials = rnd.normal(size=(10, 3))
        pairwise_potentials = rnd.normal(size=(9, 3, 3))
        for chain in [forward, backward]:
            result_ad3 = inference_ad3(unary_potentials, pairwise_potentials,
                                       chain, branch_and_bound=True)
            result_mp = inference_max_product(unary_potentials,
                                              pairwise_potentials, chain)
            assert_array_equal(result_ad3, result_mp)


def test_tree_max_product_tree():
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
    except:
        raise SkipTest("Not testing trees, scipy version >= 0.11 required")

    rnd = np.random.RandomState(0)
    for i in range(100):
        # generate random tree using mst
        graph = rnd.uniform(size=(10, 10))
        tree = minimum_spanning_tree(sparse.csr_matrix(graph))
        tree_edges = np.c_[tree.nonzero()]

        unary_potentials = rnd.normal(size=(10, 3))
        pairwise_potentials = rnd.normal(size=(9, 3, 3))
        result_ad3 = inference_ad3(unary_potentials, pairwise_potentials,
                                   tree_edges, branch_and_bound=True)
        result_mp = inference_max_product(unary_potentials,
                                          pairwise_potentials, tree_edges)
        assert_array_equal(result_ad3, result_mp)


def test_iterative_max_product_chain():
    rnd = np.random.RandomState(0)
    chain = np.c_[np.arange(9), np.arange(1, 10)]
    for i in range(10):
        unary_potentials = rnd.normal(size=(10, 3))
        pairwise_potentials = rnd.normal(size=(9, 3, 3))
        result_ad3 = inference_ad3(unary_potentials, pairwise_potentials,
                                   chain, branch_and_bound=True)
        result_mp = iterative_max_product(unary_potentials,
                                          pairwise_potentials, chain)
        assert_array_equal(result_ad3, result_mp)


def test_iterative_max_product_tree():
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
    except:
        raise SkipTest("Not testing trees, scipy version >= 0.11 required")
    rnd = np.random.RandomState(0)
    for i in range(100):
        # generate random tree using mst
        graph = rnd.uniform(size=(10, 10))
        tree = minimum_spanning_tree(sparse.csr_matrix(graph))
        tree_edges = np.c_[tree.nonzero()]

        unary_potentials = rnd.normal(size=(10, 3))
        pairwise_potentials = rnd.normal(size=(9, 3, 3))
        result_ad3 = inference_ad3(unary_potentials, pairwise_potentials,
                                   tree_edges, branch_and_bound=True)
        result_mp = iterative_max_product(unary_potentials,
                                          pairwise_potentials, tree_edges)
    assert_array_equal(result_ad3, result_mp)


def test_max_product_binary_blocks():
    X, Y = generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  0,     # pairwise
                  -4, 0])
    crf = GridCRF(inference_method='max-product')
    crf.initialize(X, Y)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_max_product_multinomial_crf():
    X, Y = generate_blocks_multinomial(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1., 0., 0.,  # unary
                  0., 1., 0.,
                  0., 0., 1.,
                 .4,           # pairwise
                 -.3, .3,
                 -.5, -.1, .3])
    crf = GridCRF(inference_method='max-product')
    crf.initialize(X, Y)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)
