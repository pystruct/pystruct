import numpy as np
from pystruct.utils.graph import chow_liu_tree
from numpy.testing import assert_array_equal

def test_chow_liu_tree_small():
    # choose y so largest MI goes to (0, 1) and (1, 2) edges
    y = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    edges_expected = np.array([[0, 1], [1, 2]])

    edges = chow_liu_tree(y)
    
    assert_array_equal(edges, edges_expected)

def test_chow_liu_tree_zero_mi():
    # choose y so that MI is nonzero between (0, 1) and 0 between (1, 2)
    y = np.array([[1, 0, 1], [0, 1, 1]])
    edges_expected = np.array([[0, 1]])

    edges = chow_liu_tree(y)
    
    assert_array_equal(edges, edges_expected)
