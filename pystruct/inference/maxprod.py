import numpy as np
from scipy import sparse


def is_tree(n_vertices, edges):
    # tests if graph given by edges is a tree
    # checks for each connected component if the number of edges is smaller
    # than number of vertices.

    if len(edges) > n_vertices - 1:
        return False
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices)).tocsr()
    n_components, component_indicators = \
        sparse.csgraph.connected_components(graph, directed=False)
    if len(edges) > n_vertices - n_components:
        return False
    return True


def max_product(unary_potentials, pairwise_potentials, edges):
    if is_tree(n_vertices=len(unary_potentials), edges=edges):
        y = tree_max_product(unary_potentials, pairwise_potentials, edges)
    else:
        y = iterative_max_product(unary_potentials, pairwise_potentials)
    return y