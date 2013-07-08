import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def edges_to_graph(edges, n_vertices=None):
    if n_vertices is None:
        n_vertices = np.max(edges) + 1
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices)).tocsr()
    return graph


def is_tree(edges, n_vertices=None):
    """Check if edges specify a tree.

    Parameters
    ----------
    edges : nd-array of int
        Edges of a graph. Shape (n_edges, 2).
    n_vertices : int or None
        Number of vertices in the graph. If None, it is inferred from the
        edges.
    """
    if n_vertices is None:
        n_vertices = np.max(edges) + 1
    if len(edges) > n_vertices - 1:
        return False
    graph = edges_to_graph(edges, n_vertices)
    n_components, component_indicators = \
        csgraph.connected_components(graph, directed=False)
    if len(edges) > n_vertices - n_components:
        return False
    return True


def inference_max_product(unary_potentials, pairwise_potentials, edges):
    """Max-product inference.

    In case the edges specify a tree, dynamic programming is used
    producing a result in only a single pass.
    """
    if is_tree(edges=edges, n_vertices=len(unary_potentials)):
        y = tree_max_product(unary_potentials, pairwise_potentials, edges)
    else:
        y = iterative_max_product(unary_potentials, pairwise_potentials)
    return y


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    graph = edges_to_graph(edges, len(unary_potentials))
    nodes, predecessors = csgraph.depth_first_order(graph, 0, directed=False)
    for i, node in enumerate(nodes):
        for neighbor in get_neighbors(edges, node):
            if neighbor == predecessors[i]:
                continue

    from IPython.core.debugger import Tracer
    Tracer()()


def iterative_max_product():
    pass
