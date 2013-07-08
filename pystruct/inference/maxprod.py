import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from inference_methods import _validate_params


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
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    n_vertices = len(unary_potentials)
    edge_hashes = edges[:, 0] + n_vertices * edges[:, 1]
    graph = edges_to_graph(edges, n_vertices)
    nodes, predecessors = csgraph.breadth_first_order(graph, 0, directed=False)
    # we store the message from pred to node in down_messages[node]
    down_messages = np.zeros((n_vertices, n_states))
    edge_potentials = []
    # down-pass
    for node, pred in zip(nodes, predecessors[nodes]):
        if pred < 0:
            edge_potentials.append([])
            continue
        # we need to get the pairwise potentials corresponding to
        # the edge between predecessor and node
        edge_number = np.where(edge_hashes == node + pred * n_vertices)[0]
        if len(edge_number):
            pairwise = pairwise_potentials[edge_number[0]]
        else:
            edge_number = np.where(edge_hashes == n_vertices * node + pred)[0]
            pairwise = pairwise_potentials[edge_number[0]].T
        edge_potentials.append(pairwise)
        incoming = down_messages[pred] + pairwise + unary_potentials[pred]
        down_messages[node] = incoming.max(axis=1)
        down_messages[node] -= down_messages[node].max()

    # up-pass
    # we store in up_messages the sum of all messages going into node
    up_messages = np.zeros((n_vertices, n_states))
    for node, pred, pairwise in zip(nodes, predecessors,
                                    edge_potentials)[::-1]:
        if pred < 0:
            continue
        # node already got all up-going messages
        # take max, normalize, send up to parent
        going_up = up_messages[node] + unary_potentials[node] + pairwise.T
        going_up = going_up.max(axis=1)
        going_up -= going_up.max()
        up_messages[pred] += going_up

    return np.argmax(up_messages + down_messages + unary_potentials, axis=1)


def iterative_max_product():
    pass
