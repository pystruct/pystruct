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
        y = iterative_max_product(unary_potentials, pairwise_potentials, edges)
    return y


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    n_vertices = len(unary_potentials)
    edge_hashes = edges[:, 0] + n_vertices * edges[:, 1]
    graph = edges_to_graph(edges, n_vertices)
    nodes, predecessors = csgraph.breadth_first_order(graph, 0, directed=False)
    predecessors = predecessors[nodes]
    # we store the message from pred to node in down_messages[node]
    down_messages = np.zeros((n_vertices, n_states))
    edge_potentials = []

    # up-pass
    # we store in up_messages the sum of all messages going into node
    up_messages = np.zeros((n_vertices, n_states))
    all_messages = dict()
    for node, pred in zip(nodes, predecessors)[::-1]:
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
        # node already got all up-going messages
        # take max, normalize, send up to parent
        going_up = up_messages[node] + unary_potentials[node] + pairwise.T
        going_up = going_up.max(axis=1)
        going_up -= going_up.max()
        up_messages[pred] += going_up
        all_messages[(node, pred)] = going_up

    # down-pass
    for node, pred, pairwise in zip(nodes, predecessors,
                                    edge_potentials[::-1]):
        if pred < 0:
            continue
        incoming = down_messages[pred] + pairwise + unary_potentials[pred]
        # add upgoing messages not coming from node
        incoming += up_messages[pred] - all_messages[(node, pred)]
        down_messages[node] = incoming.max(axis=1)
        down_messages[node] -= down_messages[node].max()

    return np.argmax(up_messages + down_messages + unary_potentials, axis=1)


def iterative_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=10):
    n_edges = len(edges)
    n_vertices, n_states = unary_potentials.shape
    messages = np.zeros((n_edges, 2, n_states))
    all_incoming = np.zeros((n_vertices, n_states))
    for i in xrange(max_iter):
        for e, (edge, pairwise) in enumerate(zip(edges, pairwise_potentials)):
            # update message from edge[0] to edge[1]
            update = (all_incoming[edge[0]] + pairwise.T +
                      unary_potentials[edge[0]]
                      - messages[e, 1])
            old_message = messages[e, 0].copy()
            messages[e, 0] = np.max(update, axis=1)
            messages[e, 0] -= np.max(messages[e, 0])
            all_incoming[edge[1]] += messages[e, 0] - old_message

            # update message from edge[1] to edge[0]
            update = (all_incoming[edge[1]] + pairwise +
                      unary_potentials[edge[1]]
                      - messages[e, 0])
            old_message = messages[e, 1].copy()
            messages[e, 1] = np.max(update, axis=1)
            messages[e, 1] -= np.max(messages[e, 1])
            all_incoming[edge[0]] += messages[e, 1] - old_message
    return np.argmax(all_incoming + unary_potentials, axis=1)
