import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .common import _validate_params
from ._viterbi import viterbi


def edges_to_graph(edges, n_vertices=None):
    if n_vertices is None:
        n_vertices = np.max(edges) + 1
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices)).tocsr()
    return graph


def is_chain(edges, n_vertices):
    """Check if edges specify a chain and are in order."""
    return (np.all(edges[:, 0] == np.arange(0, n_vertices - 1))
            and np.all(edges[:, 1] == np.arange(1, n_vertices)))


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


def inference_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=30, damping=0.5, tol=1e-5, relaxed=None):
    """Max-product inference.

    In case the edges specify a tree, dynamic programming is used
    producing a result in only a single pass.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    max_iter : int (default=10)
        Maximum number of iterations. Ignored if graph is a tree.

    damping : float (default=.5)
        Daming of messages in loopy message passing.
        Ignored if graph is a tree.

    tol : float (default=1e-5)
        Stopping tollerance for loopy message passing.
    """
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    if is_chain(edges=edges, n_vertices=len(unary_potentials)):
        y = viterbi(unary_potentials.astype(np.float).copy(),
                    # sad second copy b/c numpy 1.6
                    np.array(pairwise_potentials, dtype=np.float))
    elif is_tree(edges=edges, n_vertices=len(unary_potentials)):
        y = tree_max_product(unary_potentials, pairwise_potentials, edges)
    else:
        y = iterative_max_product(unary_potentials, pairwise_potentials, edges,
                                  max_iter=max_iter, damping=damping)
    return y


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    n_vertices, n_states = unary_potentials.shape
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
                          max_iter=10, damping=.5, tol=1e-5):
    n_edges = len(edges)
    n_vertices, n_states = unary_potentials.shape
    messages = np.zeros((n_edges, 2, n_states))
    all_incoming = np.zeros((n_vertices, n_states))
    for i in xrange(max_iter):
        diff = 0
        for e, (edge, pairwise) in enumerate(zip(edges, pairwise_potentials)):
            # update message from edge[0] to edge[1]
            update = (all_incoming[edge[0]] + pairwise.T +
                      unary_potentials[edge[0]]
                      - messages[e, 1])
            old_message = messages[e, 0].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(new_message)
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 0] = new_message
            update = new_message - old_message
            all_incoming[edge[1]] += update
            diff += np.abs(update).sum()

            # update message from edge[1] to edge[0]
            update = (all_incoming[edge[1]] + pairwise +
                      unary_potentials[edge[1]]
                      - messages[e, 0])
            old_message = messages[e, 1].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(messages[e, 1])
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 1] = new_message
            update = new_message - old_message
            all_incoming[edge[0]] += update
            diff += np.abs(update).sum()
        if diff < tol:
            break
    return np.argmax(all_incoming + unary_potentials, axis=1)
