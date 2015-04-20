import numpy as np
from scipy import sparse

from .common import _validate_params
from ..utils.graph_functions import is_forest


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
    from ._viterbi import viterbi
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    if is_chain(edges=edges, n_vertices=len(unary_potentials)):
        y = viterbi(unary_potentials.astype(np.float).copy(),
                    # sad second copy b/c numpy 1.6
                    np.array(pairwise_potentials, dtype=np.float))
    elif is_forest(edges=edges, n_vertices=len(unary_potentials)):
        y = tree_max_product(unary_potentials, pairwise_potentials, edges)
    else:
        y = iterative_max_product(unary_potentials, pairwise_potentials, edges,
                                  max_iter=max_iter, damping=damping)
    return y


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    n_vertices, n_states = unary_potentials.shape
    parents = -np.ones(n_vertices, dtype=np.int)
    visited = np.zeros(n_vertices, dtype=np.bool)
    neighbors = [[] for i in range(n_vertices)]
    pairwise_weights = [[] for i in range(n_vertices)]
    for pw, edge in zip(pairwise_potentials, edges):
        neighbors[edge[0]].append(edge[1])
        pairwise_weights[edge[0]].append(pw)
        neighbors[edge[1]].append(edge[0])
        pairwise_weights[edge[1]].append(pw.T)

    messages_forward = np.zeros((n_vertices, n_states))
    messages_backward = np.zeros((n_vertices, n_states))
    pw_forward = np.zeros((n_vertices, n_states, n_states))
    # build a breadth first search of the tree
    traversal = []
    lonely = 0
    while lonely < n_vertices:
        for i in range(lonely, n_vertices):
            if not visited[i]:
                queue = [i]
                lonely = i + 1
                visited[i] = True
                break
            lonely = n_vertices

        while queue:
            node = queue.pop(0)
            traversal.append(node)
            for pw, neighbor in zip(pairwise_weights[node], neighbors[node]):
                if not visited[neighbor]:
                    parents[neighbor] = node
                    queue.append(neighbor)
                    visited[neighbor] = True
                    pw_forward[neighbor] = pw

                elif not parents[node] == neighbor:
                    raise ValueError("Graph not a tree")
    # messages from leaves to root
    for node in traversal[::-1]:
        parent = parents[node]
        if parent != -1:
            message = np.max(messages_backward[node] + unary_potentials[node] +
                             pw_forward[node], axis=1)
            message -= message.max()
            messages_backward[parent] += message
    # messages from root back to leaves
    for node in traversal:
        parent = parents[node]
        if parent != -1:
            message = messages_forward[parent] + unary_potentials[parent] + pw_forward[node].T
            # leaves to root messages from other children
            message += messages_backward[parent] - np.max(messages_backward[node]
                                                          + unary_potentials[node]
                                                          + pw_forward[node], axis=1)
            message = message.max(axis=1)
            message -= message.max()
            messages_forward[node] += message

    return np.argmax(unary_potentials + messages_forward + messages_backward, axis=1)


def iterative_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=10, damping=.5, tol=1e-5):
    n_edges = len(edges)
    n_vertices, n_states = unary_potentials.shape
    messages = np.zeros((n_edges, 2, n_states))
    all_incoming = np.zeros((n_vertices, n_states))
    for i in range(max_iter):
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
