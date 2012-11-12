import numpy as np
import scipy.sparse as sp
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer
tracer = Tracer()


def grid_linear_programming(x, weights):
    # create LP formulation for cvxopt
    # one variable per node and edge
    #tracer()
    unary_weights = weights[0]
    assert(unary_weights == 1)
    pw = weights[1]
    #pairwise_weights = np.array([[0, pw], [0, pw]])
    width, height = x.shape[0], x.shape[1]
    n_nodes = width * height
    n_states = x.shape[2]

    # unaries are stored as node_0: state_0, state_1
    unary_potentials = x.ravel()

    # pairwise potentials
    # first all right, then all down
    # edge01, state 00, 01, 10, 11, edge12,...
    pw = np.array([[0, pw], [pw, 0]])

    n_edges = width * (height - 1) + height * (width - 1)
    n_vars = n_states * n_nodes + n_states ** 2 * n_edges
    edge_penalties = np.repeat(pw, n_edges).ravel()

    #together:
    #c = np.hstack([unary_potentials, edge_penalties])
    c = np.hstack([unary_potentials, np.repeat(100, n_edges * n_states ** 2)])

    # non-negativity constraints for all variables
    G = -np.eye(n_vars)
    h = np.zeros(n_vars)
    #G = -np.eye(n_nodes * n_states)
    #h = np.zeros(n_nodes * n_states)

    # summation constraints for unaries: state variables have to sum to one
    Aun = np.hstack(np.repeat([np.eye(n_nodes)], n_states, axis=0))
    Aun = np.hstack([Aun, np.zeros((n_nodes,
                                   n_edges * n_states ** 2))])
    # add loads of zeros for pairwise variables
    bun = np.ones(n_nodes)

    # summation constraint for pairwise: summing out one
    # state has to yield unary for the other one.
    # leading to 2 * n_edges  * n_states constraints
    constraints = np.zeros((2 * n_edges * n_states,
            n_states * n_nodes + n_states ** 2 * n_edges))
    # edges to the right
    # off is offset for edge variables, i.e. number of node variables
    off = n_nodes * n_states
    i = 0
    for row in xrange(height):
        for variable in xrange(width - 1):
            for state in xrange(n_states):
                # index of variable:
                constraints[i, n_states * (row * width + variable) + state] = -1
                # select edge with this node as left variable
                edge_ind = off + n_states ** 2 * (row * (width - 1) + variable)
                # states are sorted lexically
                for other_state in xrange(n_states):
                    constraints[i, edge_ind + state * n_states + other_state] = +1
                i += 1
    # edges to the left
    for row in xrange(height):
        for variable in xrange(width - 1):
            for state in xrange(n_states):
                # index of variable:
                constraints[i, n_states * (row * width + variable + 1) + state] = -1
                # select edge with this node as left variable
                edge_ind = off + n_states ** 2 * (row * (width - 1) + variable)
                # states are sorted lexically
                for other_state in xrange(n_states):
                    constraints[i, edge_ind + other_state * n_states + state] = +1
                i += 1
    # edges down
    # shift by number of horizontal edges
    off += height * (width - 1) * n_states ** 2
    for row in xrange(height - 1):
        for variable in xrange(width):
            for state in xrange(n_states):
                # index of variable:
                constraints[i, n_states * (row * width + variable) + state] = -1
                # select edge with this node as left variable
                edge_ind = off + n_states ** 2 * (row * width + variable)
                # states are sorted lexically
                for other_state in xrange(n_states):
                    constraints[i, edge_ind + state * n_states + other_state] = +1
                i += 1
    # edges up
    for row in xrange(height - 1):
        for variable in xrange(width):
            for state in xrange(n_states):
                # index of variable:
                constraints[i, n_states * ((row + 1) * width + variable) + state] = -1
                # select edge with this node as left variable
                edge_ind = off + n_states ** 2 * (row * width + variable)
                # states are sorted lexically
                for other_state in xrange(n_states):
                    constraints[i, edge_ind + other_state * n_states + state] = +1
                i += 1
    tracer()



    edge_consistency = np.eye(n_nodes * n_states)
    # subtract node state from sum of pairwise states
    Apw = sp.hstack([node_states, edge_consistency])
    bpw = np.zeros(n_nodes * n_states)

    A = sp.vstack([Aun, Apw])
    b = np.hstack([bun, bpw])
    #A = Aun
    #b = bun
    c_ = matrix(c)
    G_ = matrix(G)
    h_ = matrix(h)
    # we need to remove the last row to make A full rank
    #A_ = matrix(A.toarray()[:-1, :])
    #b_ = matrix(b[:-1])
    A_ = matrix(A.toarray())
    b_ = matrix(b)
    solvers.options['feastol'] = 1e-3
    #sol = solvers.lp(c_, G_, h_, A_, b_)
    sol = solvers.lp(c_, G_, h_, A_, b_)
    result = np.asarray(sol['x'])
    node_vars = result[:n_nodes * n_states]
    node_vars = node_vars.reshape(n_states, width, height)
    edge_vars = result[n_nodes * n_states:]
    right_edges = edge_vars[: (width - 1) * height * n_states ** 2]
    right_edges = right_edges.reshape(n_states ** 2, width - 1, height)
    down_edges = edge_vars[(width - 1) * height * n_states ** 2:]
    down_edges = down_edges.reshape(n_states ** 2, height - 1, width)
    plt.matshow(node_vars[0], vmin=0, vmax=1)
    plt.matshow(down_edges[0], vmin=0, vmax=1)
    plt.matshow(down_edges[1], vmin=0, vmax=1)
    plt.matshow(down_edges[2], vmin=0, vmax=1)
    plt.matshow(down_edges[3], vmin=0, vmax=1)

    plt.show()
    tracer()
    print(result)


def main():
    weights = np.array([1, -2])
    Y = np.ones((3, 4))
    Y[:, :2] = -1
    #X = Y + 1. * np.random.normal(size=Y.shape)
    X = Y
    X = np.c_['2,3,0', -X, np.zeros_like(X)]
    Y = (Y > 0).astype(np.int32)
    grid_linear_programming(X, weights)


if __name__ == "__main__":
    main()
