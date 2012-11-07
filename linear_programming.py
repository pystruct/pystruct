import toy_datasets as toy
import numpy as np
import scipy.sparse as sp


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

    # unaries are stored as "first all state0, then all state1"
    unary_potentials = x.ravel()
    # pairwise potentials
    # first all right, then all down
    # first all of state 00, then 01, 10, 11
    n_edges = width * (height - 1) + height * (width - 1)
    n_vars = n_states * n_nodes + n_states ** 2 * n_edges
    column_zeros = np.zeros(width * (height - 1))
    column_penalty = np.repeat(pw, width * (height - 1))
    down = np.hstack([column_zeros, column_penalty,
                      column_penalty, column_zeros])
    row_zeros = np.zeros(height * (width - 1))
    row_penalty = np.repeat(pw, height * (width - 1))
    right = np.hstack([row_zeros, row_penalty,
                       row_penalty, row_zeros])

    #together:
    c = np.hstack([unary_potentials, down, right])

    # non-negativity constraints for all variables
    G = -np.eye(n_vars)
    h = np.zeros(n_vars)

    # summation constraints for unaries: state variables have to sum to one
    Aun = np.hstack(np.repeat([np.eye(n_nodes)], n_states, axis=0))
    Aun = np.hstack([Aun, np.zeros((n_nodes,
                                   n_edges * n_states ** 2))])
    # add loads of zeros for pairwise variables
    bun = np.ones(n_nodes)

    # summation constraint for pairwise: have to sum to unary
    # build incidence matrix (which vertex is part of which edge)
    # for all nodes first the zero state, then the one state
    node_states = -np.eye(n_nodes * n_states)
    # for the edges to the right
    # create a block matrix
    #
    # un_0_pw_00, un_0_pw01, un_0_pw_10, un_0_pw_11
    # un_1_pw_00, un_1_pw01, un_1_pw_10, un_1_pw_11

    un_0_pw_00 = sp.block_diag([np.eye(width, width - 1)
                              + np.eye(width, width - 1, -1)] * height)
    un_1_pw_00 = np.zeros((n_nodes, height * (width - 1)))
    pw_00 = sp.vstack([un_0_pw_00, un_1_pw_00])

    un_0_pw_01 = sp.block_diag([np.eye(width, width - 1)] * height)
    un_1_pw_01 = sp.block_diag([np.eye(width, width - 1, -1)] * height)
    pw_01 = sp.vstack([un_0_pw_01, un_1_pw_01])

    un_0_pw_10 = sp.block_diag([np.eye(width, width - 1, -1)] * height)
    un_1_pw_10 = sp.block_diag([np.eye(width, width - 1)] * height)
    pw_10 = sp.vstack([un_0_pw_10, un_1_pw_10])

    un_0_pw_11 = np.zeros((n_nodes, height * (width - 1)))
    un_1_pw_11 = sp.block_diag([np.eye(width, width - 1)
                              + np.eye(width, width - 1, -1)] * height)
    pw_11 = sp.vstack([un_0_pw_11, un_1_pw_11])

    to_right = sp.hstack([pw_00, pw_01, pw_10, pw_11])

    # same for the edgeds down
    un_0_pw_00 = sp.block_diag([np.eye(height, height - 1)
                              + np.eye(height, height - 1, -1)] * width)
    un_1_pw_00 = np.zeros((n_nodes, width * (height - 1)))
    pw_00 = sp.vstack([un_0_pw_00, un_1_pw_00])

    un_0_pw_01 = sp.block_diag([np.eye(height, height - 1)] * width)
    un_1_pw_01 = sp.block_diag([np.eye(height, height - 1, -1)] * width)
    pw_01 = sp.vstack([un_0_pw_01, un_1_pw_01])

    un_0_pw_10 = sp.block_diag([np.eye(height, height - 1, -1)] * width)
    un_1_pw_10 = sp.block_diag([np.eye(height, height - 1)] * width)
    pw_10 = sp.vstack([un_0_pw_10, un_1_pw_10])

    un_0_pw_11 = np.zeros((n_nodes, width * (height - 1)))
    un_1_pw_11 = sp.block_diag([np.eye(height, height - 1)
                              + np.eye(height, height - 1, -1)] * width)
    pw_11 = sp.vstack([un_0_pw_11, un_1_pw_11])

    down = sp.hstack([pw_00, pw_01, pw_10, pw_11])

    # subtract node state from sum of pairwise states
    Apw = sp.hstack([node_states, to_right, down])
    bpw = np.zeros(n_nodes * n_states)

    A = sp.vstack([Aun, Apw])
    b = np.hstack([bun, bpw])
    tracer()
    [c, G, h, A, b]

    #bpw = np.zeros(n_nodes)


def main():
    X, Y = toy.generate_blocks(n_samples=10)
    x, y = X[0], Y[0]
    weights = np.array([1, -2])
    grid_linear_programming(x, weights)


if __name__ == "__main__":
    main()
