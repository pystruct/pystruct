import numpy as np
import glpk            # Import the GLPK module

import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer

tracer = Tracer()


def solve_lp(unaries, edges, pairwise):
    if unaries.shape[1] != pairwise.shape[0]:
        raise ValueError("incompatible shapes of unaries"
                         " and pairwise potentials.")

    lp = glpk.LPX()          # Create empty problem instance
    lp.name = 'sample'       # Assign symbolic name to problem
    lp.obj.maximize = False  # Set this as a maximization problem
    n_nodes, n_states = unaries.shape
    n_edges = len(edges)

    # this will hold our constraints:
    matrix = []

    # columns are variables. n_nodes * n_states for nodes,
    # n_edges * n_states ** 2 for edges
    n_variables = n_nodes * n_states + n_edges * n_states ** 2
    lp.cols.add(n_variables)         # Append columns to this instance

    for i, col in enumerate(lp.cols[:n_nodes * n_states]):
        col.name = "mu_%d=%d" % (i // n_states, i % n_states)

    for i, col in enumerate(lp.cols[n_nodes * n_states:]):
        edge = edges[i // n_states ** 2]
        state_pair = i % n_states ** 2
        col.name = "mu_%d,%d=%d,%d" % (edge[0], edge[1],
                                       state_pair // n_states,
                                       state_pair % n_states)

    for c in lp.cols:      # Iterate over all columns
        c.bounds = 0.0, None     # Set bound 0 <= xi < inf

    # rows are constraints. one per node,
    # and n_nodes * n_states for pairwise
    n_constraints = n_nodes + 2 * n_edges * n_states
    lp.rows.add(n_constraints)         # Append rows to this instance
    # offset to get to the edge variables in columns
    edges_offset = n_nodes * n_states

    for i, r in enumerate(lp.rows[:n_nodes]):
        r.name = "summation %d" % i
        r.bounds = 1
        for j in xrange(n_states):
            matrix.append((i, i * n_states + j, 1))

    for i, r in enumerate(lp.rows[n_nodes:]):
        row_idx = i + n_nodes
        #print("i: %d" % i)
        edge = i // (2 * n_states)
        #print("edge: %d" % edge)
        state = (i % n_states)
        #print("state: %d" % state)
        vertex_in_edge = i % (2 * n_states) // n_states
        vertex = edges[edge][vertex_in_edge]
        #print("vertex: %d" % vertex)
        r.name = "marginalization edge %d[%d] state %d" % (edge, vertex, state)
        # for one vertex iterate over all states of the other vertex
        matrix.append((row_idx, int(vertex) * n_states + state, -1))
        edge_var_index = edges_offset + edge * n_states ** 2
        if vertex_in_edge == 0:
            # first vertex in edge
            for j in xrange(n_states):
                matrix.append((row_idx, edge_var_index
                               + state * n_states + j, 1))
        else:
            # second vertex in edge
            for j in xrange(n_states):
                matrix.append((row_idx, edge_var_index
                               + j * n_states + state, 1))

        r.bounds = 0

    coef = np.ravel(unaries)
    # pairwise:
    repeated_pairwise = np.repeat(np.hstack(pairwise)[np.newaxis, :],
                                  n_edges, axis=0).ravel()
    coef = np.hstack([coef, repeated_pairwise])
    lp.obj[:] = coef.tolist()   # Set objective coefficients
    lp.matrix = matrix
    lp.simplex()           # Solve this LP with the simplex method
    #print 'Z = %g;' % lp.obj.value,  # Retrieve and print obj func value
    #print '; '.join('%s = %g' % (c.name, c.primal) for c in lp.cols)
    res = np.array([c.primal for c in lp.cols])
    unary_variables = res[:n_nodes * n_states].reshape(n_nodes, n_states)
    pairwise_variables = res[n_nodes * n_states:].reshape(n_edges,
                                                          n_states ** 2)
    assert((np.abs(unary_variables.sum(axis=1) - 1) < 1e-4).all())
    assert((np.abs(pairwise_variables.sum(axis=1) - 1) < 1e-4).all())
    return unary_variables


def main():
    # create mrf problem:
    # two nodes, binary problem
    # potts potential
    pairwise = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    import toy_datasets as toy
    X, Y = toy.generate_blocks_multinomial(n_samples=1, noise=.5)
    x, y = X[0], Y[0]
    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    inds = inds.astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])
    x = x.reshape(-1, x.shape[-1])
    unary_assignment = solve_lp(-x, pairwise, edges)
    plt.matshow(np.argmax(unary_assignment, axis=1).reshape(y.shape))
    plt.show()
    tracer()

if __name__ == "__main__":
    main()
