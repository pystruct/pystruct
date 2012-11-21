import numpy as np
import glpk            # Import the GLPK module


def solve_lp(unaries, pairwise, edges):
    n_nodes = len(unaries)
    n_edges = len(edges)

    # this will hold our constraints:
    matrix = []

    # columns are variables. n_nodes * n_states for nodes, n_edges * n_states ** 2 for edges
    n_variables = n_nodes * n_states + n_edges * n_states ** 2
    lp.cols.add(n_variables)         # Append columns to this instance

    for i, col in enumerate(lp.cols[:n_nodes * n_states]):
        col.name = "mu_%d=%d" % (i // n_states, i % n_states)

    for i, col in enumerate(lp.cols[n_nodes * n_states:]):
        edge = edges[i // n_states ** 2]
        state_pair = i % n_states ** 2
        col.name = "mu_%d,%d=%d,%d" % (edge[0], edge[1], state_pair // n_states, state_pair % n_states)

    for c in lp.cols:      # Iterate over all columns
        c.bounds = 0.0, None     # Set bound 0 <= xi < inf

    # rows are constraints. one per node, and n_nodes * n_states for pairwise   
    n_constraints = n_nodes + 2 * n_edges * n_states
    lp.rows.add(n_constraints)         # Append rows to this instance
    edges_offset = n_nodes * n_states  # offset to get to the edge variables in columns

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
        matrix.append((row_idx, vertex * n_states + state, -1))
        edge_var_index = edges_offset + edge * n_states ** 2
        if vertex_in_edge == 0:
            # first vertex in edge
            for j in xrange(n_states):
                matrix.append((row_idx, edge_var_index + state * n_states + j, 1))
        else:
            # second vertex in edge
            for j in xrange(n_states):
                matrix.append((row_idx, edge_var_index + j * n_states + state, 1))

        r.bounds = 0

    c = np.ravel(unaries)
    # pairwise:
    repeated_pairwise = np.repeat(np.hstack(pairwise)[np.newaxis, :], n_edges, axis=0).ravel()
    c = np.hstack([c, repeated_pairwise])
    lp.obj[:] = c.tolist()   # Set objective coefficients
    lp.matrix = matrix
    lp.simplex()           # Solve this LP with the simplex method
    print 'Z = %g;' % lp.obj.value,  # Retrieve and print obj func value
    print '; '.join('%s = %g' % (c.name, c.primal) for c in lp.cols)

if __name__ == "__main__":
    lp = glpk.LPX()        # Create empty problem instance
    lp.name = 'sample'     # Assign symbolic name to problem
    lp.obj.maximize = False # Set this as a maximization problem

    # create mrf problem:
    # two nodes, binary problem
    n_states = 2
    edges = [[0, 1]]
    unaries = [[1, 0], [0, 2]]
    # potts potential
    pairwise = [[0, 2], [2, 0]]

