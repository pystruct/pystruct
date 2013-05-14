import numpy as np
import cvxopt
import cvxopt.solvers


def lp_general_graph(unaries, edges, edge_weights):
    if unaries.shape[1] != edge_weights.shape[1]:
        raise ValueError("incompatible shapes of unaries"
                         " and edge_weights.")
    if edge_weights.shape[1] != edge_weights.shape[2]:
        raise ValueError("Edge weights not square!")
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError("Number of edge weights different from number of"
                         "edges")

    n_nodes, n_states = unaries.shape
    n_edges = len(edges)

    # variables: n_nodes * n_states for nodes,
    # n_edges * n_states ** 2 for edges
    n_variables = n_nodes * n_states + n_edges * n_states ** 2

    # constraints: one per node,
    # and n_nodes * n_states for pairwise
    n_constraints = n_nodes + 2 * n_edges * n_states

    # offset to get to the edge variables in columns
    edges_offset = n_nodes * n_states
    # store constraints as triple (data, I, J)
    data, I, J = [], [], []

    # summation constraints
    for i in xrange(n_nodes):
        for j in xrange(n_states):
            data.append(1)
            I.append(i)
            J.append(i * n_states + j)
            #constraints[i, i * n_states + j] = 1

    # edge marginalization constraint
    for i in xrange(2 * n_edges * n_states):
        row_idx = i + n_nodes
        #print("i: %d" % i)
        edge = i // (2 * n_states)
        #print("edge: %d" % edge)
        state = (i % n_states)
        #print("state: %d" % state)
        vertex_in_edge = i % (2 * n_states) // n_states
        vertex = edges[edge][vertex_in_edge]
        #print("vertex: %d" % vertex)
        # for one vertex iterate over all states of the other vertex
        #[row_idx, int(vertex) * n_states + state] = -1
        data.append(-1)
        I.append(row_idx)
        J.append(int(vertex) * n_states + state)
        edge_var_index = edges_offset + edge * n_states ** 2
        if vertex_in_edge == 0:
            # first vertex in edge
            for j in xrange(n_states):
                data.append(1)
                I.append(row_idx)
                J.append(edge_var_index + state * n_states + j)
                #[row_idx, edge_var_index + state * n_states + j] = 1
        else:
            # second vertex in edge
            for j in xrange(n_states):
                data.append(1)
                I.append(row_idx)
                J.append(edge_var_index + j * n_states + state)
                #[row_idx, edge_var_index + j * n_states + state] = 1

    coef = np.ravel(unaries)
    # pairwise:
    repeated_pairwise = edge_weights.ravel()
    coef = np.hstack([coef, repeated_pairwise])
    c = cvxopt.matrix(coef)
    # for positivity inequalities
    G = cvxopt.spdiag(cvxopt.matrix(-np.ones(n_variables)))
    #G = cvxopt.matrix(-np.eye(n_variables))
    h = cvxopt.matrix(np.zeros(n_variables))  # for positivity inequalities
    # unary and pairwise summation constratints
    A = cvxopt.spmatrix(data, I, J)
    b_ = np.zeros(n_constraints)  # zeros for pairwise summation constraints
    b_[:n_nodes] = 1    # ones for unary cummation constraints
    b = cvxopt.matrix(b_)

    # silence glpk
    cvxopt.solvers.options['LPX_K_MSGLEV'] = False

    result = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk')

    x = np.array(result['x'])
    unary_variables = x[:n_nodes * n_states].reshape(n_nodes, n_states)
    pairwise_variables = x[n_nodes * n_states:].reshape(n_edges, n_states ** 2)
    assert((np.abs(unary_variables.sum(axis=1) - 1) < 1e-4).all())
    assert((np.abs(pairwise_variables.sum(axis=1) - 1) < 1e-4).all())
    return unary_variables, pairwise_variables, result['primal objective']


def solve_lp(unaries, edges, pairwise):
    if unaries.shape[1] != pairwise.shape[0]:
        raise ValueError("incompatible shapes of unaries"
                         " and pairwise potentials.")

    n_edges = len(edges)
    edge_weights = np.repeat(pairwise[np.newaxis, :, :], n_edges, axis=0)
    return lp_general_graph(unaries, edges, edge_weights)


def main():
    import pystruct.toy_datasets as toy
    import matplotlib.pyplot as plt
    # create mrf model:
    pairwise = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    X, Y = toy.generate_blocks_multinomial(n_samples=1, noise=.5)
    x, y = X[0], Y[0]
    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    inds = inds.astype(np.int64)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])
    x = x.reshape(-1, x.shape[-1])
    unary_assignment, pairwise_assignment, energy = solve_lp(-x, edges,
                                                             pairwise)
    plt.matshow(np.argmax(unary_assignment, axis=1).reshape(y.shape))
    plt.show()

if __name__ == "__main__":
    main()
