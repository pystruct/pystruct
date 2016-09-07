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

    n_nodes, n_states = map(int, unaries.shape)
    n_edges = len(edges)

    # variables: n_nodes * n_states for nodes,
    # n_edges * n_states ** 2 for edges
    n_variables = n_nodes * n_states + n_edges * n_states ** 2

    # constraints: one per node,
    # and n_nodes * n_states for pairwise minus one redundant per edge
    n_constraints = n_nodes + n_edges * (2 * n_states - 1)

    # offset to get to the edge variables in columns
    edges_offset = n_nodes * n_states
    # store constraints as triple (data, I, J)
    data, I, J = [], [], []

    # summation constraints
    data += [1.0]*n_nodes*n_states
    for i in range(n_nodes):
        I += [i]*n_states
        i_n = i * n_states
        J += [i_n + j for j in range(n_states)]
        # for j in range(n_states):
            # data.append(1)
            # I.append(i)
            # J.append(i * n_states + j)
            #constraints[i, i * n_states + j] = 1
    # we row_idx tracks constraints = rows in constraint matrix
    row_idx = n_nodes
    # edge marginalization constraint
    for i in range(2 * n_edges * n_states):
        edge = i // (2 * n_states)
        state = (i % n_states)
        vertex_in_edge = i % (2 * n_states) // n_states
        vertex = edges[edge][vertex_in_edge]
        if vertex_in_edge == 1 and state == n_states - 1:
            # the last summation constraint is redundant.
            continue
        # for one vertex iterate over all states of the other vertex
        #[row_idx, int(vertex) * n_states + state] = -1
        data.append(-1)
        I.append(row_idx)
        J.append(int(vertex) * n_states + state)
        edge_var_index = edges_offset + edge * n_states ** 2
        data += [1]*n_states
        I += [row_idx]*n_states
        if vertex_in_edge == 0:
            # first vertex in edge
            for j in range(n_states):
                # data.append(1)
                # I.append(row_idx)
                J.append(edge_var_index + state * n_states + j)
                #[row_idx, edge_var_index + state * n_states + j] = 1
        else:
            # second vertex in edge
            for j in range(n_states):
                # data.append(1)
                # I.append(row_idx)
                J.append(edge_var_index + j * n_states + state)
                #[row_idx, edge_var_index + j * n_states + state] = 1
        row_idx += 1

    coef = np.ravel(unaries)
    # pairwise:
    repeated_pairwise = edge_weights.ravel()
    c = cvxopt.matrix([coef, repeated_pairwise])
    # coef = np.hstack([coef, repeated_pairwise])
    # c = cvxopt.matrix(coef, tc='d')
    # for positivity inequalities
    G = cvxopt.spdiag([-1.0]*n_variables)
    #G = cvxopt.spdiag(cvxopt.matrix(-np.ones(n_variables)))
    #G = cvxopt.matrix(-np.eye(n_variables))
    h = cvxopt.matrix(0, (n_variables,1), 'd')
    #h = cvxopt.matrix(np.zeros(n_variables))  # for positivity inequalities
    # unary and pairwise summation constratints
    A = cvxopt.spmatrix(data, I, J)
    assert(n_constraints == A.size[0])
    b_ = np.zeros(A.size[0])  # zeros for pairwise summation constraints
    b_[:n_nodes] = 1    # ones for unary summation constraints
    b = cvxopt.matrix(b_)

    # don't be verbose.
    show_progress_backup = cvxopt.solvers.options.get('show_progress', False)
    cvxopt.solvers.options['show_progress'] = False

    # print("Standard LP solver")
    # result = cvxopt.solvers.lp(c, G, h, A, b)

    # print("Cone solver with KKT factorization")
    result = _solve_lp_kkt(c, G, h, A, b)
    cvxopt.solvers.options['show_progress'] = show_progress_backup

    x = np.array(result['x'])
    unary_variables = x[:n_nodes * n_states].reshape(n_nodes, n_states)
    pairwise_variables = x[n_nodes * n_states:].reshape(n_edges, n_states ** 2)
    assert((np.abs(unary_variables.sum(axis=1) - 1) < 1e-4).all())
    assert((np.abs(pairwise_variables.sum(axis=1) - 1) < 1e-4).all())
    return unary_variables, pairwise_variables, result['primal objective']

def _solve_lp_kkt(c, G, h, A, b):
    """
    Solves the LP by tackling the KKT system directly.  Note that we do not actually
    need G as an input argument but is left here for completeness
    """
    n, m = G.size
    p, q = A.size

    dims = {'l':n, 'q':[], 's':[]}

    def G_func(x, y, alpha=1.0, beta=0.0, trans='N'):
        # note that  G = D, is a diagonal matrix with elements -1 (should always be true).
        # So that G = G^{T}, and we do not need to perform the test in theory
        if trans == 'N':
            y[:] =  -alpha*x + beta*y
        else:
            y[:] =  -alpha*x + beta*y

    def KKT_func(W):
        '''
        Solves the fully reduced system through a series of non-sparse linear
        systems.  They consist of the following three equation (in sequence)
        y = (A(W^{T}W)A^{T})^{-1} (A(W^{T}W) [bx - (W^{T}W)^{-1} bz] - by)
        x = (W^{T}W)[bx - (W^{T}W)^{-1} bz - A^{T}y]
        z = (-W^{T}W)^{-1} (bz - Gx)
        '''
        # Denote d = (W^{T}W), di = (W^{T}W)^{-1}
        d, di = W['d']**2, W['di']**2

        # because we have to use A(W^{T}W) multiple times, we do our
        # multiplication here and store
        AW = A*cvxopt.spdiag(d)
        LHS = AW*A.T

        # "Factorizing cholmod LHS of A(W^{T}W)A^{T}
        Fs = cvxopt.cholmod.symbolic(LHS)
        cvxopt.cholmod.numeric(LHS, Fs)

        def f(x, y, z):
            # a pre-compute of rx = bx - (W^{T}W)^{-1} bz
            rx = x - cvxopt.mul(di, z)
            y[:] = AW*rx - y
            cvxopt.cholmod.solve(Fs, y)

            x[:] = cvxopt.mul(d, rx - A.T*y)
            z[:] = cvxopt.mul(-W['di'], z + x)

        return f

    result = cvxopt.solvers.conelp(c, G_func, h, dims, A, b, kktsolver=KKT_func)
    return(result)


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
