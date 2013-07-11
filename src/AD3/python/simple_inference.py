import itertools
import numpy as np
#from numpy.testing.utils import assert_array_almost_equal
from . import ad3


def simple_grid(unaries, pairwise, verbose=1):
    height, width, n_states = unaries.shape

    factor_graph = ad3.PFactorGraph()

    multi_variables = []
    for i in xrange(height):
        multi_variables.append([])
        for j in xrange(width):
            new_variable = factor_graph.create_multi_variable(n_states)
            for state in xrange(n_states):
                new_variable.set_log_potential(state, unaries[i, j, state])
            multi_variables[i].append(new_variable)

    for i, j in itertools.product(xrange(height), xrange(width)):
        if (j > 0):
            #horizontal edge
            edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
            factor_graph.create_factor_dense(edge_variables, pairwise.ravel())

        if (i > 0):
            #horizontal edge
            edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
            factor_graph.create_factor_dense(edge_variables, pairwise.ravel())

    factor_graph.set_eta_ad3(.1)
    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(5000)
    factor_graph.set_verbosity(verbose)
    value, marginals, edge_marginals = factor_graph.solve_lp_map_ad3()
    marginals = np.array(marginals).reshape(unaries.shape)
    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value


def general_graph(unaries, edges, edge_weights, verbose=1, n_iterations=1000,
                  eta=.1, exact=False):
    if unaries.shape[1] != edge_weights.shape[1]:
        raise ValueError("incompatible shapes of unaries"
                         " and edge_weights.")
    if edge_weights.shape[1] != edge_weights.shape[2]:
        raise ValueError("Edge weights need to be of shape "
                         "(n_edges, n_states, n_states)!")
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError("Number of edge weights different from number of"
                         "edges")

    factor_graph = ad3.PFactorGraph()
    n_states = unaries.shape[-1]

    multi_variables = []
    for u in unaries:
        new_variable = factor_graph.create_multi_variable(n_states)
        for state, cost in enumerate(u):
            new_variable.set_log_potential(state, cost)
        multi_variables.append(new_variable)

    for i, e in enumerate(edges):
            edge_variables = [multi_variables[e[0]], multi_variables[e[1]]]
            factor_graph.create_factor_dense(edge_variables,
                                             edge_weights[i].ravel())

    factor_graph.set_eta_ad3(eta)
    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(n_iterations)
    factor_graph.set_verbosity(verbose)
    factor_graph.fix_multi_variables_without_factors()
    if exact:
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_exact_map_ad3()
    else:
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_lp_map_ad3()
    marginals = np.array(marginals).reshape(unaries.shape)

    #assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)
    solver_string = ["integral", "fractional", "infeasible", "unsolved"]

    return marginals, edge_marginals, value, solver_string[solver_status]
