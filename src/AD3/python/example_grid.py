import itertools
import numpy as np
import matplotlib.pyplot as plt

import ad3

grid_size = 20
num_states = 5

factor_graph = ad3.PFactorGraph()

multi_variables = []
random_grid = np.random.uniform(size=(grid_size, grid_size, num_states))
for i in xrange(grid_size):
    multi_variables.append([])
    for j in xrange(grid_size):
        new_variable = factor_graph.create_multi_variable(num_states)
        for state in xrange(num_states):
            new_variable.set_log_potential(state, random_grid[i, j, state])
        multi_variables[i].append(new_variable)

alpha = .5
potts_matrix = alpha * np.eye(num_states)
potts_potentials = potts_matrix.ravel().tolist()

for i, j in itertools.product(xrange(grid_size), repeat=2):
    if (j > 0):
        #horizontal edge
        edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)

    if (i > 0):
        #horizontal edge
        edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)


factor_graph.set_eta_ad3(.1)
factor_graph.adapt_eta_ad3(True)
factor_graph.set_max_iterations_ad3(1000)
value, marginals, edge_marginals, solver_status =\
    factor_graph.solve_lp_map_ad3()

res = np.array(marginals).reshape(20, 20, 5)
plt.matshow(np.argmax(res, axis=-1), vmin=0, vmax=4)
plt.matshow(np.argmax(random_grid, axis=-1), vmin=0, vmax=4)
plt.show()
