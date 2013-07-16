from collections import defaultdict
from time import time

import numpy as np
import matplotlib.pyplot as plt

from pystruct.inference import (inference_dispatch, compute_energy,
                                get_installed)
from pystruct.utils import make_grid_edges

size = 10
n_runs = 10
#inference_methods = get_installed(['ad3', 'qpbo', 'lp', 'ogm'])
inference_methods = get_installed(['ad3', 'qpbo', 'unary',
                                   ('ad3', {'branch_and_bound':True}),
                                   ('ogm', {'alg': 'bp'}),
                                   ('ogm', {'alg': 'dd'}),
                                   ('ogm', {'alg': 'trws'}),
                                   ('ogm', {'alg': 'lf'}),
                                   ])

rnd = np.random.RandomState(2)

mean_times = defaultdict(list)
mean_energies = defaultdict(list)
#list_n_states = 2. ** np.arange(1, 5)
list_n_states = np.arange(2, 10)

for n_states in list_n_states:
    print("n_states = %d" % n_states)
    print("==============")
    for run in xrange(n_runs):
        print("run %d of %d" % (run, n_runs))
        # sample n_runs random potts models
        this_times = defaultdict(list)
        this_energies = defaultdict(list)
        x = rnd.normal(size=(size, size, n_states))
        pairwise = rnd.uniform(1, 3) * np.eye(n_states)
        edges = make_grid_edges(x)
        unaries = x.reshape(-1, n_states)
        # apply all inference methods
        for i, inference_method in enumerate(inference_methods):
            print("running %s" % str(inference_method))
            start = time()
            y = inference_dispatch(unaries, pairwise, edges,
                                   inference_method=inference_method)
            this_times[i].append(time() - start)
            this_energies[i].append(compute_energy(unaries, pairwise, edges,
                                                   y))
    # summarize runs
    for i in xrange(len(inference_methods)):
        mean_times[i].append(np.mean(this_times[i]))
        mean_energies[i].append(np.mean(this_energies[i]))

fig, ax = plt.subplots(2, 1)
for i, inference_method in enumerate(inference_methods):
    ax[0].plot(mean_times[i], label=str(inference_method))
    ax[1].plot(mean_energies[i], label=str(inference_method))

ax[0].legend()
ax[1].legend()
ax[0].set_title("Mean Run Time")
ax[0].set_xlabel("number of states")
ax[1].set_title("Mean Energy")
ax[1].set_xlabel("number of states")

plt.show()
