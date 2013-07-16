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
inference_methods = get_installed(['ad3', 'qpbo', 'ogm', 'unary', 'ad3bb'])

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
        for inference_method in inference_methods:
            print("running %s" % inference_method)
            start = time()
            y = inference_dispatch(unaries, pairwise, edges,
                                   inference_method=inference_method)
            this_times[inference_method].append(time() - start)
            this_energies[inference_method].append(compute_energy(unaries,
                                                                  pairwise,
                                                                  edges, y))
    # summarize runs
    for inference_method in inference_methods:
        mean_times[inference_method].append(
            np.mean(this_times[inference_method]))
        mean_energies[inference_method].append(
            np.mean(this_energies[inference_method]))

fig, ax = plt.subplots(2, 1)
for inference_method in inference_methods:
    ax[0].plot(mean_times[inference_method], label=inference_method)
    ax[1].plot(mean_energies[inference_method], label=inference_method)

ax[0].legend()
ax[1].legend()
ax[0].set_title("Mean Run Time")
ax[0].set_xlabel("number of states")
ax[1].set_title("Mean Energy")
ax[1].set_xlabel("number of states")

plt.show()
