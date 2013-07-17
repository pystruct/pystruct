from collections import defaultdict
from time import time

import numpy as np
import matplotlib.pyplot as plt

from pystruct.inference import (inference_dispatch, compute_energy,
                                get_installed)
from pystruct.utils import make_grid_edges


def inf_repr(inference_method):
    if not isinstance(inference_method, tuple):
        return inference_method
    return inference_method[0] + " " + " ".join([str(s) for s in
                                                 inference_method[1].values()])


def generate_comparison_plot(inference_methods, n_runs, size, list_n_states):
    n_methods = len(inference_methods)

    mean_times = defaultdict(list)
    mean_energies = defaultdict(list)

    for n_states in list_n_states:
        print("\nn_states = %d" % n_states)
        print("==============")
        for run in xrange(n_runs):
            print("run %d of %d" % (run, n_runs))
            # sample n_runs random potts models
            this_times = defaultdict(list)
            this_energies = defaultdict(list)
            x = rnd.normal(size=(size, size, n_states))
            pairwise = rnd.normal(size=(n_states, n_states))
            edges = make_grid_edges(x)
            unaries = x.reshape(-1, n_states)
            # apply all inference methods
            for i, inference_method in enumerate(inference_methods):
                #print("running %s" % str(inference_method))
                start = time()
                y = inference_dispatch(unaries, pairwise, edges,
                                       inference_method=inference_method)
                this_times[i].append(time() - start)
                this_energies[i].append(compute_energy(unaries, pairwise,
                                                       edges, y))
        # summarize runs
        for i in xrange(n_methods):
            mean_times[i].append(np.mean(this_times[i]))
            mean_energies[i].append(np.mean(this_energies[i]))

    fig, ax = plt.subplots(2, 1)
    for i, inference_method in enumerate(inference_methods):
        rep = inf_repr(inference_method)
        color = plt.cm.jet(float(i) / n_methods)
        ax[0].plot(list_n_states, mean_times[i], label=rep, c=color)
        ax[1].plot(list_n_states, mean_energies[i], label=rep, c=color)
        ax[0].text(list_n_states[-1], mean_times[i][-1], rep)
        ax[1].text(list_n_states[-1], mean_energies[i][-1], rep)

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[0].set_title("Mean Run Time (lower is better)")
    ax[0].set_xlabel("number of states")
    ax[1].set_title("Mean Energy (higher is better)")
    ax[1].set_xlabel("number of states")

diverse_methods = get_installed(['ad3', 'qpbo', 'unary', 'lp',
                                 ('ad3', {'branch_and_bound': True}),
                                 ('ogm', {'alg': 'bp'}),
                                 ('ogm', {'alg': 'dd'}),
                                 ('ogm', {'alg': 'trws'}),
                                 ('ogm', {'alg': 'trw'}),
                                 ('ogm', {'alg': 'gibbs'}),
                                 ('ogm', {'alg': 'fm'}),
                                 ('ogm', {'alg': 'icm'}),
                                 ('ogm', {'alg': 'mqpbo'}),
                                 ('ogm', {'alg': 'lf'}), ])

fast_methods = get_installed(['qpbo', 'unary',
                              ('ogm', {'alg': 'bp'}),
                              ('ogm', {'alg': 'fm'}),
                              ('ogm', {'alg': 'trw'}),
                              ('ogm', {'alg': 'dd'}),
                              ('ogm', {'alg': 'icm'}),
                              ('ogm', {'alg': 'mqpbo'}),
                              ('ogm', {'alg': 'lf'}), ])

very_fast_methods = get_installed(['qpbo', 'unary',
                                   ('ogm', {'alg': 'fm'}),
                                   ('ogm', {'alg': 'icm'}),
                                   ('ogm', {'alg': 'mqpbo'}),
                                   ('ogm', {'alg': 'lf'}), ])

rnd = np.random.RandomState(2)
generate_comparison_plot(inference_methods=diverse_methods, n_runs=5, size=10,
                         list_n_states=np.arange(2, 6))

generate_comparison_plot(inference_methods=fast_methods, n_runs=5, size=20,
                         list_n_states=np.arange(2, 20, 4))

generate_comparison_plot(inference_methods=very_fast_methods, n_runs=5,
                         size=10, list_n_states=np.arange(2, 50, 5))
plt.show()
