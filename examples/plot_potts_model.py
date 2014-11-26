"""
=================================================
Comparing inference times on a simple Potts model
=================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from time import time

from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

size = 20
n_states = 5

rnd = np.random.RandomState(2)
x = rnd.normal(size=(size, size, n_states))
pairwise = np.eye(n_states)
edges = make_grid_edges(x)
unaries = x.reshape(-1, n_states)

fig, ax = plt.subplots(1, 6)
for a, inference_method in zip(ax, ['ad3bb', 'ad3', 'qpbo', 'mp', 'lp',
                                    'ogm']):
    start = time()
    y = inference_dispatch(unaries, pairwise, edges,
                           inference_method=inference_method)
    took = time() - start
    a.matshow(y.reshape(size, size))
    energy = compute_energy(unaries, pairwise, edges, y)
    a.set_title("time: %.2f energy %.2f" % (took, energy))
    a.set_xticks(())
    a.set_yticks(())
plt.show()
