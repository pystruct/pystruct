"""
=================================
Latent Variable Hierarchical CRF
=================================
Solving a 2d grid toy problem by introducing an additional layer of latent
variables.
"""
import numpy as np
import itertools

#from numpy.testing import assert_array_equal, assert_array_almost_equal
#from nose.tools import assert_equal
from pystruct.models import GraphCRF, LatentNodeCRF
from pystruct.learners import StructuredSVM
from pystruct.learners import LatentSubgradientSSVM
#from pystruct.learners import LatentSSVM
import pystruct.toy_datasets as toy
from pystruct.utils import make_grid_edges
import matplotlib.pyplot as plt


def plot_boxes(boxes, size=4):
    if boxes[0].size == size * size:
        fig, ax = plt.subplots(1, len(boxes), figsize=(20, 10))
        for a, x in zip(ax, boxes):
            a.matshow(x[:size * size].reshape(size, size))
    else:
        # have hidden states
        fig, ax = plt.subplots(2, len(boxes), figsize=(20, 10))
        for a, x in zip(ax[0], boxes):
            a.matshow(x[size * size:].reshape(size / 2, size / 2))
        for a, x in zip(ax[1], boxes):
            a.matshow(x[:size * size].reshape(size, size))


# learn the "easy" 2x2 boxes dataset.
# a 2x2 box is placed randomly in a 4x4 grid
# we add a latent variable for each 2x2 patch
# that should make the model fairly simple

X, Y = toy.make_simple_2x2(seed=1)

# flatten X and Y
X_flat = [x.reshape(-1, 1).astype(np.float) for x in X]
Y_flat = [y.ravel() for y in Y]


# first, use standard graph CRF. Can't do much, high loss.
crf = GraphCRF(n_states=2, n_features=1, inference_method='lp')
svm = StructuredSVM(model=crf, max_iter=200, C=1, verbose=0,
                    check_constraints=True, break_on_bad=False, n_jobs=1)

# make dataset from X and graph without edges
#G_ = [np.zeros((0, 2), dtype=np.int) for x in X]
G = [make_grid_edges(x) for x in X]

asdf = zip(X_flat, G)
svm.fit(asdf, Y_flat)
plot_boxes(svm.predict(asdf))
print("Training score multiclass svm CRF: %f" % svm.score(asdf, Y_flat))

# using one latent variable for each 2x2 rectangle
latent_crf = LatentNodeCRF(n_labels=2, n_features=1, inference_method='lp',
                           n_hidden_states=2)
#latent_svm = LatentSSVM(model=latent_crf, max_iter=200, C=10, verbose=10,
                        #check_constraints=True, break_on_bad=True, n_jobs=1,
                        #latent_iter=10, base_svm='subgradient', tol=-1,
                        #inactive_window=0, learning_rate=0.01, momentum=0)
latent_svm = LatentSubgradientSSVM(model=latent_crf, max_iter=200, C=100,
                                   verbose=1, n_jobs=1, show_loss_every=10,
                                   learning_rate=0.01, momentum=0)

# make edges for hidden states:
edges = []
node_indices = np.arange(4 * 4).reshape(4, 4)
for i, (x, y) in enumerate(itertools.product([0, 2], repeat=2)):
    for j in xrange(x, x + 2):
        for k in xrange(y, y + 2):
            edges.append([i + 4 * 4, node_indices[j, k]])

G = [np.vstack([make_grid_edges(x), edges]) for x in X]
#G = [make_grid_edges(x) for x in X]

#H_init = [np.hstack([y.ravel(), 2 + y[1: -1, 1: -1].ravel()]) for y in Y]
H_init = [np.hstack([y.ravel(), np.random.randint(2, 4, size=2 * 2)]) for y in
          Y]
plot_boxes(H_init)


X_ = zip(X_flat, G, [2 * 2 for x in X_flat])

latent_svm.fit(X_, Y_flat, H_init)

print("Training score with latent nodes: %f " % latent_svm.score(X_, Y_flat))
H = latent_svm.predict_latent(X_)
plot_boxes(H)
plt.show()
