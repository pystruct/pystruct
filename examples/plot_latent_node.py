"""
=================================
Latent Variable Hierarchical CRF
=================================

Solving a 2d grid toy problem by introducing an additional layer of latent
variables.
"""
import numpy as np
import itertools

from pystruct.models import GraphCRF, LatentNodeCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM, LatentSSVM
from pystruct.datasets import make_simple_2x2
from pystruct.utils import make_grid_edges, plot_grid
import matplotlib.pyplot as plt


def plot_boxes(boxes, size=4, title=""):
    cmap = plt.cm.gray
    if boxes[0].size == size * size:
        fig, ax = plt.subplots(1, len(boxes), figsize=(8, 0.7))
        for a, x in zip(ax, boxes):
            plot_grid(x[:size * size].reshape(size, size), cmap=cmap, axes=a,
                      border_color="green")
    else:
        # have hidden states
        fig, ax = plt.subplots(2, len(boxes), figsize=(8, 1))
        for a, x in zip(ax[0], boxes):
            plot_grid(x[size * size:].reshape(size / 2, size / 2), cmap=cmap,
                      axes=a, border_color="green")
        for a, x in zip(ax[1], boxes):
            plot_grid(x[:size * size].reshape(size, size), cmap=cmap, axes=a,
                      border_color="green")
    fig.subplots_adjust(.01, .03, .98, .75, .2, .05)
    fig.suptitle(title)


# learn the "easy" 2x2 boxes dataset.
# a 2x2 box is placed randomly in a 4x4 grid
# we add a latent variable for each 2x2 patch
# that should make the model fairly simple

X, Y = make_simple_2x2(seed=1)

# flatten X and Y
X_flat = [x.reshape(-1, 1).astype(np.float) for x in X]
Y_flat = [y.ravel() for y in Y]


# first, use standard graph CRF. Can't do much, high loss.
crf = GraphCRF()
svm = NSlackSSVM(model=crf, max_iter=200, C=1, n_jobs=1)

G = [make_grid_edges(x) for x in X]

X_grid_edges = list(zip(X_flat, G))
svm.fit(X_grid_edges, Y_flat)
plot_boxes(svm.predict(X_grid_edges), title="Non-latent SSVM predictions")
print("Training score binary grid CRF: %f" % svm.score(X_grid_edges, Y_flat))

# using one latent variable for each 2x2 rectangle
latent_crf = LatentNodeCRF(n_labels=2, n_features=1, n_hidden_states=2,
                           inference_method='lp')

ssvm = OneSlackSSVM(model=latent_crf, max_iter=200, C=100,
                    n_jobs=-1, show_loss_every=10, inference_cache=50)
latent_svm = LatentSSVM(ssvm)

# make edges for hidden states:
edges = []
node_indices = np.arange(4 * 4).reshape(4, 4)
for i, (x, y) in enumerate(itertools.product([0, 2], repeat=2)):
    for j in range(x, x + 2):
        for k in range(y, y + 2):
            edges.append([i + 4 * 4, node_indices[j, k]])

G = [np.vstack([make_grid_edges(x), edges]) for x in X]

# Random initialization
H_init = [np.hstack([y.ravel(), np.random.randint(2, 4, size=2 * 2)])
          for y in Y]
plot_boxes(H_init, title="Top: Random initial hidden states. Bottom: Ground"
           "truth labeling.")

X_ = list(zip(X_flat, G, [2 * 2 for x in X_flat]))

latent_svm.fit(X_, Y_flat, H_init)

print("Training score with latent nodes: %f " % latent_svm.score(X_, Y_flat))
H = latent_svm.predict_latent(X_)
plot_boxes(H, title="Top: Hidden states after training. Bottom: Prediction.")
plt.show()
