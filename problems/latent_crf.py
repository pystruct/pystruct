######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np
from scipy import sparse

from sklearn.cluster import KMeans

from . import GridCRF, DirectionalGridCRF
from ..inference import inference_dispatch
from ..utils import make_grid_edges

from IPython.core.debugger import Tracer
tracer = Tracer()


def kmeans_init(X, Y, edges, n_states_per_label=2, symmetric=True):
    n_labels = X[0].shape[-1]
    shape = Y[0].shape
    gx, gy = np.ogrid[:shape[0], :shape[1]]
    all_feats = []
    # iterate over samples
    for x, y in zip(X, Y):
        # first, get neighbor counts from nodes
        labels = np.zeros((shape[0], shape[1], n_labels),
                          dtype=np.int)
        labels[gx, gy, y] = 1
        size = np.prod(y.shape)
        graphs = [sparse.coo_matrix((np.ones(e.shape[0]), e.T), (size, size))
                  for e in edges]
        if symmetric:
            directions = [g + g.T for g in graphs]
        else:
            directions = [T for g in graphs for T in [g, g.T]]
        features = [s * labels.reshape(size, -1) for s in directions]
        features = np.hstack(features)
        # normalize (for borders)
        features /= features.sum(axis=1)[:, np.newaxis]

        # add unaries
        #features = np.dstack([x, neighbors])
        all_feats.append(features)
    all_feats = np.vstack(all_feats)
    # states (=clusters) will be saved in H
    H = np.zeros_like(Y, dtype=np.int)
    km = KMeans(n_clusters=n_states_per_label)
    # for each state, run k-means over whole dataset
    for label in np.arange(n_labels):
        indicator = Y.ravel() == label
        f = all_feats[indicator]
        states = km.fit_predict(f)
        H.ravel()[indicator] = states + label * n_states_per_label
    return H


class LatentGridCRF(GridCRF):
    """Latent variable CRF with 2d grid graph.
    """
    def __init__(self, n_labels, n_features=None, n_states_per_label=2,
                 inference_method='qpbo'):
        self.n_states_per_label = n_states_per_label
        self.n_labels = n_labels
        if n_features is None:
            n_features = n_labels

        n_states = n_labels * n_states_per_label
        GridCRF.__init__(self, n_states, n_features,
                         inference_method=inference_method)

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = make_grid_edges(X[0], neighborhood=self.neighborhood,
                                return_lists=False)
        return kmeans_init(X, Y, [edges],
                           n_states_per_label=self.n_states_per_label)

    def loss_augmented_inference(self, x, h, w, relaxed=False,
                                 return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self.get_unary_potentials(x, w)
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        # do loss-augmentation
        for l in np.arange(self.n_states):
            # for each class, decrement features
            # for loss-agumention
            unary_potentials[h // self.n_states_per_label
                             != l // self.n_states_per_label, l] += 1.

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        unary_potentials = self.get_unary_potentials(x, w)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = (np.arange(self.n_states) / self.n_states_per_label !=
                        y[:, :, np.newaxis])
        unary_potentials[other_states] = -1000
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (h // self.n_states_per_label != y).any():
            if np.any(w):
                print("inconsistent h and y")
                tracer()
                h = y * self.n_states_per_label
            else:
                h = y * self.n_states_per_label
        return h

    def loss(self, h, h_hat):
        return np.sum(h // self.n_states_per_label
                      != h_hat // self.n_states_per_label)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_hat_org = y_hat.reshape(y.shape[0], y.shape[1],
                                  self.n_labels,
                                  self.n_states_per_label).sum(axis=-1)
        y_org = y / self.n_states_per_label
        return super(LatentGridCRF, self).continuous_loss(y_org, y_hat_org)


class LatentDirectionalGridCRF(DirectionalGridCRF, LatentGridCRF):
    """Latent variable CRF with directional 2d grid graph.

    Multiple inheritance weirdness....
    All features / potentials are the same as in the DirectionalGridCRF,
    so use these.

    Things that have to do with y or h need to call the
    LatentGridCRF function - that simply works because the feature are right.
    """
    def __init__(self, n_labels, n_features=None, n_states_per_label=2,
                 inference_method='qpbo', neighborhood=4):
        LatentGridCRF.__init__(self, n_labels, n_features, n_states_per_label,
                               inference_method=inference_method)
        DirectionalGridCRF.__init__(self, self.n_states, self.n_features,
                                    inference_method=inference_method,
                                    neighborhood=neighborhood)

    def init_latent(self, X, Y):
        edges = make_grid_edges(X[0], neighborhood=self.neighborhood,
                                return_lists=True)
        return kmeans_init(X, Y, edges,
                           n_states_per_label=self.n_states_per_label,
                           symmetric=False)

    def loss_augmented_inference(self, x, h, w, relaxed=False):
        h = LatentGridCRF.loss_augmented_inference(self, x, h, w,
                                                   relaxed=relaxed)
        return h
