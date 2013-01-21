######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
#

import numpy as np

from scipy import sparse
from sklearn.cluster import KMeans

from . import GraphCRF
from ..inference import inference_dispatch

from IPython.core.debugger import Tracer
tracer = Tracer()


def kmeans_init(X, Y, all_edges, n_labels=2, n_states_per_label=2,
                symmetric=True):
    # flatten grids
    X = X.reshape(X.shape[0], -1, X.shape[-1])
    all_feats = []
    # iterate over samples
    for x, y, edges in zip(X, Y, all_edges):
        # first, get neighbor counts from nodes
        n_nodes = x.shape[0]
        labels_one_hot = np.zeros((n_nodes, n_labels), dtype=np.int)
        y = y.ravel()
        gx = np.ogrid[:n_nodes]
        labels_one_hot[gx, y] = 1

        size = np.prod(y.shape)
        graphs = [sparse.coo_matrix((np.ones(e.shape[0]), e.T), (size, size))
                  for e in edges]
        if symmetric:
            directions = [g + g.T for g in graphs]
        else:
            directions = [T for g in graphs for T in [g, g.T]]
        neighbors = [s * labels_one_hot.reshape(size, -1) for s in directions]
        neighbors = np.hstack(neighbors)
        # normalize (for borders)
        neighbors /= np.maximum(neighbors.sum(axis=1)[:, np.newaxis], 1)

        # add unaries
        features = np.hstack([x, neighbors])
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


class LatentGraphCRF(GraphCRF):
    """Latent variable CRF with 2d grid graph.
    """
    def __init__(self, n_labels, n_features=None, n_states_per_label=2,
                 inference_method='qpbo'):
        self.n_states_per_label = n_states_per_label
        self.n_labels = n_labels
        if n_features is None:
            n_features = n_labels

        n_states = n_labels * n_states_per_label
        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method)

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[self.get_edges(x)] for x in X]
        features = np.array([self.get_features(x) for x in X])
        return kmeans_init(features, Y, edges,
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
                        y[:, np.newaxis])
        unary_potentials[other_states] = -1000
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (h // self.n_states_per_label != y).any():
            print("inconsistent h and y")
            h = y * self.n_states_per_label
            tracer()
        return h

    def loss(self, h, h_hat):
        return np.sum(h // self.n_states_per_label
                      != h_hat // self.n_states_per_label)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_hat_org = y_hat.reshape(-1,
                                  self.n_labels,
                                  self.n_states_per_label).sum(axis=-1)
        y_org = y / self.n_states_per_label
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)
