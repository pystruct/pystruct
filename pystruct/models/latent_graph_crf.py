######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# Implements a HRF / Latent Dynamic CRF
# For each output node there is one hidden node that is assigned a latent
# subclass.

import numbers

import numpy as np

from scipy import sparse
from sklearn.cluster import KMeans

from . import GraphCRF
from ..inference import inference_dispatch


def kmeans_init(X, Y, all_edges, n_labels, n_states_per_label,
                symmetric=True):
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
    all_feats_stacked = np.vstack(all_feats)
    Y_stacked = np.hstack(Y).ravel()
    # for each state, run k-means over whole dataset
    H = [np.zeros_like(y) for y in Y]
    label_indices = np.hstack([0, np.cumsum(n_states_per_label)])
    for label in np.unique(Y_stacked):
        km = KMeans(n_clusters=n_states_per_label[label])
        indicator = Y_stacked == label
        f = all_feats_stacked[indicator]
        km.fit(f)
        for feats_sample, y, h in zip(all_feats, Y, H):
            indicator_sample = y.ravel() == label
            h.ravel()[indicator_sample] = km.predict(
                feats_sample[indicator_sample]) + label_indices[label]
    return H


class LatentGraphCRF(GraphCRF):
    """Latent variable CRF with arbitrary graph.
    """
    def __init__(self, n_labels, n_features=None, n_states_per_label=2,
                 inference_method='qpbo'):
        self.n_labels = n_labels
        if n_features is None:
            n_features = n_labels

        if isinstance(n_states_per_label, numbers.Integral):
            # same for all labels
            n_states_per_label = np.array([n_states_per_label
                                           for i in xrange(n_labels)])
        else:
            n_states_per_label = np.array(n_states_per_label)
            if len(n_states_per_label) != n_labels:
                raise ValueError("states_per_label must be integer"
                                 "or array-like of length n_labels. Got %s"
                                 % str(n_states_per_label))
        self.n_states_per_label = n_states_per_label
        n_states = np.sum(n_states_per_label)

        # compute mapping from latent states to labels
        ranges = np.cumsum(n_states_per_label)
        states_map = np.zeros(n_states, dtype=np.int)
        for l in xrange(1, n_labels):
            states_map[ranges[l - 1]: ranges[l]] = l
        self._states_map = states_map

        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method)

    def label_from_latent(self, h):
        return self._states_map[h]

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[self.get_edges(x)] for x in X]
        features = np.array([self.get_features(x) for x in X])
        return kmeans_init(features, Y, edges, n_labels=self.n_labels,
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
            unary_potentials[self.label_from_latent(h)
                             != self.label_from_latent(l), l] += 1.

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        unary_potentials = self.get_unary_potentials(x, w)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = self._states_map != y[:, np.newaxis]
        max_entry = np.maximum(np.max(unary_potentials), 1)
        unary_potentials[other_states] = -1e2 * max_entry
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (self.label_from_latent(h) != y).any():
            print("inconsistent h and y")
            from IPython.core.debugger import Tracer
            Tracer()()
            h = np.hstack([0, np.cumsum(self.n_states_per_label)])[y]
        return h

    def loss(self, h, h_hat):
        if isinstance(h_hat, tuple):
            return self.continuous_loss(h, h_hat[0])
        return GraphCRF.loss(self, self.label_from_latent(h),
                             self.label_from_latent(h_hat))

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        y_hat_org = np.zeros((y_hat.shape[0], self.n_labels))
        for s in xrange(self.n_states):
            y_hat_org[:, self._states_map[s]] += y_hat[:, s]
        y_org = self.label_from_latent(y)
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)
