######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np
from scipy import sparse

from sklearn.cluster import KMeans

from crf import FixedGraphCRF, GridCRF, CRF
from inference_methods import _make_grid_edges
from pyqpbo import alpha_expansion_graph, alpha_expansion_grid

from IPython.core.debugger import Tracer
tracer = Tracer()


def kmeans_init(X, Y, edges, n_states_per_label=2):
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


class LatentCRF(CRF):
    def __repr__(self):
        return ("LatentCRF, n_labels : %d, n_states: %d, inference_method: %s"
                % (self.n_states, self.n_labels, self.inference_method))

    def loss_augment(self, x, h, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=-1)
        unary_params = w[:self.n_states].copy()
        # avoid division by zero:
        unary_params[unary_params == 0] = 1e-10
        for s in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_wide[h / self.n_states_per_label
                   != s / self.n_states_per_label, s] += 1. / unary_params[s]
        return x_wide


class LatentFixedGraphCRF(LatentCRF, FixedGraphCRF):
    """CRF with general graph that is THE SAME for all examples.
    graph is given by scipy sparse adjacency matrix.
    """
    def __init__(self, n_labels, n_states_per_label, graph):
        self.n_states_per_label = n_states_per_label
        n_states = n_labels * n_states_per_label
        super(LatentFixedGraphCRF, self).__init__(n_states, graph)
        # n_labels unary parameters, upper triangular for pairwise
        self.n_states = n_states

    def psi(self, x, h):
        # x is unaries
        # h is latent labeling
        ## unary features:
        x_wide = np.repeat(x, self.n_states_per_label, axis=1)
        return super(LatentFixedGraphCRF, self).psi(x_wide, h)

    def inference(self, x, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=1)
        # do usual inference
        h = super(LatentFixedGraphCRF, self).inference(x_wide, w)
        return h

    def loss_augmented_inference(self, x, h, w, relaxed=False):
        # augment unary potentials for latent states
        x_wide = self.loss_augment(x, h, w)
        # do usual inference
        h = super(LatentFixedGraphCRF, self).inference(x_wide, w, relaxed)
        return h

    def latent(self, x, y, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=1)
        # do usual inference
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = (pairwise_params + pairwise_params.T
                           - np.diag(np.diag(pairwise_params)))
        unaries = (- 10 * unary_params * x_wide).astype(np.int32)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = (np.arange(self.n_states) / self.n_states_per_label !=
                        y[:, np.newaxis])
        unaries[other_states] = +1000000
        pairwise = (-10 * pairwise_params).astype(np.int32)
        h = alpha_expansion_graph(self.edges, unaries, pairwise)
        if (h / self.n_states_per_label != y).any():
            if np.any(w):
                print("inconsistent h and y")
                #tracer()
                h = y * self.n_states_per_label
            else:
                h = y * self.n_states_per_label
        return h

    def loss(self, h, h_hat):
        return np.sum(h / self.n_states_per_label
                      != h_hat / self.n_states_per_label)


class LatentGridCRF(LatentCRF, GridCRF):
    """Latent variable CRF with 2d grid graph
    """
    def __init__(self, n_labels, n_states_per_label=2,
                 inference_method='qpbo'):
        self.n_states_per_label = n_states_per_label
        self.n_labels = n_labels

        n_states = n_labels * n_states_per_label
        super(LatentGridCRF, self).__init__(n_states,
                                            inference_method=inference_method)

    def init_latent(self, X, Y):
        edges = _make_grid_edges(X[0], neighborhood=self.neighborhood,
                                 return_lists=True)
        return kmeans_init(X, Y, edges,
                           n_states_per_label=self.n_states_per_label)

    def psi(self, x, h):
        # x is unaries
        # h is latent labeling
        ## unary features:
        x_wide = np.repeat(x, self.n_states_per_label, axis=-1)
        return super(LatentGridCRF, self).psi(x_wide, h)

    def inference(self, x, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=-1)
        # do usual inference
        h = super(LatentGridCRF, self).inference(x_wide, w)
        return h

    def loss_augmented_inference(self, x, h, w, relaxed=False):
        # augment unary potentials for latent states
        x_wide = self.loss_augment(x, h, w)
        # do usual inference
        h = super(LatentGridCRF, self).inference(x_wide, w, relaxed=relaxed)
        return h

    def latent(self, x, y, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=-1)
        # do usual inference
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = (pairwise_params + pairwise_params.T
                           - np.diag(np.diag(pairwise_params)))
        unaries = (- 10 * unary_params * x_wide).astype(np.int32)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = (np.arange(self.n_states) / self.n_states_per_label !=
                        y[:, :, np.newaxis])
        unaries[other_states] = +1000000
        pairwise = (-10 * pairwise_params).astype(np.int32)
        h = alpha_expansion_grid(unaries, pairwise)
        if (h / self.n_states_per_label != y).any():
            if np.any(w):
                print("inconsistent h and y")
                tracer()
                h = y * self.n_states_per_label
            else:
                h = y * self.n_states_per_label
        return h

    def loss(self, h, h_hat):
        return np.sum(h / self.n_states_per_label
                      != h_hat / self.n_states_per_label)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_hat_org = y_hat.reshape(y.shape[0], y.shape[1],
                                  self.n_labels,
                                  self.n_states_per_label).sum(axis=-1)
        y_org = y / self.n_states_per_label
        return super(LatentGridCRF, self).continuous_loss(y_org, y_hat_org)
