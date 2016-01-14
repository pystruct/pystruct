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
    H = [np.zeros(y.shape, dtype=np.int) for y in Y]
    label_indices = np.hstack([0, np.cumsum(n_states_per_label)])
    for label in np.unique(Y_stacked):
        try:
            km = KMeans(n_clusters=n_states_per_label[label])
        except TypeError:
            # for old versions :-/
            km = KMeans(k=n_states_per_label[label])
        indicator = Y_stacked == label
        f = all_feats_stacked[indicator]
        km.fit(f)
        for feats_sample, y, h in zip(all_feats, Y, H):
            indicator_sample = y.ravel() == label
            if np.any(indicator_sample):
                pred = km.predict(feats_sample[indicator_sample]).astype(np.int)
                h.ravel()[indicator_sample] = pred + label_indices[label]
    return H


class LatentGraphCRF(GraphCRF):
    """CRF with latent states for variables.

    This is also called "hidden dynamics CRF".
    For each output variable there is an additional variable which
    can encode additional states and interactions.

    Parameters
    ----------
    n_labels : int
        Number of states of output variables.

    n_featues : int or None (default=None).
        Number of input features per input variable.
        ``None`` means it is equal to ``n_labels``.

    n_states_per_label : int or list (default=2)
        Number of latent states associated with each observable state.
        Can be either an integer, which means the same number
        of hidden states will be used for all observable states, or a list
        of integers of length ``n_labels``.

    inference_method : string, default="ad3"
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.
    """
    def __init__(self, n_labels=None, n_features=None, n_states_per_label=2,
                 inference_method=None):
        self.n_labels = n_labels
        self.n_states_per_label = n_states_per_label
        GraphCRF.__init__(self, n_states=None, n_features=n_features,
                          inference_method=inference_method)

    def _set_size_joint_feature(self):
        if None in [self.n_features, self.n_labels]:
            return

        if isinstance(self.n_states_per_label, numbers.Integral):
            # same for all labels
            n_states_per_label = np.array([
                self.n_states_per_label for i in range(self.n_labels)])
        else:
            n_states_per_label = np.array(self.n_states_per_label)
            if len(n_states_per_label) != self.n_labels:
                raise ValueError("states_per_label must be integer"
                                 "or array-like of length n_labels. Got %s"
                                 % str(n_states_per_label))
        self.n_states_per_label = n_states_per_label
        self.n_states = np.sum(n_states_per_label)

        # compute mapping from latent states to labels
        ranges = np.cumsum(n_states_per_label)
        states_map = np.zeros(self.n_states, dtype=np.int)
        for l in range(1, self.n_labels):
            states_map[ranges[l - 1]: ranges[l]] = l
        self._states_map = states_map
        GraphCRF._set_size_joint_feature(self)

    def initialize(self, X, Y):
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_labels = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_labels is None:
            self.n_labels = n_labels
        elif self.n_labels != n_labels:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_labels, n_labels))
        self._set_size_joint_feature()
        self._set_class_weight()

    def label_from_latent(self, h):
        return self._states_map[h]

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[self._get_edges(x)] for x in X]
        features = np.array([self._get_features(x) for x in X])
        return kmeans_init(features, Y, edges, n_labels=self.n_labels,
                           n_states_per_label=self.n_states_per_label)

    def loss_augmented_inference(self, x, h, w, relaxed=False,
                                 return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
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
        unary_potentials = self._get_unary_potentials(x, w)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = self._states_map != y[:, np.newaxis]
        max_entry = np.maximum(np.max(unary_potentials), 1)
        unary_potentials[other_states] = -1e2 * max_entry
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (self.label_from_latent(h) != y).any():
            print("inconsistent h and y")
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
        for s in range(self.n_states):
            y_hat_org[:, self._states_map[s]] += y_hat[:, s]
        y_org = self.label_from_latent(y)
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)

    def base_loss(self, y, y_hat):
        if isinstance(y_hat, tuple):
            return GraphCRF.continuous_loss(self, y, y_hat)
        return GraphCRF.loss(self, y, y_hat)
