######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
#
# Implements a CRF with arbitrary unobserved nodes.
# All unobserved nodes share the same state-space, which is separate from the
# observed states.
# Unobserved nodes don't have unary potentials currently (should they?)

import numpy as np

from scipy import sparse
from sklearn.cluster import KMeans

from . import GraphCRF
from ..inference import inference_dispatch
#from .latent_graph_crf import kmeans_init


def kmeans_init(X, Y, n_labels, n_hidden_states):
    all_feats = []
    # iterate over samples
    for x, y in zip(X, Y):
        # first, get neighbor counts from nodes
        features, edges, n_hidden = x
        n_visible = features.shape[0]
        if np.max(edges) != n_hidden + n_visible - 1:
            raise ValueError("Edges don't add up")

        labels_one_hot = np.zeros((n_visible, n_labels), dtype=np.int)
        y = y.ravel()
        gx = np.ogrid[:n_visible]
        labels_one_hot[gx, y] = 1

        graph = sparse.coo_matrix((np.ones(edges.shape[0]), edges.T),
                                  (n_visible + n_hidden, n_visible + n_hidden))
        graph = (graph + graph.T)[-n_hidden:, :n_visible]

        neighbors = graph * labels_one_hot.reshape(n_visible, -1)
        # normalize (for borders)
        neighbors /= np.maximum(neighbors.sum(axis=1)[:, np.newaxis], 1)

        all_feats.append(neighbors)
    all_feats_stacked = np.vstack(all_feats)
    km = KMeans(n_clusters=n_hidden_states)
    km.fit(all_feats_stacked)
    H = []
    for y, feats in zip(Y, all_feats):
        H.append(np.hstack([y, km.predict(feats) + n_labels]))

    return H


class LatentNodeCRF(GraphCRF):
    """Latent variable CRF with 2d grid graph.
    Input x is tuple (features, edges, n_hidden)
    First features.shape[0] nodes are observed, then n_hidden unobserved nodes.
    """
    def __init__(self, n_labels, n_features=None, n_hidden_states=2,
                 inference_method='lp'):
        self.n_labels = n_labels
        if n_features is None:
            n_features = n_labels

        self.n_hidden_states = n_hidden_states
        n_states = n_hidden_states + n_labels

        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method)

        self.size_psi = (n_labels * self.n_features
                         + n_states * (n_states + 1) / 2)

    def get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        pairwise_flat = np.asarray(w[self.n_labels * self.n_features:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        # set lower triangle of matrix, then make symmetric
        # we could try to redo this using ``scipy.spatial.distance`` somehow
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        return (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

    def get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)
        n_visible, n_hidden = features.shape[0], x[2]
        # assemble unary potentials for all nodes from observed evidence
        unaries = np.zeros((n_visible + n_hidden, self.n_states))
        unary_params = w[:self.n_labels * self.n_features].reshape(
            self.n_labels, self.n_features)
        unaries_observed = np.dot(features, unary_params.T)
        # paste observed into large matrix
        unaries[:n_visible, :self.n_labels] = unaries_observed
        # forbid latent states for observable nodes
        max_entry = np.maximum(np.max(unaries_observed), 1)
        unaries[:n_visible, self.n_labels:] = -1e2 * max_entry
        # forbid observed states for latent nodes
        unaries[n_visible:, :self.n_labels] = -1e2 * max_entry
        return unaries

    #def init_latent(self, X, Y):
        ## treat all edges the same
        #edges = [[self.get_edges(x)] for x in X]
        #features = np.array([self.get_features(x) for x in X])
        #return kmeans_init(features, Y, edges, n_labels=self.n_labels,
                           #n_states_per_label=self.n_states_per_label)

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
            unary_potentials[np.where(self.label_from_latent(h)
                             != l)[0], l] += 1.

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        unary_potentials = self.get_unary_potentials(x, w)
        # clamp observed nodes by modifying unary potentials
        max_entry = np.maximum(np.max(unary_potentials), 1)
        unary_potentials[np.arange(len(y)), y] = 1e2 * max_entry
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (h[:len(y)] != y).any():
            print("inconsistent h and y")
            from IPython.core.debugger import Tracer
            Tracer()()
            h[:len(y)] = y
        return h

    def label_from_latent(self, h):
        return h[h < self.n_labels]

    def loss(self, h, h_hat):
        if isinstance(h_hat, tuple):
            return self.continuous_loss(h, h_hat[0])
        return GraphCRF.loss(self, self.label_from_latent(h),
                             self.label_from_latent(h_hat))

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        y_org = self.label_from_latent(y)
        y_hat_org = y_hat[:y_org.size, :self.n_labels]
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)

    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : tuple
            Unary evidence.

        y : ndarray or tuple
            Either y is an integral ndarray, giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            n_nodes = y.size
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ##accumulated pairwise
            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])
        n_visible = features.shape[0]
        unaries_acc = np.dot(unary_marginals[:n_visible, :self.n_labels].T,
                             features)
        pw = pw + pw.T - np.diag(np.diag(pw))  # make symmetric

        psi_vector = np.hstack([unaries_acc.ravel(),
                                pw[np.tri(self.n_states, dtype=np.bool)]])
        return psi_vector

    def init_latent(self, X, Y):
        # treat all edges the same
        return kmeans_init(X, Y, n_labels=self.n_labels,
                           n_hidden_states=self.n_hidden_states)
