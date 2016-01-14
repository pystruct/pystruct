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
from ..utils import expand_sym, compress_sym


def kmeans_init(X, Y, n_labels, n_hidden_states, latent_node_features=False):
    all_feats = []
    # iterate over samples
    for x, y in zip(X, Y):
        # first, get neighbor counts from nodes
        if len(x) == 3:
            features, edges, n_hidden = x
        elif len(x) == 4:
            # edge features are discarded
            features, edges, _, n_hidden = x
        else:
            raise ValueError("Something is fishy!")
        n_visible = features.shape[0]
        if latent_node_features:
            n_visible -= n_hidden
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
    try:
        km = KMeans(n_clusters=n_hidden_states)
    except TypeError:
        # for old versions :-/
        km = KMeans(k=n_hidden_states)

    km.fit(all_feats_stacked)
    H = []
    for y, feats in zip(Y, all_feats):
        H.append(np.hstack([y, km.predict(feats) + n_labels]))

    return H


class LatentNodeCRF(GraphCRF):
    """CRF with latent variables.

    Input x is tuple (features, edges, n_hidden)
    First features.shape[0] nodes are observed, then n_hidden unobserved nodes.

    Currently unobserved nodes don't have features.

    Parameters
    ----------
    n_labels : int, default=2
        Number of states for observed variables.

    n_hidden_states : int, default=2
        Number of states for hidden variables.

    n_features : int, default=None
        Number of features per node. None means n_states.

    inference_method : string, default=None
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    latent_node_features : bool, default=False
        Whether latent nodes have features. We assume that if True,
        the number of features is the same as for visible nodes.
    """
    def __init__(self, n_labels=None, n_features=None, n_hidden_states=2,
                 inference_method=None, class_weight=None,
                 latent_node_features=False):
        self.n_labels = n_labels
        self.n_hidden_states = n_hidden_states
        if n_labels is not None:
            n_states = n_hidden_states + n_labels
        else:
            n_states = None
        self.latent_node_features = latent_node_features

        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method,
                          class_weight=class_weight)

    def _set_size_joint_feature(self):
        if None in [self.n_states, self.n_features]:
            return

        if self.latent_node_features:
            n_input_states = self.n_states
        else:
            n_input_states = self.n_labels
        self.n_input_states = n_input_states
        self.size_joint_feature = (n_input_states * self.n_features +
                                   self.n_states * (self.n_states + 1) / 2)

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
            raise ValueError("Expected %d labels, got %d"
                             % (self.n_labels, n_labels))
        self.n_states = self.n_hidden_states + n_labels
        self._set_size_joint_feature()
        self._set_class_weight()

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        return expand_sym(w[self.n_input_states * self.n_features:])

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features = self._get_features(x)
        unary_params = w[:self.n_input_states * self.n_features].reshape(
            self.n_input_states, self.n_features)

        if self.latent_node_features:
            unaries = np.dot(features, unary_params.T)
            n_hidden = x[2]
            n_visible = features.shape[0] - n_hidden
        else:
            # we only have features for visible nodes
            n_visible, n_hidden = features.shape[0], x[2]
            # assemble unary potentials for all nodes from observed evidence
            unaries = np.zeros((n_visible + n_hidden, self.n_states))
            unaries_observed = np.dot(features, unary_params.T)
            # paste observed into large matrix
            unaries[:n_visible, :self.n_labels] = unaries_observed
        # forbid latent states for observable nodes
        max_entry = np.maximum(np.max(unaries), 1)
        unaries[:n_visible, self.n_labels:] = -1e2 * max_entry
        # forbid observed states for latent nodes
        unaries[n_visible:, :self.n_labels] = -1e2 * max_entry
        return unaries

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
            inds = np.where(self.label_from_latent(h) != l)[0]
            unary_potentials[inds, l] += self.class_weight[
                self.label_from_latent(h)][inds]

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        unary_potentials = self._get_unary_potentials(x, w)
        # clamp observed nodes by modifying unary potentials
        max_entry = np.maximum(np.max(unary_potentials), 1)
        unary_potentials[np.arange(len(y)), y] = 1e2 * max_entry
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (h[:len(y)] != y).any():
            raise ValueError("Inconsistent h and y in latent node CRF!")
            h[:len(y)] = y
        return h

    def label_from_latent(self, h):
        return h[h < self.n_labels]

    def loss(self, h, h_hat):
        if isinstance(h_hat, tuple):
            return self.continuous_loss(h, h_hat[0])
        return GraphCRF.loss(self, self.label_from_latent(h),
                             self.label_from_latent(h_hat))

    def base_loss(self, y, y_hat):
        if isinstance(y_hat, tuple):
            return GraphCRF.continuous_loss(self, y, y_hat)
        return GraphCRF.loss(self, y, y_hat)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        y_org = self.label_from_latent(y)
        y_hat_org = y_hat[:y_org.size, :self.n_labels]
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the
        configuration (x, y) and a weight vector w is given by
        np.dot(w, joint_feature(x, y)).

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
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)

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
        unaries_acc = np.dot(unary_marginals[:n_visible,
                                             :self.n_input_states].T, features)

        joint_feature_vector = np.hstack([unaries_acc.ravel(),
                                          compress_sym(pw)])
        return joint_feature_vector

    def init_latent(self, X, Y):
        # treat all edges the same
        return kmeans_init(X, Y, n_labels=self.n_labels,
                           n_hidden_states=self.n_hidden_states,
                           latent_node_features=self.latent_node_features)

    def max_loss(self, h):
        # maximum possible los on y for macro averages
        y = self.label_from_latent(h)
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y])


class EdgeFeatureLatentNodeCRF(LatentNodeCRF):
    """CRF with latent variables and edge features.

    Yeah that's totally not a mess.

    Input x is tuple (features, edges, edge_features, n_hidden)
    First features.shape[0] nodes are observed, then n_hidden unobserved nodes.


    Parameters
    ----------
    n_labels : int, default=2
        Number of states for observed variables.

    n_hidden_states : int, default=2
        Number of states for hidden variables.

    n_edge_features : int, default=1
        Number of features per edge.

    n_features : int, default=None
        Number of features per node. None means n_states.

    inference_method : string, default="None"
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    latent_node_features : bool, default=False
        Whether latent nodes have features. We assume that if True,
        the number of features is the same as for visible nodes.

    symmetric_edge_features : None or list
        Indices of edge features that are forced to be symmetric.
        Often the direction of the edge has no immediate meaning.

    antisymmetric_edge_features : None or list
        Indices of edge features that are forced to be anti-symmetric.

    """
    def __init__(self, n_labels=2, n_features=None, n_edge_features=1,
                 n_hidden_states=2, inference_method=None, class_weight=None,
                 latent_node_features=False, symmetric_edge_features=None,
                 antisymmetric_edge_features=None):

        self.n_labels = n_labels
        if n_features is None:
            n_features = n_labels

        self.n_hidden_states = n_hidden_states
        n_states = n_hidden_states + n_labels
        self.n_edge_features = n_edge_features

        if latent_node_features:
            n_input_states = n_states
        else:
            n_input_states = n_labels

        self.n_input_states = n_input_states
        self.size_joint_feature = (
            n_input_states * n_features + n_edge_features * n_states ** 2)
        self.latent_node_features = latent_node_features

        if symmetric_edge_features is None:
            symmetric_edge_features = []
        if antisymmetric_edge_features is None:
            antisymmetric_edge_features = []

        if not set(symmetric_edge_features).isdisjoint(
                antisymmetric_edge_features):
            raise ValueError("symmetric_edge_features and "
                             " antisymmetric_edge_features share an entry."
                             " That doesn't make any sense.")

        self.symmetric_edge_features = symmetric_edge_features
        self.antisymmetric_edge_features = antisymmetric_edge_features

        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method,
                          class_weight=class_weight)

    def _set_size_joint_feature(self):
        if None in [self.n_states, self.n_features]:
            return

        if self.latent_node_features:
            n_input_states = self.n_states
        else:
            n_input_states = self.n_labels
        self.n_input_states = n_input_states
        self.size_joint_feature = (
            n_input_states * self.n_features + self.n_edge_features *
            self.n_states ** 2)

    def _check_size_x(self, x):
        GraphCRF._check_size_x(self, x)

        _, edges, edge_features, n_hidden = x
        if edges.shape[0] != edge_features.shape[0]:
            raise ValueError("Got %d edges but %d edge features."
                             % (edges.shape[0], edge_features.shape[0]))
        if edge_features.shape[1] != self.n_edge_features:
            raise ValueError("Got edge features of size %d, but expected %d."
                             % (edge_features.shape[1], self.n_edge_features))

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        edge_features = x[2]
        pairwise = np.asarray(w[self.n_input_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        return np.dot(edge_features, pairwise).reshape(
            edge_features.shape[0], self.n_states, self.n_states)

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features = self._get_features(x)
        unary_params = w[:self.n_input_states * self.n_features].reshape(
            self.n_input_states, self.n_features)

        if self.latent_node_features:
            unaries = np.dot(features, unary_params.T)
            n_hidden = x[2]
            n_visible = features.shape[0] - n_hidden
        else:
            # we only have features for visible nodes
            n_visible, n_hidden = features.shape[0], x[3]
            # assemble unary potentials for all nodes from observed evidence
            unaries = np.zeros((n_visible + n_hidden, self.n_states))
            unaries_observed = np.dot(features, unary_params.T)
            # paste observed into large matrix
            unaries[:n_visible, :self.n_labels] = unaries_observed
        # forbid latent states for observable nodes
        max_entry = np.maximum(np.max(unaries), 1)
        unaries[:n_visible, self.n_labels:] = -1e2 * max_entry
        # forbid observed states for latent nodes
        unaries[n_visible:, :self.n_labels] = -1e2 * max_entry
        return unaries

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
            inds = np.where(self.label_from_latent(h) != l)[0]
            unary_potentials[inds, l] += self.class_weight[
                self.label_from_latent(h)][inds]

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        unary_potentials = self._get_unary_potentials(x, w)
        # clamp observed nodes by modifying unary potentials
        max_entry = np.maximum(np.max(unary_potentials), 1)
        unary_potentials[np.arange(len(y)), y] = 1e2 * max_entry
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        if (h[:len(y)] != y).any():
            print("inconsistent h and y")
            h[:len(y)] = y
        return h

    def label_from_latent(self, h):
        return h[h < self.n_labels]

    def loss(self, h, h_hat):
        if isinstance(h_hat, tuple):
            return self.continuous_loss(h, h_hat[0])
        return GraphCRF.loss(self, self.label_from_latent(h),
                             self.label_from_latent(h_hat))

    def base_loss(self, y, y_hat):
        if isinstance(y_hat, tuple):
            return GraphCRF.continuous_loss(self, y, y_hat)
        return GraphCRF.loss(self, y, y_hat)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        y_org = self.label_from_latent(y)
        y_hat_org = y_hat[:y_org.size, :self.n_labels]
        return GraphCRF.continuous_loss(self, y_org, y_hat_org)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the
        configuration (x, y) and a weight vector w is given by
        np.dot(w, joint_feature(x, y)).

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
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]
        edge_features = x[2]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
        else:
            n_nodes = y.size
            gx = np.ogrid[:n_nodes]

            # make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            # pairwise
            pw = [np.outer(unary_marginals[edge[0]].T,
                           unary_marginals[edge[1]]).ravel()
                  for edge in edges]
            pw = np.vstack(pw)

        pw = np.dot(edge_features.T, pw)
        for i in self.symmetric_edge_features:
            pw_ = pw[i].reshape(self.n_states, self.n_states)
            pw[i] = (pw_ + pw_.T).ravel() / 2.

        for i in self.antisymmetric_edge_features:
            pw_ = pw[i].reshape(self.n_states, self.n_states)
            pw[i] = (pw_ - pw_.T).ravel() / 2.

        n_visible = features.shape[0]
        unaries_acc = np.dot(unary_marginals[:n_visible,
                                             :self.n_input_states].T, features)

        joint_feature_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return joint_feature_vector

    def init_latent(self, X, Y):
        # treat all edges the same
        return kmeans_init(X, Y, n_labels=self.n_labels,
                           n_hidden_states=self.n_hidden_states,
                           latent_node_features=self.latent_node_features)

    def max_loss(self, h):
        # maximum possible los on y for macro averages
        y = self.label_from_latent(h)
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y])
        return y.size
        return y.size
