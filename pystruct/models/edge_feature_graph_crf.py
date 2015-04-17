import numpy as np

from .graph_crf import GraphCRF
from .crf import CRF


class EdgeFeatureGraphCRF(GraphCRF):
    """Pairwise CRF with features/strength associated to each edge.

    Pairwise potentials are asymmetric and shared over all edges.
    They are weighted by an edge-specific features, though.
    This allows for contrast sensitive potentials or directional potentials
    (using a {-1, +1} encoding of the direction for example).

    More complicated interactions are also possible, of course.

    Node features and edge features are given as a tuple of shape (n_nodes,
    n_features) and (n_edges, n_edge_features) respectively.

    An instance ``x`` is represented as a tuple ``(node_features, edges,
    edge_features)`` where edges is an array of shape (n_edges, 2),
    representing the graph.

    Labels ``y`` are given as array of shape (n_nodes)

    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables. Inferred from data if not provided.

    n_features : int, default=None
        Number of features per node. Inferred from data if not provided.

    n_edge_features : int, default=None
        Number of features per edge. Inferred from data if not provided.

    inference_method : string, default="ad3"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    symmetric_edge_features : None or list
        Indices of edge features that are forced to be symmetric.
        Often the direction of the edge has no immediate meaning.

    antisymmetric_edge_features : None or list
        Indices of edge features that are forced to be anti-symmetric.

    """
    def __init__(self, n_states=None, n_features=None, n_edge_features=None,
                 inference_method=None, class_weight=None,
                 symmetric_edge_features=None,
                 antisymmetric_edge_features=None):
        self.n_edge_features = n_edge_features

        if symmetric_edge_features is None:
            symmetric_edge_features = []
        if antisymmetric_edge_features is None:
            antisymmetric_edge_features = []
        self.symmetric_edge_features = symmetric_edge_features
        self.antisymmetric_edge_features = antisymmetric_edge_features

        GraphCRF.__init__(self, n_states, n_features, inference_method,
                          class_weight=class_weight)

    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features, self.n_edge_features]:
            self.size_joint_feature = (self.n_states * self.n_features +
                                       self.n_edge_features
                                       * self.n_states ** 2)

        if self.n_edge_features is not None:
            if np.any(np.hstack([self.symmetric_edge_features,
                                 self.antisymmetric_edge_features]) >=
                      self.n_edge_features):
                raise ValueError("Got (anti) symmetric edge feature index that"
                                 " is larger than n_edge_features.")

            if not set(self.symmetric_edge_features).isdisjoint(
                    self.antisymmetric_edge_features):
                raise ValueError("symmetric_edge_features and "
                                 " antisymmetric_edge_features share an entry."
                                 " That doesn't make any sense.")

    def initialize(self, X, Y):
        n_edge_features = X[0][2].shape[1]
        if self.n_edge_features is None:
            self.n_edge_features = n_edge_features
        elif self.n_edge_features != n_edge_features:
            raise ValueError("Expected %d edge features, got %d"
                             % (self.n_edge_features, n_edge_features))
        CRF.initialize(self, X, Y)

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s, n_features: %d, "
                "n_edge_features: %d)"
                % (type(self).__name__, self.n_states, self.inference_method,
                   self.n_features, self.n_edge_features))

    def _check_size_x(self, x):
        GraphCRF._check_size_x(self, x)

        _, edges, edge_features = x
        if edges.shape[0] != edge_features.shape[0]:
            raise ValueError("Got %d edges but %d edge features."
                             % (edges.shape[0], edge_features.shape[0]))
        if edge_features.shape[1] != self.n_edge_features:
            raise ValueError("Got edge features of size %d, but expected %d."
                             % (edge_features.shape[1], self.n_edge_features))

    def _get_edge_features(self, x):
        return x[2]

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
        edge_features = self._get_edge_features(x)
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        return np.dot(edge_features, pairwise).reshape(
            edge_features.shape[0], self.n_states, self.n_states)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : tuple
            Input representation.

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
        edge_features = self._get_edge_features(x)

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)

        else:
            y = y.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ## pairwise
            pw = np.zeros((edges.shape[0], self.n_states ** 2))
            class_pair_ind = (y[edges[:, 1]] + self.n_states *
                              y[edges[:, 0]])
            pw[np.arange(len(edges)), class_pair_ind] = 1

        pw = np.dot(edge_features.T, pw)
        for i in self.symmetric_edge_features:
            pw_ = pw[i].reshape(self.n_states, self.n_states)
            pw[i] = (pw_ + pw_.T).ravel() / 2.

        for i in self.antisymmetric_edge_features:
            pw_ = pw[i].reshape(self.n_states, self.n_states)
            pw[i] = (pw_ - pw_.T).ravel() / 2.

        unaries_acc = np.dot(unary_marginals.T, features)

        joint_feature_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return joint_feature_vector
