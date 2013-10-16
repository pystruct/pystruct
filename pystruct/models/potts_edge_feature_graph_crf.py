import numpy as np

from .edge_feature_graph_crf import EdgeFeatureGraphCRF


class PottsEdgeFeatureGraphCRF(EdgeFeatureGraphCRF):
    """Potts CRF with features/strength associated to each edge.

    Edge potentials are 0 if neighbors share the same state, some
    other value otherwise.

    Node features and edge features are given as a tuple of shape (n_nodes,
    n_features) and (n_edges, n_edge_features) respectively.

    An instance ``x`` is represented as a tuple ``(node_features, edges,
    edge_features)`` where edges is an array of shape (n_edges, 2),
    representing the graph.

    Labels ``y`` are given as array of shape (n_features)

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    n_features : int, default=None
        Number of features per node. None means n_states.

    n_edge_features : int, default=1
        Number of features per edge.

    inference_method : string, default="ad3"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'qpbo' for QPBO + alpha expansion.
            - 'dai' for LibDAI bindings (which has another parameter).
            - 'lp' for Linear Programming relaxation using GLPK.
            - 'ad3' for AD3 dual decomposition.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    """
    def __init__(self, n_states=None, n_features=None, n_edge_features=None,
                 inference_method=None, class_weight=None):

        EdgeFeatureGraphCRF.__init__(self, n_states, 
                                     n_features, 
                                     n_edge_features,
                                     inference_method, 
                                     class_weight)

    def _set_size_psi(self):
        if not None in [self.n_states, self.n_features, self.n_edge_features]:
            self.size_psi = (self.n_states * self.n_features
                             + self.n_edge_features)


    def _get_pairwise_potentials(self, x, w):
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
        edge_features = self._get_edge_features(x)
        n_edges = edge_features.shape[0]
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        pairwise = np.dot(edge_features, pairwise)

        matrix_potentials = np.repeat(pairwise, 
                                      self.n_states ** 2).reshape(n_edges,
                                                                  self.n_states,
                                                                  self.n_states)
        for i in xrange(n_edges) :
             for j in xrange(self.n_states) :
                 matrix_potentials[i, j, j] = 0

        return matrix_potentials


    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

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
        p : ndarray, shape (size_psi,)
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

            class_pair_ind = y[edges[:, 0]] != y[edges[:, 1]]
            pw = np.sum(edge_features[class_pair_ind], axis=0)

        unaries_acc = np.dot(unary_marginals.T, features)

        psi_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return psi_vector

