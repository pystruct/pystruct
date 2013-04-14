import numpy as np

from .graph_crf import GraphCRF


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

    Labels ``y`` are given as array of shape (n_features)

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    n_features : int, default=None
        Number of features per node. None means n_states.

    n_edge_features : int, default=1
        Number of features per edge.

    inference_method : string, default="qpbo"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'qpbo' for QPBO + alpha expansion.
            - 'dai' for LibDAI bindings (which has another parameter).
            - 'lp' for Linear Programming relaxation using GLPK.
            - 'ad3' for AD3 dual decomposition.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    symmetric_edge_features : None or list
        Indices of edge features that are forced to be symmetric.
        Often the direction of the edge has no immediate meaning.

    antisymmetric_edge_features : None or list
        Indices of edge features that are forced to be anti-symmetric.

    """
    def __init__(self, n_states=2, n_features=None, n_edge_features=1,
                 inference_method='qpbo', class_weight=None,
                 symmetric_edge_features=None,
                 antisymmetric_edge_features=None):
        GraphCRF.__init__(self, n_states, n_features, inference_method,
                          class_weight=class_weight)
        self.n_edge_features = n_edge_features
        self.size_psi = (n_states * self.n_features
                         + self.n_edge_features
                         * n_states ** 2)
        if not set(symmetric_edge_features).isdisjoint(
                antisymmetric_edge_features):
            raise ValueError("symmetric_edge_features and "
                             " antisymmetric_edge_features share an entry."
                             " That doesn't make any sense.")

        self.symmetric_edge_features = symmetric_edge_features
        self.antisymmetric_edge_features = antisymmetric_edge_features

    def _check_size_x(self, x):
        GraphCRF._check_size_x(self, x)

        _, edges, edge_features = x
        if edges.shape[0] != edge_features.shape[0]:
            raise ValueError("Got %d edges but %d edge features."
                             % (edges.shape[0], edge_features.shape[0]))
        if edge_features.shape[1] != self.n_edge_features:
            raise ValueError("Got edge features of size %d, but expected %d."
                             % (edge_features.shape[1], self.n_edge_features))

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
        edge_features = x[2]
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        return np.dot(edge_features, pairwise).reshape(
            edge_features.shape[0], self.n_states, self.n_states)

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
        features, edges = self.get_features(x), self.get_edges(x)
        n_nodes = features.shape[0]
        edge_features = x[2]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # weighted accumulation of pairwise

            #pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            y = y.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ## pairwise
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

        unaries_acc = np.dot(unary_marginals.T, features)

        psi_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return psi_vector
