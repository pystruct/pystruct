import numpy as np

from .graph_crf import GraphCRF


class EdgeFeatureGraphCRF(GraphCRF):
    """Pairwise CRF with features/strength associated to each edge.

    Pairwise potentials are a-symmetric and shared over all edges.
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
    """
    def __init__(self, n_states=2, n_features=None, n_edge_features=1,
                 inference_method='qpbo', class_weight=None):
        GraphCRF.__init__(self, n_states, n_features, inference_method,
                          class_weight=class_weight)
        self.n_edge_features = n_edge_features
        self.size_psi = (n_states * self.n_features
                         + self.n_edge_features
                         * n_states ** 2)

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
