import numpy as np

from .crf import CRF


class GraphCRF(CRF):
    """Pairwise CRF on a general graph.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Node features are given as a tuple of shape (n_nodes, n_features),
    An instance ``x`` is represented as a tuple ``(features, edges)``
    where edges is an array of shape (n_edges, 2), representing the graph.

    Labels ``y`` are given as array of shape (n_features)

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    n_features : int, default=None
        Number of features per node. None means n_states.

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
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo',
                 class_weight=None):
        CRF.__init__(self, n_states, n_features, inference_method,
                     class_weight=class_weight)
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = (n_states * self.n_features
                         + n_states * (n_states + 1) / 2)

    def get_edges(self, x):
        return x[1]

    def get_features(self, x):
        return x[0]

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
        pairwise_flat = np.asarray(w[self.n_states * self.n_features:])
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
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)
        return np.dot(features, unary_params.T)

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
        n_nodes = features.shape[0]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            y = y.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ##accumulated pairwise
            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])

        unaries_acc = np.dot(unary_marginals.T, features)
        pw = pw + pw.T - np.diag(np.diag(pw))  # make symmetric

        psi_vector = np.hstack([unaries_acc.ravel(),
                                pw[np.tri(self.n_states, dtype=np.bool)]])
        return psi_vector


class EdgeTypeGraphCRF(GraphCRF):
    """CRF with several kinds of edges, each having their own parameters.

    Pairwise potentials are not symmetric and are independend for each kind of
    edges. This leads to n_classes * n_features parameters for unary potentials
    and n_edge_types * n_classes ** 2 parameters for edge potentials.

    Unary evidence is given as a tuple of shape (n_nodes, n_features),
    An instance ``x`` is represented as a tuple ``(unaries, edges)``.
    Edges is a list of arrays of shape (n_edges_i, 2), one array
    per type of edge.

    Labels ``y`` are given as array of shape (n_features).

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    inference_method : string, default="qpbo"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'qpbo' for QPBO + alpha expansion.
            - 'dai' for LibDAI bindings (which has another parameter).
            - 'lp' for Linear Programming relaxation using GLPK.
            - 'ad3' for AD3 dual decomposition.

    n_edge_types : int, default=1
        How many different edge-types there are.

    """
    def __init__(self, n_states=2, n_features=None, inference_method='lp',
                 n_edge_types=1):
        GraphCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method,)
        self.n_edge_types = n_edge_types
        self.size_psi = (n_states * self.n_features
                         + self.n_edge_types * n_states ** 2)

    def get_edges(self, x, flat=True):
        if flat:
            # flatten edge-types
            return np.vstack(x[1])
        return x[1]

    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : tuple
            Input representation.

        y : ndarray or tuple
            Either y is an integral ndarray of shape (n_nodes,), giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``, where
            unary_marginals is an array of shape (n_nodes, n_states) and
            pairwise_marginals is an array of shape (n_states, n_states) of
            accumulated pairwise marginals.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        # x is unaries
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x, flat=False)
        n_nodes = features.shape[0]
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            unary_marginals, pw = y

            # pw contains separate entries for all edges
            # we need to find out which belong to which kind
            n_edges = [len(e) for e in edges]
            n_edges.insert(0, 0)
            edge_boundaries = np.cumsum(n_edges)
            pw_accumulated = []
            for i, j in zip(edge_boundaries[:-1], edge_boundaries[1:]):
                pw_accumulated.append(pw[i:j].sum(axis=0))
            pw = np.vstack(pw_accumulated)
        else:
            ##accumulated pairwise
            #make one hot encoding
            y = y.reshape(n_nodes)
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ##accumulated pairwise
            pw = []
            for edge_type in edges:
                pw.append(np.dot(unary_marginals[edge_type[:, 0]].T,
                                 unary_marginals[edge_type[:, 1]]))
            pw = np.vstack(pw)

        unaries_acc = np.dot(unary_marginals.reshape(-1, self.n_states).T,
                             features)
        feature = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return feature

    def get_pairwise_potentials(self, x, w):
        self._check_size_w(w)
        self._check_size_x(x)
        edges = self.get_edges(x, flat=False)
        n_edges = [len(e) for e in edges]
        pairwise_params = w[self.n_states * self.n_features:].reshape(
            self.n_edge_types, self.n_states, self.n_states)
        edge_weights = [np.repeat(pw[np.newaxis, :, :], n, axis=0)
                        for pw, n in zip(pairwise_params, n_edges)]
        return np.vstack(edge_weights)
