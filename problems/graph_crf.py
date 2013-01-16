import numpy as np

from .crf import CRF


class GraphCRF(CRF):
    """Pairwise CRF on a general graph.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Unary evidence is given as array of shape (width, height, n_states),
    labels ``y`` are given as array of shape (width, height) and the graph
    is given as nd-array of edges of shape (n_edges, 2).
    An instance ``x`` is represented as a tuple ``(unaries, edges)``.

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
    """
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo'):
        CRF.__init__(self, n_states, n_features, inference_method)
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
