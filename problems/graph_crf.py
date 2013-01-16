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
        CRF.__init__(self, n_states, inference_method)
        # n_states unary parameters, upper triangular for pairwise
        if n_features is None:
            # backward compatibilty hack
            n_features = n_states
        self.n_features = n_features
        self.size_psi = n_states * n_features + n_states * (n_states + 1) / 2

    def get_pairwise_potentials(self, x, w):
        """Extracts the pairwise part of the weight vector.

        Parameters
        ----------
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
        """Extracts the unary part of the weight vector.

        Parameters
        ----------
        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features, edges = x
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)
        return np.dot(features, unary_params.T)

    def get_edges(self, x):
        return x[1]

    def get_features(self, x):
        return x[0]
