import numpy as np

from .crf import CRF
from ..utils import expand_sym, compress_sym


class GraphCRF(CRF):
    """Pairwise CRF on a general graph.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Examples, i.e. X, are given as an iterable of n_examples.
    An example, x, is represented as a tuple (features, edges) where
    features is a numpy array of shape (n_nodes, n_attributes), and
    edges is is an array of shape (n_edges, 2), representing the graph.

    Labels, Y, are given as an interable of n_examples. Each label, y, in Y
    is given by a numpy array of shape (n_nodes,).

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    n_features : int, default=None
        Number of features per node. None means n_states.

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
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None):
        CRF.__init__(self, n_states, n_features, inference_method,
                     class_weight=class_weight)
        # n_states unary parameters, upper triangular for pairwise

    def _set_size_psi(self):
        # try to set the size of psi if possible
        if self.n_features is not None and self.n_states is not None:
            self.size_psi = (self.n_states * self.n_features
                             + self.n_states * (self.n_states + 1) / 2)

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
        return expand_sym(w[self.n_states * self.n_features:])

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

        psi_vector = np.hstack([unaries_acc.ravel(), compress_sym(pw)])
        return psi_vector
