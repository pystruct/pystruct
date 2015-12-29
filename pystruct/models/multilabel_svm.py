import numpy as np
from .crf import CRF


class MultiLabelClf(CRF):
    """Multi-label model for predicting several binary classes.

    Multi-label classification is a generalization of multi-class
    classification, in that multiple classes can be present in each
    example. This can also be thought of as predicting
    binary indicator per class.

    This class supports different models via the "edges" parameter.
    Giving no edges yields independent classifiers for each class. Giving
    "full" yields a fully connected graph over the labels, while "tree"
    yields the best tree-shaped graph (using the Chow-Liu algorithm).
    It is also possible to specify a custom connectivity structure.

    Parameters
    ----------
    n_labels : int (default=None)
        Number of labels. Inferred from data if not provided.

    n_features : int (default=None)
        Number of input features. Inferred from data if not provided.

    edges : array-like, string or None
        Either None, which yields independent models, 'tree',
        which yields the Chow-Liu tree over the labels, 'full',
        which yields a fully connected graph, or an array-like
        of edges for a custom dependency structure.

    inference_method :
        The inference method to be used.

    """
    def __init__(self, n_labels=None, n_features=None, edges=None,
                 inference_method=None):
        self.n_labels = n_labels
        self.edges = edges
        CRF.__init__(self, 2, n_features, inference_method)

    def _set_size_joint_feature(self):
        # try to set the size of joint_feature if possible
        if self.n_features is not None and self.n_states is not None:
            if self.edges is None:
                self.edges = np.zeros(shape=(0, 2), dtype=np.int)
            self.size_joint_feature = (self.n_features * self.n_labels + 4 *
                                       self.edges.shape[0])

    def initialize(self, X, Y):
        n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_labels = Y.shape[1]
        if self.n_labels is None:
            self.n_labels = n_labels
        elif self.n_labels != n_labels:
            raise ValueError("Expected %d labels, got %d"
                             % (self.n_labels, n_labels))

        self._set_size_joint_feature()
        self._set_class_weight()

    def _get_edges(self, x):
        return self.edges

    def _get_unary_potentials(self, x, w):
        unary_params = w[:self.n_labels * self.n_features].reshape(
            self.n_labels, self.n_features)
        unary_potentials = np.dot(x, unary_params.T)
        return np.vstack([-unary_potentials, unary_potentials]).T

    def _get_pairwise_potentials(self, x, w):
        pairwise_params = w[self.n_labels * self.n_features:].reshape(
            self.edges.shape[0], self.n_states, self.n_states)
        return pairwise_params

    def joint_feature(self, x, y):
        if isinstance(y, tuple):
            # continuous pairwise marginals
            y_cont, pairwise_marginals = y
            y_signs = 2 * y_cont[:, 1] - 1
            unary_marginals = np.repeat(x[np.newaxis, :], len(y_signs), axis=0)
            unary_marginals *= y_signs[:, np.newaxis]
        else:
            # discrete y
            y_signs = 2 * y - 1
            unary_marginals = np.repeat(x[np.newaxis, :], len(y_signs), axis=0)
            unary_marginals *= y_signs[:, np.newaxis]
            pairwise_marginals = []
            for edge in self.edges:
                # indicator of one of four possible states of the edge
                pw = np.zeros((2, 2))
                pw[y[edge[0]], y[edge[1]]] = 1
                pairwise_marginals.append(pw)

        if len(pairwise_marginals):
            pairwise_marginals = np.vstack(pairwise_marginals)
            return np.hstack([unary_marginals.ravel(),
                              pairwise_marginals.ravel()])
        return unary_marginals.ravel()
