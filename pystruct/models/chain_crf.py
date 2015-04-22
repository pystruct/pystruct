import numpy as np

from .graph_crf import GraphCRF


def make_chain_edges(x):
    # this can be optimized sooooo much!
    inds = np.arange(x.shape[0])
    edges = np.concatenate([inds[:-1, np.newaxis], inds[1:, np.newaxis]],
                           axis=1)
    return edges


class ChainCRF(GraphCRF):
    """Linear-chain CRF.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to ``n_classes`` parameters for unary potentials.
    If ``directed=True``, there are ``n_classes * n_classes`` parameters
    for pairwise potentials, if ``directed=False``, there are only
    ``n_classes * (n_classes + 1) / 2`` (for a symmetric matrix).

    Unary evidence ``x`` is given as array of shape (n_nodes, n_features), and
    labels ``y`` are given as array of shape (n_nodes,). Chain lengths do not
    need to be constant over the dataset.

    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables.
        Inferred from data if not provided.

    inference_method : string or None, default=None
        Function to call do do inference and loss-augmented inference.
        Defaults to "max-product" for max-product belief propagation.
        As chains can be solved exactly and efficiently, other settings
        are not recommended.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    directed : boolean, default=False
        Whether to model directed or undirected connections.
        In undirected models, interaction terms are symmetric,
        so an edge ``a -> b`` has the same energy as ``b -> a``.
    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None, directed=True):
        if inference_method is None:
            inference_method = "max-product"
        GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method,
                          class_weight=class_weight, directed=directed)

    def _get_edges(self, x):
        return make_chain_edges(x)

    def _get_features(self, x):
        return x

    def initialize(self, X, Y):
        n_features = X[0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight()
