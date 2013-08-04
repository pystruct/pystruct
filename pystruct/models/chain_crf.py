import numpy as np

from .graph_crf import GraphCRF


def make_chain_edges(x):
    # this can be optimized sooooo much!
    inds = np.arange(x.shape[0])
    edges = np.c_[inds[:-1], inds[1:]]
    return edges


# TESTME
# def make_chain_edges_fast(x):
#     n_nodes = np.shape(x)[0]
#     return as_strided(np.arange(n_noes), shape=(n_nodes - 1, 2),
#                       strides=(8, 8)  # 8 should be dtype size


class ChainCRF(GraphCRF):
    """Linear-chain CRF

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Unary evidence ``x`` is given as array of shape (n_nodes, n_features), and
    labels ``y`` are given as array of shape (n_nodes,). Chain lengths do not
    need to be constant over the dataset.

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
    def __init__(self, n_states, n_features, inference_method='qpbo'):
        GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method)

    def get_edges(self, x):
        return make_chain_edges(x)

    def get_features(self, x):
        return x
