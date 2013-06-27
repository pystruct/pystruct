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

    def _reshape_y(self, y, shape_x, return_energy):
        if return_energy:
            y, energy = y

        if isinstance(y, tuple):
            y = (y[0].reshape(shape_x , y[0].shape[1]), y[1])
        else:
            y = y.reshape(shape_x,)  # works for chains too

        if return_energy:
            return y, energy
        return y

    def inference(self, x, w, relaxed=False, return_energy=False):
        y = GraphCRF.inference(self, x, w, relaxed=relaxed,
                               return_energy=return_energy)
        return self._reshape_y(y, len(x), return_energy)

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        y_hat = GraphCRF.loss_augmented_inference(self, x, y.ravel(), w,
                                                  relaxed=relaxed,
                                                  return_energy=return_energy)
        return self._reshape_y(y_hat, x.shape, return_energy)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        return GraphCRF.continuous_loss(
            self, y.ravel(), y_hat.reshape(-1, y_hat.shape[-1]))
