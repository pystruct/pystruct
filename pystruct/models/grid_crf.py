import numpy as np

from .graph_crf import GraphCRF, EdgeTypeGraphCRF
from ..utils import make_grid_edges


def pairwise_grid_features(grid_labels, neighborhood=4):
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood has to be 4 or 8.")
    n_states = grid_labels.shape[-1]
    features = []
    # horizontal edges
    right = np.dot(grid_labels[:, :-1, :].reshape(-1, n_states).T,
                   grid_labels[:, 1:, :].reshape(-1, n_states))
    features.append(right)
    # vertical edges
    down = np.dot(grid_labels[:-1, :, :].reshape(-1, n_states).T,
                  grid_labels[1:, :, :].reshape(-1, n_states))
    features.append(down)
    if neighborhood == 8:
        upright = np.dot(grid_labels[1:, :-1, :].reshape(-1, n_states).T,
                         grid_labels[:-1, 1:, :].reshape(-1, n_states))
        features.append(upright)
        downright = np.dot(grid_labels[:-1, :-1, :].reshape(-1, n_states).T,
                           grid_labels[1:, 1:, :].reshape(-1, n_states))
        features.append(downright)
    return features


class GridCRF(GraphCRF):
    """Pairwise CRF on a 2d grid.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Unary evidence ``x`` is given as array of shape (width, height, n_states),
    labels ``y`` are given as array of shape (width, height). Grid sizes do not
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

    neighborhood : int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo',
                 neighborhood=4):
        GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method)
        self.neighborhood = neighborhood

    def get_edges(self, x):
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def get_features(self, x):
        return x.reshape(-1, self.n_features)

    def _reshape_y(self, y, shape_x, return_energy):
        if return_energy:
            y, energy = y

        if isinstance(y, tuple):
            y = (y[0].reshape(shape_x[0], shape_x[1], y[0].shape[1]), y[1])
        else:
            y = y.reshape(shape_x[:-1])

        if return_energy:
            return y, energy
        return y

    def inference(self, x, w, relaxed=False, return_energy=False):
        y = GraphCRF.inference(self, x, w, relaxed=relaxed,
                               return_energy=return_energy)
        return self._reshape_y(y, x.shape, return_energy)

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


class DirectionalGridCRF(GridCRF, EdgeTypeGraphCRF):
    """CRF in which each direction of edges has their own set of parameters.

    Pairwise potentials are not symmetric and are independend for each kind of
    edges. This leads to n_classes * n_features parameters for unary potentials
    and n_edge_types * n_classes ** 2 parameters for edge potentials.
    The number of edge-types is two for a 4-connected neighborhood
    (horizontal and vertical) or 4 for a 8 connected neighborhood (additionally
    two diagonals).

    Unary evidence ``x`` is given as array of shape (width, height, n_states),
    labels ``y`` are given as array of shape (width, height). Grid sizes do not
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

    neighborhood : int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=2, n_features=None, inference_method='lp',
                 neighborhood=4):
        GridCRF.__init__(self, n_states, n_features,
                         inference_method=inference_method,
                         neighborhood=neighborhood)
        self.n_edge_types = 2 if neighborhood == 4 else 4
        self.size_psi = (n_states * self.n_features
                         + self.n_edge_types * n_states ** 2)

    def get_edges(self, x, flat=True):
        return make_grid_edges(x, neighborhood=self.neighborhood,
                               return_lists=not flat)

    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : ndarray, shape (width, height, n_states)
            Unary evidence / input.

        y : ndarray or tuple
            Either y is an integral ndarray of shape (width, height), giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``, where
            unary_marginals is an array of shape (width, height, n_states) and
            pairwise_marginals is an array of shape (n_states, n_states) of
            accumulated pairwise marginals.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        return EdgeTypeGraphCRF.psi(self, x, y)

    def get_pairwise_potentials(self, x, w):
        return EdgeTypeGraphCRF.get_pairwise_potentials(self, x, w)
