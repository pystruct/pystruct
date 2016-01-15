from .graph_crf import GraphCRF
from .edge_feature_graph_crf import EdgeFeatureGraphCRF
from .crf import CRF
from ..utils import make_grid_edges, edge_list_to_features


class GridCRF(GraphCRF):
    """Pairwise CRF on a 2d grid.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Unary evidence ``x`` is given as array of shape (width, height, n_features),
    labels ``y`` are given as array of shape (width, height). Grid sizes do not
    need to be constant over the dataset.

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    inference_method : string, default="ad3"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

    neighborhood : int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 neighborhood=4):
        self.neighborhood = neighborhood
        GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method)

    def _get_edges(self, x):
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def _get_features(self, x):
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


class DirectionalGridCRF(GridCRF, EdgeFeatureGraphCRF):
    """CRF in which each direction of edges has their own set of parameters.

    Pairwise potentials are not symmetric and are independend for each kind of
    edges. This leads to n_classes * n_features parameters for unary potentials
    and n_edge_features * n_classes ** 2 parameters for edge potentials.
    The number of edge-types is two for a 4-connected neighborhood
    (horizontal and vertical) or 4 for a 8 connected neighborhood (additionally
    two diagonals).

    Unary evidence ``x`` is given as array of shape (width, height, n_features),
    labels ``y`` are given as array of shape (width, height). Grid sizes do not
    need to be constant over the dataset.

    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables.

    inference_method : string, default=None
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

    neighborhood : int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 neighborhood=4):
        self.neighborhood = neighborhood
        n_edge_features = 2 if neighborhood == 4 else 4
        EdgeFeatureGraphCRF.__init__(self, n_states, n_features,
                                     n_edge_features,
                                     inference_method=inference_method)

    def _set_size_joint_feature(self):
        if self.n_features is not None and self.n_states is not None:
            self.size_joint_feature = (
                self.n_states * self.n_features
                + self.n_edge_features * self.n_states ** 2)

    def _check_size_x(self, x):
        GridCRF._check_size_x(self, x)

    def initialize(self, X, Y):
        # we don't want to infere n_edge_features as in EdgeFeatureGraphCRF
        CRF.initialize(self, X, Y)

    def _get_edges(self, x, flat=True):
        return make_grid_edges(x, neighborhood=self.neighborhood,
                               return_lists=not flat)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : ndarray, shape (width, height, n_features)
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
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        return EdgeFeatureGraphCRF.joint_feature(self, x, y)

    def _get_edge_features(self, x):
        return edge_list_to_features(self._get_edges(x, flat=False))
