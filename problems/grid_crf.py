import numpy as np

from .graph_crf import GraphCRF
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
        GraphCRF.__init__(self, n_states, n_features, inference_method)
        self.neighborhood = neighborhood

    def get_edges(self, x):
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def get_features(self, x):
        return x.reshape(-1, self.n_features)

    def get_unary_potentials(self, x, w):
        res = GraphCRF.get_unary_potentials(self, x, w)
        return res.reshape(x.shape[0], x.shape[1], self.n_states)


class DirectionalGridCRF(GridCRF):
    """CRF in which each direction of edges has their own set of parameters.

    Pairwise potentials are not symmetric and are independend for each kind of
    edges. This leads to n_classes parameters for unary potentials and
    n_edge_types * n_classes ** 2 parameters for edge potentials.
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
        GridCRF.__init__(self, n_states, n_features, inference_method,
                         neighborhood=neighborhood)
        self.n_edge_types = 2 if neighborhood == 4 else 4
        self.size_psi = (n_states * self.n_features
                         + self.n_edge_types * n_states ** 2)

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
        # x is unaries
        self._check_size_x(x)
        x_flat = x.reshape(-1, x.shape[-1])
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            unary_marginals, pw = y

            # pw contains separate entries for all edges
            # we need to find out which belong to which kind
            edges = make_grid_edges(x, neighborhood=self.neighborhood,
                                    return_lists=True)
            n_edges = [len(e) for e in edges]
            n_edges.insert(0, 0)
            edge_boundaries = np.cumsum(n_edges)
            pw_accumulated = []
            for i, j in zip(edge_boundaries[:-1], edge_boundaries[1:]):
                pw_accumulated.append(pw[i:j].sum(axis=0))
            pw = np.hstack(pw_accumulated)
        else:
            ## unary features:
            gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]

            ##accumulated pairwise
            #make one hot encoding
            unary_marginals = np.zeros((y.shape[0], y.shape[1], self.n_states),
                                       dtype=np.int)
            unary_marginals[gx, gy, y] = 1
            pw = np.vstack(pairwise_grid_features(unary_marginals,
                                                  self.neighborhood))

        unaries_acc = np.dot(x_flat.T, unary_marginals.reshape(-1,
                                                               self.n_states))
        feature = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return feature

    def get_pairwise_potentials(self, x, w):
        self._check_size_w(w)
        self._check_size_x(x)
        edges = make_grid_edges(x, neighborhood=self.neighborhood,
                                return_lists=True)
        n_edges = [len(e) for e in edges]
        pairwise_params = w[self.n_states * self.n_features:].reshape(
            self.n_edge_types, self.n_states, self.n_states)
        edge_weights = [np.repeat(pw[np.newaxis, :, :], n, axis=0)
                        for pw, n in zip(pairwise_params, n_edges)]
        return np.vstack(edge_weights)
