import itertools

import numpy as np

from inference_methods import (_inference_qpbo, _inference_dai, _inference_lp,
                               _inference_ad3, _make_grid_edges)

from IPython.core.debugger import Tracer
tracer = Tracer()


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


def unwrap_pairwise(y):
    """given a y that may contain pairwise marginals, yield plain y."""
    if isinstance(y, tuple):
        return y[0]
    return y


class StructuredProblem(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least psi and inference.
    """

    def __init__(self):
        self.size_psi = None

    def psi(self, x, y):
        # IMPLEMENT ME
        pass

    def _loss_augmented_dpsi(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.psi(x_loss_augmented, y)
                - self.psi(x_loss_augmented, y_hat))

    def inference(self, x, w):
        # IMPLEMENT ME
        pass

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_one_hot = np.zeros_like(y_hat)
        if y.ndim == 2:
            gx, gy = np.indices(y.shape)
            y_one_hot[gx, gy, y] = 1
        else:
            gx = np.indices(y.shape)
            y_one_hot[gx, y] = 1

        # all entries minus correct ones
        return np.prod(y.shape) - np.sum(y_one_hot * y_hat)

    def loss_augmented_inference(self, x, y, w):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)


class CRF(StructuredProblem):
    """Abstract base class"""
    def __init__(self, n_states=2, inference_method='qpbo'):
        self.n_states = n_states
        self.inference_method = inference_method
        self.inference_calls = 0

    def __repr__(self):
        return ("GridCRF, n_states: %d, inference_method: %s"
                % (self.n_states, self.inference_method))

    def loss_augment(self, x, y, w):
        """Modifies x to model loss-augmentation.

        Modifies x such that
        np.dot(psi(x, y_hat), w) == np.dot(psi(x, y_hat), w) + loss(y, y_hat)

        Parameters
        ----------
        x : ndarray, shape (n_nodes, n_states)
            Unary evidence / input to augment.

        y : ndarray, shape (n_nodes,)
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_psi,)
            Weights that will be used for inference.
            TODO: refactor this :-/

        Return
        """
        unary_params = w[:self.n_states].copy()
        # avoid division by zero:
        if (unary_params == 0).any():
            raise ValueError("Unary params are exactly zero, can not do"
                             " loss-augmentation!")
        x_ = x.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_[y != l, l] += 1. / unary_params[l]
        return x_

    def loss_augmented_inference(self, x, y, w, relaxed=False):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        x_ = self.loss_augment(x, y, w)
        return self.inference(x_, w, relaxed)


class GridCRF(CRF):
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

    neighborhood: int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=2, inference_method='qpbo', neighborhood=4):
        CRF.__init__(self, n_states, inference_method)
        self.neighborhood = neighborhood
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * (n_states + 1) / 2

    def get_pairwise_weights(self, w):
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
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        # set lower triangle of matrix, then make symmetric
        # we could try to redo this using ``scipy.spatial.distance`` somehow
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        return (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

    def get_unary_weights(self, w):
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
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[:self.n_states]

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
            pairwise_marginals is an array of shape
            (n_edges, n_states, n_states).

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        # x is unaries
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            y, pw = y
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            unaries_acc = np.sum(x_flat * y_flat, axis=0)
        else:
            ## unary features:
            gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
            selected_unaries = x[gx, gy, y]
            unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                      minlength=self.n_states)

            ##accumulated pairwise
            #make one hot encoding
            labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                              dtype=np.int)
            labels[gx, gy, y] = 1
            pw = np.sum(pairwise_grid_features(labels, self.neighborhood),
                        axis=0)

        pw = pw + pw.T - np.diag(np.diag(pw))
        feature = np.hstack([unaries_acc,
                             pw[np.tri(self.n_states, dtype=np.bool)]])
        return feature

    def inference(self, x, w, relaxed=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, psi(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : ndarray, shape=(width, height, n_states)
            Unary evidence / input.

        w : ndarray, shape=(size_psi,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(width, height)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (width, height, n_states),
            pairwise_marginals is an array of
            shape (n_edges, n_states, n_states).

        """
        unary_params = self.get_unary_weights(w)
        pairwise_params = self.get_pairwise_weights(w)
        self.inference_calls += 1
        edges = _make_grid_edges(x, neighborhood=self.neighborhood)
        if self.inference_method == "qpbo":
            return _inference_qpbo(x, unary_params, pairwise_params, edges)
        elif self.inference_method == "dai":
            return _inference_dai(x, unary_params, pairwise_params, edges)
        elif self.inference_method == "lp":
            return _inference_lp(x, unary_params, pairwise_params, edges,
                                 relaxed)
        elif self.inference_method == "ad3":
            return _inference_ad3(x, unary_params, pairwise_params, edges,
                                  relaxed)
        else:
            raise ValueError("inference_method must be 'qpbo' or 'dai', got %s"
                             % self.inference_method)


class DirectionalGridCRF(CRF):
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
    def __init__(self, n_states=2, inference_method='lp', neighborhood=4):
        CRF.__init__(self, n_states, inference_method)
        self.n_edge_types = 2 if neighborhood == 4 else 4
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * n_states * self.n_edge_types
        self.neighborhood = neighborhood

    def get_pairwise_weights(self, w):
        """Extracts the pairwise part of the weight vector.

        Parameters
        ----------
        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_edge_types, n_states, n_states)
            Pairwise weights.
        """

        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[self.n_states:].reshape(self.n_edge_types, self.n_states,
                                         self.n_states)

    def get_unary_weights(self, w):
        """Extracts the unary part of the weight vector.

        Parameters
        ----------
        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states,)
            Unary weights.
        """
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[:self.n_states]

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
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            y, pw = y
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            unaries_acc = np.sum(x_flat * y_flat, axis=0)
            # pw contains separate entries for all edges
            # we need to find out which belong to which kind
            edges = _make_grid_edges(x, neighborhood=self.neighborhood,
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
            selected_unaries = x[gx, gy, y]
            unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                      minlength=self.n_states)

            ##accumulated pairwise
            #make one hot encoding
            labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                              dtype=np.int)
            labels[gx, gy, y] = 1
            pw = np.vstack(pairwise_grid_features(labels, self.neighborhood))

        feature = np.hstack([unaries_acc, pw.ravel()])
        return feature

    def inference(self, x, w, relaxed=False, return_energy=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, psi(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : ndarray, shape=(width, height, n_states)
            Unary evidence / input.

        w : ndarray, shape=(size_psi,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(width, height)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (width, height, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self.inference_calls += 1
        # extract unary weights
        unary_params = self.get_unary_weights(w)
        # extract pairwise weights of shape n_edge_types x n_states x n_states
        pairwise_params = self.get_pairwise_weights(w)
        edges = _make_grid_edges(x, neighborhood=self.neighborhood,
                                 return_lists=True)
        n_edges = [len(e) for e in edges]
        # replicate pairwise weights for edges of certain type
        edge_weights = [np.repeat(pw[np.newaxis, :, :], n, axis=0)
                        for pw, n in zip(pairwise_params, n_edges)]
        edge_weights = np.vstack(edge_weights)
        edges = np.vstack(edges)

        if self.inference_method == "qpbo":
            return _inference_qpbo(x, unary_params, edge_weights, edges)
        #elif self.inference_method == "dai":
            #return _inference_dai(x, unary_params, edge_weights, edges)
        elif self.inference_method == "lp":
            return _inference_lp(x, unary_params, edge_weights, edges, relaxed,
                                 return_energy=return_energy)
        elif self.inference_method == "ad3":
            return _inference_ad3(x, unary_params, edge_weights, edges,
                                  relaxed)
        else:
            raise ValueError("inference_method must be 'lp' or"
                             " 'ad3', got %s" % self.inference_method)


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
    def __init__(self, n_states=2, inference_method='qpbo'):
        CRF.__init__(self, n_states, inference_method)
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * (n_states + 1) / 2

    def get_pairwise_weights(self, w):
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
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        # set lower triangle of matrix, then make symmetric
        # we could try to redo this using ``scipy.spatial.distance`` somehow
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        return (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

    def get_unary_weights(self, w):
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
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[:self.n_states]

    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        y : ndarray or tuple
            Either y is an integral ndarray of shape (n_nodes), giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``, where
            unary_marginals is an array of shape (n_nodes, n_states) and
            pairwise_marginals is an array of shape
            (n_states, n_states).

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        unary_evidence, edges = x

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unaries_acc = np.sum(unary_evidence * unary_marginals, axis=0)
        else:
            n_nodes = y.shape[0]
            gx = np.ogrid[:n_nodes]
            selected_unaries = unary_evidence[gx, y]
            unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                      minlength=self.n_states)

            ##accumulated pairwise
            #make one hot encoding
            labels = np.zeros((n_nodes, self.n_states),
                              dtype=np.int)
            gx = np.ogrid[:n_nodes]
            labels[gx, y] = 1
            pw = np.dot(labels[edges[:, 0]].T, labels[edges[:, 1]])

        pw = pw + pw.T - np.diag(np.diag(pw))  # make symmetric

        feature = np.hstack([unaries_acc,
                             pw[np.tri(self.n_states, dtype=np.bool)]])
        return feature

    def inference(self, x, w, relaxed=False, return_energy=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, psi(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        w : ndarray, shape=(size_psi,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(width, height)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (width, height, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        unaries, edges = x
        self.inference_calls += 1
        unary_params = self.get_unary_weights(w)
        pairwise_params = self.get_pairwise_weights(w)
        self.inference_calls += 1
        if self.inference_method == "qpbo":
            return _inference_qpbo(unaries, unary_params, pairwise_params,
                                   edges)
        elif self.inference_method == "dai":
            return _inference_dai(unaries, unary_params, pairwise_params,
                                  edges)
        elif self.inference_method == "lp":
            return _inference_lp(unaries, unary_params, pairwise_params, edges,
                                 relaxed)
        elif self.inference_method == "ad3":
            return _inference_ad3(unaries, unary_params, pairwise_params,
                                  edges, relaxed)
        else:
            raise ValueError("inference_method must be 'qpbo' or 'dai', got %s"
                             % self.inference_method)

    def loss_augment(self, x, y, w):
        unary_params = w[:self.n_states].copy()
        unaries, edges = x
        # avoid division by zero:
        if (unary_params == 0).any():
            raise ValueError("Unary params are exactly zero, can not do"
                             " loss-augmentation!")
        unaries_ = unaries.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            unaries_[y != l, l] += 1. / unary_params[l]
        return (unaries_, edges)


def exhaustive_loss_augmented_inference(problem, x, y, w):
    size = np.prod(x.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(problem.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(x.shape[:-1])
        #print("trying %s" % repr(y_hat))
        psi = problem.psi(x, y_hat)
        energy = -problem.loss(y, y_hat) - np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y
