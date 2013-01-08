import numpy as np

from ..inference import inference_dispatch
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
        self._check_size_w(w)
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
        self._check_size_w(w)
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
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
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
        self._check_size_w(w)
        features, edges = x
        self.inference_calls += 1
        unary_params = self.get_unary_weights(w)
        unary_potentials = features * unary_params

        pairwise_params = self.get_pairwise_weights(w)
        return inference_dispatch(unary_potentials, pairwise_params, edges,
                                  self.inference_method, relaxed)

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented Inference for x relative to y using parameters w.

        Finds (approximately)
        armin_y_hat np.dot(w, psi(x, y_hat)) + loss(y, y_hat)
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        y : ndarray, shape (n_nodes,)
            Ground truth labeling relative to which the loss
            will be measured.

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
        self._check_size_w(w)
        features, edges = x
        unary_params = self.get_unary_weights(w)
        unary_potentials = features * unary_params
        pairwise_params = self.get_pairwise_weights(w)

        # do loss-augmentation
        for l in np.arange(self.n_states):
            # for each class, decrement features
            # for loss-agumention
            unary_potentials[y != l, l] += 1.

        return inference_dispatch(unary_potentials, pairwise_params, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)
