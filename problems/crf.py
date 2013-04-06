import numpy as np

from .base import StructuredProblem
from ..inference import inference_dispatch


class CRF(StructuredProblem):
    """Abstract base class"""
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo',
                 class_weight=None):
        self.n_states = n_states
        self.inference_method = inference_method
        self.inference_calls = 0
        if n_features is None:
            # backward compatibilty hack
            n_features = n_states
        self.n_features = n_features

        if class_weight is not None:
            if len(class_weight) != n_states:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(class_weight)
        else:
            self.class_weight = np.ones(n_states)

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s)"
                % (type(self).__name__, self.n_states,
                   self.inference_method))

    def _check_size_x(self, x):
        features = self.get_features(x)
        if features.shape[1] != self.n_features:
            raise ValueError("Unary evidence should have %d feature per node,"
                             " got %s instead."
                             % (self.n_features, features.shape[1]))

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
            unaries are an nd-array of shape (n_nodes, n_features),
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
            By default an inter ndarray of shape=(n_nodes)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (n_nodes, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self.get_unary_potentials(x, w)
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        # do loss-augmentation
        for l in np.arange(self.n_states):
            # for each class, decrement features
            # for loss-agumention
            mask = y != l
            unary_potentials[mask, l] += self.class_weight[y][mask]

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

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
        self.inference_calls += 1
        unary_potentials = self.get_unary_potentials(x, w)
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)
