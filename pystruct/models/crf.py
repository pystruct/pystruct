import numpy as np

from .base import StructuredModel
from ..inference import inference_dispatch, get_installed
from .utils import loss_augment_unaries


class CRF(StructuredModel):
    """Abstract base class"""
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None):
        self.n_states = n_states
        if inference_method is None:
            # get first in list that is installed
            inference_method = get_installed(['ad3', 'max-product', 'lp'])[0]
        self.inference_method = inference_method
        self.inference_calls = 0
        self.n_features = n_features
        self.class_weight = class_weight
        self._set_size_joint_feature()
        self._set_class_weight()

    def initialize(self, X, Y):
        # Works for both GridCRF and GraphCRF, but not ChainCRF.
        # funny that ^^
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight()

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s)"
                % (type(self).__name__, self.n_states,
                   self.inference_method))

    def _check_size_x(self, x):
        features = self._get_features(x)
        if features.shape[1] != self.n_features:
            raise ValueError("Unary evidence should have %d feature per node,"
                             " got %s instead."
                             % (self.n_features, features.shape[1]))

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented Inference for x relative to y using parameters w.

        Finds (approximately)
        armin_y_hat np.dot(w, joint_feature(x, y_hat)) + loss(y, y_hat)
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

        w : ndarray, shape=(size_joint_feature,)
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
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        loss_augment_unaries(unary_potentials, np.asarray(y), self.class_weight)

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def inference(self, x, w, relaxed=False, return_energy=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, joint_feature(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        w : ndarray, shape=(size_joint_feature,)
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
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)
