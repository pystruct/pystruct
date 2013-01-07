import numpy as np

from .base import StructuredProblem

from IPython.core.debugger import Tracer
tracer = Tracer()


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

        Returns
        -------
        x_hat : ndarray, shape (n_nodes, n_states)
            Loss-augmented unary evidence.
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
