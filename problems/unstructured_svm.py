import numpy as np

from .base import StructuredProblem


class BinarySVMProblem(StructuredProblem):
    """Formulate standard linear binary SVM in CRF framework.

    Inputs x are simply feature arrays, labels y are -1 or 1.
    An additional constant 1 feature (leading to a penalized bias/intercept) is
    added to x.

    This implementation is only for demonstration and testing purposes.

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
    """
    def __init__(self, n_features):
        self.size_psi = n_features + 1

    def psi(self, x, y):
        if y not in [-1, 1]:
            raise ValueError("y has to be either -1 or +1, got %s" % repr(y))
        return y * np.vstack([x, [1]])

    def inference(self, x, w):
        return np.sign(np.dot(x, w[:-1]) + w[-1])

    def loss_augmented_inference(self, x, y, w):
        return np.sign(np.dot(x, w[:-1]) + w[-1] - y)
