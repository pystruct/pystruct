import numpy as np

from .base import StructuredProblem

from IPython.core.debugger import Tracer
tracer = Tracer()


class BinarySVMProblem(StructuredProblem):
    """Formulate standard linear binary SVM in CRF framework.

    Inputs x are simply feature arrays, labels y are -1 or 1.
    An additional constant 1 feature (leading to a penalized bias/intercept) is
    added to x.

    Needless to say, this implementation is only for demonstration and testing
    purposes.

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
    """
    def __init__(self, n_features):
        self.size_psi = n_features + 1
        self.n_states = 2
        self.inference_calls = 0

    def psi(self, x, y):
        if y not in [-1, 1]:
            raise ValueError("y has to be either -1 or +1, got %s" % repr(y))
        return y * np.hstack([x, [1]])

    def inference(self, x, w, relaxed=None):
        self.inference_calls += 1
        return np.sign(np.dot(x, w[:-1]) + w[-1])

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        self.inference_calls += 1
        return np.sign(np.dot(x, w[:-1]) + w[-1] - y)

    def get_unary_weights(self, w):
        # this is here for an ugly hack and should be removed soon
        return np.array([])


class CrammerSingerSVMProblem(StructuredProblem):
    """Formulate linear multiclass SVM in C-S style in CRF framework.

    Inputs x are simply feature arrays, labels y are 0 to n_classes.
    An additional constant 1 feature (leading to a penalized bias/intercept) is
    added to x.

    Needless to say, this implementation is only for demonstration and testing
    purposes.

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
    """
    def __init__(self, n_features, n_classes=2):
        # one weight-vector per class, each with additional bias term
        self.size_psi = n_classes * (n_features + 1)
        self.n_states = n_classes
        self.inference_calls = 0

    def psi(self, x, y):
        if y not in range(self.n_states):
            raise ValueError("y has to be between 0 and %d, got %s."
                             % (self.n_states, repr(y)))
        # put feature vector in the place of the weights corresponding to y
        result = np.zeros(self.size_psi).reshape(self.n_states, -1)
        result[y, :] = np.hstack([x, [1]])
        return result.ravel()

    def inference(self, x, w, relaxed=None):
        scores = np.dot(w.reshape(self.n_states, -1), np.hstack([x, [1]]))
        return np.argmax(scores)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        if y not in range(self.n_states):
            raise ValueError("y has to be between 0 and %d, got %s."
                             % (self.n_states, repr(y)))
        scores = np.dot(w.reshape(self.n_states, -1), np.hstack([x, [1]]))
        return np.argmax(scores)

    def get_unary_weights(self, w):
        # this is here for an ugly hack and should be removed soon
        return np.array([])
