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
        """Compute joint feature vector of x and y.

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        y : int
            Class label, either +1 or -1.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).
        """
        if y not in [-1, 1]:
            raise ValueError("y has to be either -1 or +1, got %s" % repr(y))
        return y * np.hstack([x, [1]])

    def inference(self, x, w, relaxed=None):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, psi(x, y)), i.e. best possible prediction.

        For a binary SVM, this is just sign(np.dot(w, x) + b))

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_psi,)
            Parameters of the SVM.

        relaxed : ignored

        Returns
        -------
        y_pred : int
            Predicted class label.
        """
        self.inference_calls += 1
        return np.sign(np.dot(x, w[:-1]) + w[-1])

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        """Loss-augmented inference for x and y using parameters w.

        Minimizes over y_hat:
        np.dot(psi(x, y_hat), w) + loss(y, y_hat)
        which is just
        sign(np.dot(x, w) + b - y)

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_psi,)
            Weights that will be used for inference.

        Returns
        -------
        y_hat : int
            Label with highest sum of loss and score.
        """
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
        """Compute joint feature vector of x and y.

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        y : int
            Class label. Between 0 and n_classes.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).
        """
        if y not in range(self.n_states):
            raise ValueError("y has to be between 0 and %d, got %s."
                             % (self.n_states, repr(y)))
        # put feature vector in the place of the weights corresponding to y
        result = np.zeros(self.size_psi).reshape(self.n_states, -1)
        result[y, :] = np.hstack([x, [1]])
        return result.ravel()

    def inference(self, x, w, relaxed=None):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, psi(x, y)), i.e. best possible prediction.

        For an unstructured multi-class problem, this problem, this
        can easily done by enumerating all possible y.

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_psi,)
            Parameters of the SVM.

        relaxed : ignored

        Returns
        -------
        y_pred : int
            Predicted class label.
        """
        scores = np.dot(w.reshape(self.n_states, -1), np.hstack([x, [1]]))
        return np.argmax(scores)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        """Loss-augmented inference for x and y using parameters w.

        Minimizes over y_hat:
        np.dot(psi(x, y_hat), w) + loss(y, y_hat)

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_psi,)
            Weights that will be used for inference.

        Returns
        -------
        y_hat : int
            Label with highest sum of loss and score.
        """
        if y not in range(self.n_states):
            raise ValueError("y has to be between 0 and %d, got %s."
                             % (self.n_states, repr(y)))
        scores = np.dot(w.reshape(self.n_states, -1), np.hstack([x, [1]]))
        scores[y] -= 1
        return np.argmax(scores)

    def get_unary_weights(self, w):
        # this is here for an ugly hack and should be removed soon
        return np.array([])
