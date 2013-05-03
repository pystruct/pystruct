import numpy as np

from .base import StructuredProblem
from .utils import crammer_singer_psi


class BinarySVMProblem(StructuredProblem):
    """Formulate standard linear binary SVM in CRF framework.

    Inputs x are simply feature arrays, labels y are -1 or 1.
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
    """
    def __init__(self, n_features):
        self.size_psi = n_features
        self.n_states = 2
        self.inference_calls = 0

    def __repr__(self):
        return ("%s, n_features: %d"
                % (type(self).__name__, self.size_psi))

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
        return y * x / 2.

    def batch_psi(self, X, Y):
        return np.sum(X * np.array(Y)[:, np.newaxis] / 2., axis=0)

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
        return np.sign(np.dot(x, w))

    def batch_inference(self, X, w):
        return np.sign(np.dot(X, w))

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
        return np.sign(np.dot(x, w) - y)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        return np.sign(np.dot(X, w) - Y)

    def batch_loss(self, Y, Y_hat):
        return Y != Y_hat


class CrammerSingerSVMProblem(StructuredProblem):
    """Formulate linear multiclass SVM in C-S style in CRF framework.

    Inputs x are simply feature arrays, labels y are 0 to n_classes.
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.

    n_classes : int, default=2
        Number of classes in dataset.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.
    """
    def __init__(self, n_features, n_classes=2, class_weight=None):
        # one weight-vector per class
        self.size_psi = n_classes * n_features
        self.n_states = n_classes
        self.n_features = n_features
        if class_weight is not None:
            if len(class_weight) != n_classes:
                raise ValueError("class_weight must have length n_classes or"
                                 " be None")
            class_weight = np.array(class_weight)
        else:
            class_weight = np.ones(n_classes)
        self.class_weight = class_weight
        self.inference_calls = 0

    def __repr__(self):
        return ("%s(n_features=%d, n_classes=%d)"
                % (type(self).__name__, self.n_features, self.n_states))

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
        # put feature vector in the place of the weights corresponding to y
        result = np.zeros((self.n_states, self.n_features))
        result[y, :] = x
        return result.ravel()

    def batch_psi(self, X, Y):
        out = np.zeros((self.n_states, self.n_features))
        #for l in xrange(self.n_states):
            #result[l, :] = np.sum(X[Y == l, :], axis=0)
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.n_features)
        crammer_singer_psi(X, Y, out)
        return out.ravel()

    def inference(self, x, w, relaxed=None, return_energy=False):
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
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_states, -1), x)
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def loss_augmented_inference(self, x, y, w, relaxed=None,
                                 return_energy=False):
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
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_states, -1), x)
        other_classes = np.arange(self.n_states) != y
        scores[other_classes] += self.class_weight[y]
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        scores = np.dot(X, w.reshape(self.n_states, -1).T)
        other_classes = (np.arange(self.n_states) != np.vstack(Y))
        scores[other_classes] += np.repeat(self.class_weight[Y],
                                           self.n_states - 1)
        return np.argmax(scores, axis=1)

    def batch_inference(self, X, w, relaxed=None):
        scores = np.dot(X, w.reshape(self.n_states, -1).T)
        return np.argmax(scores, axis=1)

    def batch_loss(self, Y, Y_hat):
        return self.class_weight[Y] * (Y != Y_hat)

    def loss(self, y, y_hat):
        return self.class_weight[y] * (y != y_hat)
