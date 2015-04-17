import numpy as np

from .base import StructuredModel
from .utils import crammer_singer_joint_feature


class BinaryClf(StructuredModel):
    """Formulate standard linear binary SVM in CRF framework.

    Inputs x are simply feature arrays, labels y are -1 or 1.

    Notes
    -----
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.

    It is also highly recommended to use n_jobs=1 in the learner when using
    this model. Trying to parallelize the trivial inference will slow
    the infernce down a lot!

    Parameters
    ----------
    n_features : int or None, default=None
        Number of features of inputs x.
        If None, it is inferred from data.
    """
    def __init__(self, n_features=None):
        self.size_joint_feature = n_features
        self.n_states = 2
        self.inference_calls = 0

    def initialize(self, X, Y):
        n_features = X.shape[1]
        if self.size_joint_feature is None:
            self.size_joint_feature = n_features
        elif self.size_joint_feature != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.size_joint_feature, n_features))

    def __repr__(self):
        return ("%s, n_features: %d"
                % (type(self).__name__, self.size_joint_feature))

    def joint_feature(self, x, y):
        """Compute joint feature vector of x and y.

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        y : int
            Class label, either +1 or -1.

        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).
        """
        if y not in [-1, 1]:
            raise ValueError("y has to be either -1 or +1, got %s" % repr(y))
        return y * x / 2.

    def batch_joint_feature(self, X, Y):
        return np.sum(X * np.array(Y)[:, np.newaxis] / 2., axis=0)

    def inference(self, x, w, relaxed=None):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, joint_feature(x, y)), i.e. best possible prediction.

        For a binary SVM, this is just sign(np.dot(w, x) + b))

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_joint_feature,)
            Parameters of the SVM.

        relaxed : ignored

        Returns
        -------
        y_pred : int
            Predicted class label.
        """
        self.inference_calls += 1
        return 2 * (np.dot(x, w) >= 0) - 1

    def batch_inference(self, X, w):
        return 2 * (np.dot(X, w) >= 0) - 1

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        """Loss-augmented inference for x and y using parameters w.

        Minimizes over y_hat:
        np.dot(joint_feature(x, y_hat), w) + loss(y, y_hat)
        which is just
        sign(np.dot(x, w) + b - y)

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_joint_feature,)
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


class MultiClassClf(StructuredModel):
    """Formulate linear multiclass SVM in C-S style in CRF framework.

    Inputs x are simply feature arrays, labels y are 0 to n_classes.

    Notes
    ------
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.

    It is also highly recommended to use n_jobs=1 in the learner when using
    this model. Trying to parallelize the trivial inference will slow
    the infernce down a lot!

    Parameters
    ----------
    n_features : int
        Number of features of inputs x.
        If None, it is inferred from data.

    n_classes : int, default=None
        Number of classes in dataset.
        If None, it is inferred from data.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    rescale_C : bool, default=False
        Whether the class-weights should be used to rescale C (liblinear-style)
        or just rescale the loss.
    """
    def __init__(self, n_features=None, n_classes=None, class_weight=None,
                 rescale_C=False):
        # one weight-vector per class
        self.n_states = n_classes
        self.n_features = n_features
        self.rescale_C = rescale_C
        self.class_weight = class_weight
        self.inference_calls = 0
        self._set_size_joint_feature()
        self._set_class_weight()

    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features]:
            self.size_joint_feature = self.n_states * self.n_features

    def initialize(self, X, Y):
        n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_classes = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_classes
        elif self.n_states != n_classes:
            raise ValueError("Expected %d classes, got %d"
                             % (self.n_states, n_classes))
        self._set_size_joint_feature()
        self._set_class_weight()

    def __repr__(self):
        return ("%s(n_features=%d, n_classes=%d)"
                % (type(self).__name__, self.n_features, self.n_states))

    def joint_feature(self, x, y, y_true=None):
        """Compute joint feature vector of x and y.

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : nd-array, shape=(n_features,)
            Input sample features.

        y : int
            Class label. Between 0 and n_classes.

        y_true : int
            True class label. Needed if rescale_C==True.


        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).
        """
        # put feature vector in the place of the weights corresponding to y
        result = np.zeros((self.n_states, self.n_features))
        result[y, :] = x
        if self.rescale_C:
            if y_true is None:
                raise ValueError("rescale_C is true, but no y_true was passed"
                                 " to joint_feature.")
            result *= self.class_weight[y_true]

        return result.ravel()

    def batch_joint_feature(self, X, Y, Y_true=None):
        result = np.zeros((self.n_states, self.n_features))
        if self.rescale_C:
            if Y_true is None:
                raise ValueError("rescale_C is true, but no y_true was passed"
                                 " to joint_feature.")
            for l in range(self.n_states):
                mask = Y == l
                class_weight = self.class_weight[Y_true[mask]][:, np.newaxis]
                result[l, :] = np.sum(X[mask, :] * class_weight, axis=0)
        else:
            # if we don't have class weights, we can use our efficient
            # implementation
            assert(X.shape[0] == Y.shape[0])
            assert(X.shape[1] == self.n_features)
            crammer_singer_joint_feature(X, Y, result)
        return result.ravel()

    def inference(self, x, w, relaxed=None, return_energy=False):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, joint_feature(x, y)), i.e. best possible prediction.

        For an unstructured multi-class model (this model), this
        can easily done by enumerating all possible y.

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_joint_feature,)
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
        np.dot(joint_feature(x, y_hat), w) + loss(y, y_hat)

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_joint_feature,)
            Weights that will be used for inference.

        Returns
        -------
        y_hat : int
            Label with highest sum of loss and score.
        """
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_states, -1), x)
        other_classes = np.arange(self.n_states) != y
        if self.rescale_C:
            scores[other_classes] += 1
        else:
            scores[other_classes] += self.class_weight[y]
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        scores = np.dot(X, w.reshape(self.n_states, -1).T)
        other_classes = (np.arange(self.n_states) != Y[:, np.newaxis])
        if self.rescale_C or self.uniform_class_weight:
            scores[other_classes] += 1
        else:
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
