import numpy as np


class StructuredModel(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least psi and inference.
    """
    def __repr__(self):
        return ("%s, size_psi: %d"
                % (type(self).__name__, self.size_psi))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_psi, the dimensionalty of the joint features for
        an instance with labeling (x, y).
        """
        self.size_psi = None

    def _check_size_w(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))

    def initialize(self, X, Y):
        # set any data-specific parameters in the model
        pass

    def psi(self, x, y):
        raise NotImplementedError()

    def batch_psi(self, X, Y, Y_true=None):
        psi_ = np.zeros(self.size_psi)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                psi_ += self.psi(x, y, y_true)
        else:
            for x, y in zip(X, Y):
                psi_ += self.psi(x, y)
        return psi_

    def _loss_augmented_dpsi(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.psi(x_loss_augmented, y)
                - self.psi(x_loss_augmented, y_hat))

    def inference(self, x, w, relaxed=None):
        raise NotImplementedError()

    def batch_inference(self, X, w, relaxed=None):
        # default implementation of batch inference
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, x, y, y_hat, node_weights=None):
        # hamming loss:
        if isinstance(y_hat, tuple):
            if node_weights == None:
                return self.continuous_loss(x, y, y_hat[0])
            else:
                return self.continuous_loss(x, y, y_hat[0], node_weights)
        if node_weights == None:
            node_weights = 1
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat) * node_weights)
        return np.sum((y != y_hat) * node_weights)

    def batch_loss(self, X, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(x, y, y_hat) for x, y, y_hat in zip(X, Y, Y_hat)]

    def max_loss(self, x, y, node_weights=None):
        # maximum possible los on y for macro averages
        if hasattr(self, 'class_weight'):
            if node_weights == None:
                node_weights = 1
            return np.sum(self.class_weight[y] * node_weights)
        if node_weights == None:
            return y.size
        return sum(node_weights)

    def continuous_loss(self, x, y, y_hat, node_weights=None):
        # continuous version of the loss
        # y is the result of linear programming
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if node_weights == None:
            node_weights = 1
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result * node_weights)
        return np.sum(result * node_weights)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]

    def _set_class_weight(self):
        if not hasattr(self, 'size_psi'):
            # we are not initialized yet
            return

        if hasattr(self, 'n_labels'):
            n_things = self.n_labels
        else:
            n_things = self.n_states

        if self.class_weight is not None:

            if len(self.class_weight) != n_things:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(n_things)
            self.uniform_class_weight = True
