import numpy as np


class StructuredProblem(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least psi and inference.
    """
    def __repr__(self):
        return ("%s, size_psi: %d"
                % (type(self).__name__, self.size_psi))

    def __init__(self):
        """Initialize the problem.
        Needs to set self.size_psi, the dimensionalty of the joint features for
        an instance with labeling (x, y).
        """
        self.size_psi = None

    def _check_size_w(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))

    def psi(self, x, y):
        # IMPLEMENT ME
        pass

    def batch_psi(self, X, Y):
        psi_ = np.zeros(self.size_psi)
        for x, y in zip(X, Y):
            psi_ += self.psi(x, y)
        return psi_

    def _loss_augmented_dpsi(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.psi(x_loss_augmented, y)
                - self.psi(x_loss_augmented, y_hat))

    def inference(self, x, w, relaxed=None):
        # IMPLEMENT ME
        pass

    def batch_inference(self, X, w, relaxed=None):
        # default implementation of batch inference
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, y, y_hat):
        # hamming loss:
        if isinstance(y_hat, tuple):
            return self.continuous_loss(y, y_hat[0])
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def max_loss(self, y):
        # maximum possible los on y for macro averages
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y])
        return y.size

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]
