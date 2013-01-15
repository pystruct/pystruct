import numpy as np


class StructuredProblem(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least psi and inference.
    """

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

    def _loss_augmented_dpsi(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.psi(x_loss_augmented, y)
                - self.psi(x_loss_augmented, y_hat))

    def inference(self, x, w, relaxed=None):
        # IMPLEMENT ME
        pass

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_one_hot = np.zeros_like(y_hat)
        if y.ndim == 2:
            gx, gy = np.indices(y.shape)
            y_one_hot[gx, gy, y] = 1
        else:
            gx = np.indices(y.shape)
            y_one_hot[gx, y] = 1

        # all entries minus correct ones
        return np.prod(y.shape) - np.sum(y_one_hot * y_hat)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)
