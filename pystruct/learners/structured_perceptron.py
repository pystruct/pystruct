import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from .ssvm import BaseSSVM


def inference(model, x, w):
    return model.inference(x, w)


class StructuredPerceptron(BaseSSVM):
    """Structured Perceptron training.

    Implements a simple structured perceptron without regularization.
    The structured perceptron approximately minimizes the zero-one loss.
    Therefore the learning does not take model.loss into account.
    It is just shown to illustrate the learning progress.
    As the perceptron learning is not margin-based, the model does not
    need to provide loss_augmented_inference.

    Parameters
    ----------
    model : StructuredModel
        Object containing model structure. Has to implement
        `loss`, `inference`.

    max_iter : int (default=100)
        Maximum number of passes over dataset to find constraints and update
        parameters.

    verbose : int (default=0)
        Verbosity

    batch: bool (default=False)
        Whether to do batch learning or online learning.

    Attributes
    ----------
    w : nd-array, shape=(model.psi,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.
    """
    def __init__(self, model, max_iter=100, verbose=0, batch=False):
        BaseSSVM.__init__(self, model, max_iter=max_iter, verbose=verbose)
        self.batch = batch

    def fit(self, X, Y):
        """Learn parameters using structured perceptron.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.
        """

        n_samples = len(X)
        size_psi = self.model.size_psi
        w = np.zeros(size_psi)
        loss_curve = []
        try:
            for iteration in xrange(self.max_iter):
                alpha = 1. / (1 + iteration)
                losses = 0
                if self.verbose:
                    print("iteration %d" % iteration)
                if self.batch:
                    Y_hat = (Parallel(n_jobs=self.n_jobs)(
                        delayed(inference)(self.model, x, w) for x, y in
                        zip(X, Y)))
                    for x, y, y_hat in zip(X, Y, Y_hat):
                        current_loss = self.model.loss(y, y_hat)
                        losses += current_loss
                        if current_loss:
                            w += alpha * (self.model.psi(x, y) -
                                          self.model.psi(x, y_hat))
                else:
                    # standard online update
                    for x, y in zip(X, Y):
                        y_hat = self.model.inference(x, w)
                        current_loss = self.model.loss(y, y_hat)
                        losses += current_loss
                        if current_loss:
                            w += alpha * (self.model.psi(x, y) -
                                          self.model.psi(x, y_hat))
                loss_curve.append(float(losses) / n_samples)
                if self.verbose:
                    print("avg loss: %f w: %s" % (loss_curve[-1], str(w)))
                    print("alpha: %f" % alpha)
        except KeyboardInterrupt:
            pass
        self.loss_curve_ = loss_curve
        self.w = w
